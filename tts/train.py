import os
import time
import argparse
import math
import random
import wandb

import torch
from distributed import apply_gradient_allreduce
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from model import load_model
from data_utils import TextMelLoader, TextMelCollate
from loss_function import Tacotron2Loss
from text import cmudict, text_to_sequence, decompose_hangul
from logger import Tacotron2Logger
from hparams import create_hparams


def reduce_tensor(tensor, n_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n_gpus
    return rt


def init_distributed(hparams, n_gpus, rank, group_name):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    print("Initializing Distributed")

    # Set cuda device so everything is done on the right GPU.
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Initialize distributed communication
    dist.init_process_group(
        backend=hparams.dist_backend, init_method=hparams.dist_url,
        world_size=n_gpus, rank=rank, group_name=group_name)

    print("Done initializing distributed")

    
def prepare_directories_and_logger(hparams,
                                   prj_name, run_name, resume,
                                   output_directory,
                                   rank,
                                   inference_text, logging_num):
    
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
            
        inference_decomposed_text = decompose_hangul(inference_text)
        arpabet_dict = cmudict.CMUDict('data/cmu_dictionary')
        inputs = torch.LongTensor(text_to_sequence(inference_decomposed_text, hparams.text_cleaners, arpabet_dict))[None, :].repeat(logging_num, 1)
        languages_ids = torch.zeros(logging_num).long()
        speaker_ids = torch.zeros(logging_num).long()
        emotion_ids = torch.zeros(logging_num).long()
        
        x_inference = (inputs.cuda(), languages_ids.cuda(), speaker_ids.cuda(), emotion_ids.cuda())
        
        logger = Tacotron2Logger(prj_name, run_name, resume, inference_text, hparams.n_mel_channels, logging_num,
                                hparams.languages_name, hparams.speakers_name, hparams.emotions_name)
    else:
        logger = None
        x_inference = None
        
    return logger, x_inference


def prepare_dataloaders(hparams):
    # Get data, data loaders and collate function ready
    trainset = TextMelLoader(hparams.training_files, hparams)
    valset = TextMelLoader(hparams.validation_files, hparams,
                           style_ids=trainset.style_ids)
    collate_fn = TextMelCollate(hparams.n_frames_per_step)

    if hparams.distributed_run:
        train_sampler = DistributedSampler(trainset)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_loader = DataLoader(trainset, num_workers=1, shuffle=shuffle,
                              sampler=train_sampler,
                              batch_size=hparams.batch_size, pin_memory=False,
                              drop_last=True, collate_fn=collate_fn)
    return train_loader, valset, collate_fn, train_sampler


def warm_start_model(checkpoint_path, model, ignore_layers, freeze_layers, partial_use=False, freeze_pre=False):
    assert os.path.isfile(checkpoint_path)
    print("Warm starting from checkpoint {}".format(checkpoint_path))
    
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_dict = model.state_dict()
    diff_model_dict = checkpoint_dict['state_dict']
    for k, v in model_dict.items():
        if k in diff_model_dict.keys():
            if v.shape == diff_model_dict[k].shape:
                if freeze_pre:
                    freeze_layers.append(k)
                model_dict[k] = diff_model_dict[k]
            elif partial_use:
                # using partial part
                if len(model_dict[k].shape) == len(diff_model_dict[k].shape):
                    k_shape = list()
                    for i, s in enumerate(model_dict[k].shape):
                        k_shape.append(min(s, diff_model_dict[k].shape[i]))
                if len(k_shape) == 1:
                    model_dict[k][:k_shape[0]] = diff_model_dict[k][:k_shape[0]]
                elif len(k_shape) == 2:
                    model_dict[k][:k_shape[0],:k_shape[1]] = diff_model_dict[k][:k_shape[0],:k_shape[1]]
                elif len(k_shape) == 3:
                    model_dict[k][:k_shape[0],:k_shape[1],:k_shape[2]] = diff_model_dict[k][:k_shape[0],:k_shape[1],:k_shape[2]]
    
    if len(ignore_layers) > 0:
        model_dict = {k: v for k, v in model_dict.items()
                    if k not in ignore_layers}
        dummy_dict = model.state_dict()
        dummy_dict.update(model_dict)
        model_dict = dummy_dict

    model.load_state_dict(model_dict)
    if len(freeze_layers) > 0:
        for k, v in model.named_parameters():
            if k in freeze_layers:
                v.requires_grad = False

    return model


def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath))
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)
    
    
def wav_logging_select(rank, x_inference, style_logging_len, logging_num):
    if rank == 0:
        inputs, languages_ids, speaker_ids, emotion_ids = x_inference
        
        for i in range(logging_num):
            languages_ids[i] = random.randint(0, style_logging_len[0] - 1)
            speaker_ids[i] = random.randint(0, style_logging_len[1] - 1)
            emotion_ids[i] = random.randint(0, style_logging_len[2] - 1)
                
        x_inference = (inputs, languages_ids, speaker_ids, emotion_ids)
    
    return x_inference
    

def validate(model, criterion, valset, iteration, batch_size, n_gpus,
             collate_fn, logger, distributed_run, rank,
             sample_rate, epoch, x_inference):
    """Handles all the validation scoring and printing"""
    valid_start = time.perf_counter()
    model.eval()
    with torch.no_grad():
        val_sampler = DistributedSampler(valset) if distributed_run else None
        val_loader = DataLoader(valset, sampler=val_sampler, num_workers=1,
                                shuffle=False, batch_size=batch_size,
                                pin_memory=False, collate_fn=collate_fn)
        
        val_loss = 0.0
        for i, batch in enumerate(val_loader):
            x, y = model.parse_batch(batch)
            y_pred = model(x)
            loss, log_loss = criterion(y_pred, y)
            if distributed_run:
                reduced_val_loss = reduce_tensor(loss.data, n_gpus).item()
            else:
                reduced_val_loss = loss.item()
            val_loss += reduced_val_loss
            
        val_loss = val_loss / (i + 1)
        
    if rank == 0:
        with torch.no_grad():
            y_inference = model.inference(x_inference)
        logger.log_validation(val_loss, log_loss, y, y_pred, x_inference, y_inference, iteration, sample_rate, epoch)
        valid_duration = time.perf_counter() - valid_start
        print("Validation loss {} {:9f} {:.2f}s".format(iteration, reduced_val_loss, valid_duration))
    model.train()

    
def train(prj_name, run_name, resume,
          output_directory, checkpoint_path, warm_start, partial_use, freeze_pre,
          n_gpus, rank, group_name,
          inference_text, logging_num, hparams):
    """
    Training and validation logging results to tensorboard and stdout

    Params
    ------
    output_directory (string): directory to save checkpoints
    log_directory (string) directory to save tensorboard logs
    checkpoint_path(string): checkpoint path
    n_gpus (int): number of gpus
    rank (int): rank of current gpu
    hparams (object): comma separated list of "name=value" pairs.
    """
    if hparams.distributed_run:
        init_distributed(hparams, n_gpus, rank, group_name)

    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)

    model = load_model(hparams)
    learning_rate = hparams.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=hparams.weight_decay)

    criterion = Tacotron2Loss()
    
    logger, x_inference = prepare_directories_and_logger(hparams,
                                                         prj_name, run_name, resume,
                                                         output_directory, rank,
                                                        inference_text, logging_num)
    style_logging_len = (len(hparams.languages_name), len(hparams.speakers_name), len(hparams.emotions_name))
    
    if hparams.fp16_run:
        from apex import amp
        model, optimizer = amp.initialize(
            model, optimizer, opt_level='O2')

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)
    
    if rank == 0:
        wandb.watch(model)
        
    train_loader, valset, collate_fn, train_sampler = prepare_dataloaders(hparams)

    # Load checkpoint if one exists
    iteration = 0
    epoch_offset = 0
    if checkpoint_path is not None:
        if warm_start:
            model = warm_start_model(
                checkpoint_path, model, hparams.ignore_layers, hparams.freeze_layers, partial_use=partial_use, freeze_pre=freeze_pre)
        else:
            model, optimizer, _learning_rate, iteration = load_checkpoint(
                checkpoint_path, model, optimizer)
            if hparams.use_saved_learning_rate:
                learning_rate = _learning_rate
            iteration += 1
            epoch_offset = max(0, int(iteration / len(train_loader)))

    model.train()
    is_overflow = False
    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in range(epoch_offset, hparams.epochs):
        print("Epoch: {}".format(epoch))
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for i, batch in enumerate(train_loader):
            start = time.perf_counter()
            if iteration > 0 and iteration % hparams.learning_rate_anneal == 0:
                learning_rate = max(
                    hparams.learning_rate_min, learning_rate * 0.5)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate

            model.zero_grad()
            x, y = model.parse_batch(batch)
            y_pred = model(x)

            loss, log_loss = criterion(y_pred, y)
            if hparams.distributed_run:
                reduced_loss = reduce_tensor(loss.data, n_gpus).item()
            else:
                reduced_loss = loss.item()

            if hparams.fp16_run:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if hparams.fp16_run:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    amp.master_params(optimizer), hparams.grad_clip_thresh)
                is_overflow = math.isnan(grad_norm)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), hparams.grad_clip_thresh)

            optimizer.step()

            if not is_overflow and rank == 0:
                duration = time.perf_counter() - start
                print("Train loss {} {:.6f} Grad Norm {:.6f} {:.2f}s/it".format(
                    iteration, reduced_loss, grad_norm, duration))
                with torch.no_grad():
                    logger.log_training(
                    reduced_loss, log_loss, y_pred, grad_norm, learning_rate, duration, iteration, epoch)

            if not is_overflow and (iteration % hparams.iters_per_checkpoint == 0):
                validate(model, criterion, valset, iteration,
                        hparams.batch_size, n_gpus, collate_fn, logger,
                        hparams.distributed_run, rank,
                        hparams.sampling_rate, epoch,
                        wav_logging_select(rank, x_inference, style_logging_len, logging_num))
                if rank == 0:
                    checkpoint_path = os.path.join(
                        output_directory, "checkpoint_{}".format(iteration))
                    save_checkpoint(model, optimizer, learning_rate, iteration,
                                    checkpoint_path)

            iteration += 1


if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--prj_name', default='tacotron2_flagship', type=str,
                        help='give a project name for this running')
    parser.add_argument('--run_name', default='Emotion_ETRI_LJS_LibriTTS_KNTTS_KETTS',type=str,
                        help='give a distinct name for this running')
    parser.add_argument('--wandb_resume',
                        default=False, action='store_true',
                        help='resume wandb')
    
    parser.add_argument('-o', '--output_directory', type=str,
                        default="outdir/Emotion_ETRI_LJS_LibriTTS_KNTTS_KETTS",
                        help='directory to save checkpoints')
    parser.add_argument('-c', '--checkpoint_path', type=str,
                        required=False,
                        help='checkpoint path')
    parser.add_argument('--warm_start',
                        action='store_true',
                        help='load model weights only, ignore specified layers')
    parser.add_argument('--partial_use',
                        action='store_true',
                        help='warm up include paritally equal')
    parser.add_argument('--freeze_pre',
                        action='store_true',
                        help='freeze pre_model parameters, not partial')
    
    parser.add_argument('--n_gpus', type=int,
                        required=False,
                        help='number of gpus')
    parser.add_argument('--enable_gpus', type=str,
                        default='0,1,2,3,4,5,6,7',
                        help='number of gpus')
    parser.add_argument('--rank', type=int,
                        default='0',
                        help='rank of current gpu')
    parser.add_argument('--group_name', type=str,
                        required=False,
                        help='Distributed group name')
    
    parser.add_argument('--inference_text', type=str,
                        default='안녕하세요. 저는 감정을 담아 말하는 음성합성기입니다.',
                         help='Text for inference')
    parser.add_argument('--logging_num', type=int,
                        default=7,
                         help='wav logging num for one checkpoint')
    parser.add_argument('--hparams', type=str,
                        required=False,
                        help='comma separated name=value pairs')

    args = parser.parse_args()
    hparams = create_hparams(args.hparams)
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=args.enable_gpus
    
    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark
    
    print("FP16 Run:", hparams.fp16_run)
    print("Dynamic Loss Scaling:", hparams.dynamic_loss_scaling)
    print("Distributed Run:", hparams.distributed_run)
    print("cuDNN Enabled:", hparams.cudnn_enabled)
    print("cuDNN Benchmark:", hparams.cudnn_benchmark)
            
    train(args.prj_name, args.run_name, args.wandb_resume,
          args.output_directory, args.checkpoint_path, args.warm_start, args.partial_use, args.freeze_pre,
          args.n_gpus, args.rank, args.group_name,
          args.inference_text, args.logging_num, hparams)
