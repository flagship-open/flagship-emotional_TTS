import random
import torch
import wandb
import numpy as np
import sys
import os

from plotting_utils import plot_alignment_to_numpy, plot_spectrogram_to_numpy
from plotting_utils import plot_gate_outputs_to_numpy, plot_inference_gate_outputs_to_numpy

sys.path.append('/data3/sejikpark/.jupyter/workspace/desktop/tts/mellotron/')
sys.path.append('/data3/sejikpark/.jupyter/workspace/desktop/tts/waveglow/')

from denoiser import Denoiser
from text import sequence_to_text


class Tacotron2Logger():
    def __init__(self, prj_name, run_name, resume, inference_text, n_mel_channels,
                 logging_num, language_name, speakers_name, emotions_name):
        super(Tacotron2Logger, self).__init__()

        if resume:
            wandb.init(project=prj_name, resume=run_name)
        else:
            wandb.init(project=prj_name, name=run_name)
        wandb.config['hostname'] = os.uname()[1]

        self.waveglow = torch.load('/data3/sejikpark/.jupyter/workspace/desktop/tts/models/waveglow_256channels_ljs_v3.pt')['model'].cuda().eval()
        self.denoiser = Denoiser(self.waveglow).cuda().eval()
        
        self.logging_num = logging_num
        
        self.inference_text = inference_text
        self.language_name = language_name
        self.speakers_name = speakers_name
        self.emotions_name = emotions_name
        
        self.silence = self.mel2wav(torch.zeros(n_mel_channels, 3).cuda())

    def mel2wav(self, mel_outputs_postnet):
        with torch.no_grad():
            audio = self.denoiser(self.waveglow.infer(mel_outputs_postnet.unsqueeze(0), sigma=0.8), 0.01)[:, 0]
        data = audio[0].data[:].cpu().numpy()

        return data
    
    def get_mel_length(self, gate_outputs):
        gate_outputs = torch.cumsum(torch.sigmoid(gate_outputs) > 0.5, dim=1)
        return (gate_outputs == 0).sum(dim=1)
    
    def forward_attention_ratio(self, alignments, mel_len):      
        max_alignments = torch.argmax(alignments, dim=2)
        increasing_alignments = ((max_alignments[:,1:]-max_alignments[:,:-1]) >= 0)
        
        forward_attention_ratio = 0.0
        no_len = 0
        for i in range(increasing_alignments.shape[0]):
            if mel_len[i] > 0:
                forward_attention_ratio += increasing_alignments[i, :mel_len[i]].sum() / float(mel_len[i])
            else:
                no_len += 1
        if (i + 1 - no_len) == 0:
            forward_attention_ratio = torch.tensor(0.0)
        else:
            forward_attention_ratio = forward_attention_ratio / float(i+1-no_len)

        return forward_attention_ratio.cpu()

    def log_training(self, reduced_loss, log_loss, y_pred, grad_norm, learning_rate, duration,
                     iteration, epoch):
        
        _, _, gate_outputs, alignments, _ = y_pred
        mel_len = self.get_mel_length(gate_outputs)
        forward_att_ratio = self.forward_attention_ratio(alignments, mel_len)

        wandb_log = {"loss/train": reduced_loss,
                   "loss/train_mel_loss": log_loss[0],
                   "loss/train_gate_loss": log_loss[1],
                   "forward_att_ratio/train": forward_att_ratio,
                   "grad_norm/train": grad_norm,
                   "learning_rate/train": learning_rate,
                   "iter_duration/train": duration,
                   "epoch": epoch,
                   "iteration": iteration}
        
        if log_loss[2] is not None:
            wandb_log["loss/train_spk_adv_loss"] = log_loss[2]
        
        wandb.log(wandb_log)

    def log_validation(self, reduced_loss, log_loss, y, y_pred, x_inference, y_inference, iteration, sample_rate, epoch):
        wandb_log = dict()
        wandb_log['epoch'] = epoch
        wandb_log['iteration'] = iteration
        wandb_log['loss/validation'] = reduced_loss
        wandb_log['loss/validation_mel_loss'] = log_loss[0]
        wandb_log['loss/validation_gate_loss'] = log_loss[1]
        
        if log_loss[2] is not None:
            wandb_log['loss/validation_spk_adv_loss'] = log_loss[2]
        
        mel_targets, gate_targets, _ = y
        
        # validation
        _, mel_outputs, gate_outputs, alignments, _ = y_pred
        mel_len = self.get_mel_length(gate_outputs)
        forward_att_ratio = self.forward_attention_ratio(alignments, mel_len)
        wandb_log['forward_att_ratio/validation'] = forward_att_ratio
        
        idx = random.randint(0, alignments.size(0) - 1)
        np_alignment = plot_alignment_to_numpy(alignments[idx].data.cpu().numpy().T)
        wandb_log['alignment/validation'] = wandb.Image(np_alignment)
        np_mel_target = plot_spectrogram_to_numpy(mel_targets[idx].data.cpu().numpy())
        wandb_log['mel_spectrogram/validation_target'] = wandb.Image(np_mel_target)
        np_mel_predicted = plot_spectrogram_to_numpy(mel_outputs[idx].data.cpu().numpy())
        wandb_log['mel_spectrogram/validation_predicted'] = wandb.Image(np_mel_predicted)
        np_gate = plot_gate_outputs_to_numpy(
                gate_targets[idx].data.cpu().numpy(),
                torch.sigmoid(gate_outputs[idx]).data.cpu().numpy())
        wandb_log['gate/validation'] = wandb.Image(np_gate)
        
        if mel_len[idx] < 3:
            np_wav = self.silence
        else:
            mel = mel_outputs[idx, :, :mel_len[idx]]
            np_wav = self.mel2wav(mel)
        wandb_log['audio/validation'] = wandb.Audio(np_wav.astype(np.float32), sample_rate=sample_rate)
        
        # inference
        _, language_ids, speaker_ids, emotion_ids = x_inference
        _, mel_outputs_inf, gate_outputs_inf, alignments_inf = y_inference
        mel_len_inf = self.get_mel_length(gate_outputs_inf)
        forward_att_ratio_inf = self.forward_attention_ratio(alignments_inf, mel_len_inf)
        wandb_log['forward_att_ratio/inference'] = forward_att_ratio_inf
        
        spk_idx = random.randint(0, alignments_inf.size(0) - 1)
        np_alignment_inf = plot_alignment_to_numpy(alignments_inf[spk_idx].data.cpu().numpy().T)
        wandb_log['alignment/inference'] = wandb.Image(np_alignment_inf)
        np_mel_predicted_inf = plot_spectrogram_to_numpy(mel_outputs_inf[spk_idx].data.cpu().numpy())
        wandb_log['mel_spectrogram/inference'] = wandb.Image(np_mel_predicted_inf)
        np_gate_inf = plot_inference_gate_outputs_to_numpy(
                torch.sigmoid(gate_outputs_inf[spk_idx]).data.cpu().numpy())
        wandb_log['gate/inference'] = wandb.Image(np_gate_inf)
        
        for i in range(self.logging_num):
            if mel_len_inf[i] < 3:
                np_wav_inf = self.silence
            else:
                mel_inf = mel_outputs_inf[i, :, :mel_len_inf[i]]
                np_wav_inf = self.mel2wav(mel_inf)
            log_name = 'audio/inference_' + str(i) +'_'+ self.inference_text
            caption = self.language_name[language_ids[i]] + '_' + self.speakers_name[speaker_ids[i]] + '_' + self.emotions_name[emotion_ids[i]]
            wandb_log[log_name] = wandb.Audio(np_wav_inf.astype(np.float32), caption=caption, sample_rate=sample_rate)

        wandb.log(wandb_log)
