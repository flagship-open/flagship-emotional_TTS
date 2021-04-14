from flask import Flask, jsonify, request, Response, make_response, send_file
import argparse, librosa, os, sys
import soundfile as sf
import copy
import numpy as np
import time
import re
sys.path.append('./tts')
sys.path.append('./waveglow')
bb = time.time()

import torch
import torch.nn as nn
from torch.autograd import Variable

from config_dict import emo_dict, gen_dict, age_dict
from tts.model import load_model
from tts.text import cmudict, text_to_sequence, decompose_hangul
from tts.text.korean_normalization import txt_preprocessing_only_num
from tts.hparams import create_hparams
from waveglow.denoiser import Denoiser

from VocGAN.model.generator import ModifiedGenerator
from VocGAN.utils.hparams import HParam, load_hparam_str
from VocGAN.denoiser import Denoiser as VocGAN_Denoiser 

import zipfile
import torch.nn.functional as F

aa = time.time()
app = Flask(__name__)

parser = argparse.ArgumentParser(description='training script')
# generation option
parser.add_argument('--out_dir', type=str, default='generated', help='')
parser.add_argument('--init_cmudict', type=str,
                    default='./data/cmu_dictionary')
parser.add_argument('--init_model', type=str,
                    default='./models/allinone_37000')
parser.add_argument('--init_waveglow', type=str,
                    default='./models/waveglow_256channels_ljs_v3.pt')
parser.add_argument('--init_VocGAN', type=str,
                    default='./models/VocGAN_0364.pt')
parser.add_argument('--config_VocGAN', type=str,
                    default='./VocGAN/config/default.yaml')
parser.add_argument('--use_GST',
                    default=False,
                    action='store_true')
parser.add_argument('--enable_gpus', type=str,
                        default='0,1,2,3,4,5,6,7',
                        help='number of gpus')
parser.add_argument('--init_from', type=str, default='allinone_37000')
parser.add_argument('--port', type=int, default=8081, help='port number for api')
parser.add_argument('--details', default=True, action='store_false', help='attention and text will be saved as well')
new_args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=new_args.enable_gpus

hparams = create_hparams()
hparams.max_decoder_steps=2000
model = load_model(hparams).cuda().eval()
model.load_state_dict(torch.load(new_args.init_model)['state_dict'])

waveglow = torch.load(new_args.init_waveglow)['model'].cuda().eval()
denoiser = Denoiser(waveglow).cuda().eval()

hp = HParam(new_args.config_VocGAN)
checkpoint = torch.load(new_args.init_VocGAN)
VocGAN = ModifiedGenerator(hp.audio.n_mel_channels, hp.model.n_residual_layers,
                        ratios=hp.model.generator_ratio, mult = hp.model.mult,
                        out_band = hp.model.out_channels).cuda()
VocGAN.load_state_dict(checkpoint['model_g'])
VocGAN.eval(inference=True)
VocGAN_denoiser = VocGAN_Denoiser(VocGAN).cuda().eval()

arpabet_dict = cmudict.CMUDict(new_args.init_cmudict)

use_GST = new_args.use_GST
special_gst_mel = torch.load('./gst_mel/gst_mel').cuda()


def saveAttention(input_sentence, attentions, outpath):
    # Set up figure with colorbar
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    fig, ax = plt.subplots()
    cax = ax.matshow(attentions.cpu().numpy(), aspect='auto', origin='upper',cmap='gray')
    # fig.colorbar(cax)
    plt.ylabel('Encoder timestep', fontsize=18)
    plt.xlabel('Decoder timestep', fontsize=18)

    if input_sentence:
        plt.ylabel('Encoder timestep', fontsize=18)
        # Set up axes
        # ax.set_yticklabels([' '] + list(input_sentence) + [' '])
        # Show label at every tick
        # ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.tight_layout()
    plt.savefig(outpath)
    plt.close('all')


def generate(sentence, emotion, age, gender, intensity, file_name, padding=2.0, gst_mel=None, korean_num=True):
    """
        sentence(str): sentence to be synthesized
        emotion(int): emotion index
        age(int): age index
        gender(int): gender index
        intensity(float): intensty of the emotion
        padding(float): length of synthesized

    """
    # write to file        
    if korean_num == True:
        sentence = txt_preprocessing_only_num(sentence)
        
    text = decompose_hangul(sentence)
    # text = '. ' + text + ' .'
    inputs = torch.LongTensor(text_to_sequence(text, hparams.text_cleaners, arpabet_dict))[None, :].cuda()

    emo_id = Variable(torch.LongTensor(emotion), requires_grad=False)
    emo_id = emo_id.cuda()
    
    gen_id = Variable(torch.LongTensor(gender), requires_grad=False)
    gen_id = gen_id.cuda()
    
    x_inference = (inputs.long(), torch.tensor([0]).cuda(), gen_id, emo_id)
    
    start = time.time()
    model.decoder.max_decoder_steps = int(inputs.shape[-1] * 5 * padding) + 50
    with torch.no_grad():
        _, mel_outputs_postnet, _, alignments = model.inference(x_inference, logging=False, gst_mel=gst_mel)
        # wave = denoiser(waveglow.infer(mel_outputs_postnet, sigma=0.7), 0.01)[:, 0]
        wave = VocGAN_denoiser(VocGAN.inference(mel_outputs_postnet).squeeze(0), 0.01)[:, 0]
    wave = wave.squeeze()
    wave = wave[:-(hp.audio.hop_length*10)]
    wave_length = len(wave) / 22050
    generate_time = time.time() - start
    generation_speed = wave_length / generate_time
    print('{:.2f}s/s: it takes {:.2f}s for {:.2f}s wave.'.format(generation_speed, generate_time, wave_length))
    
    # amplifying
    wave = wave.squeeze().cpu().numpy()
    wave = librosa.core.resample(wave, 22050, 44100)
    wave = np.stack((wave, wave))
    maxv = 2 ** (16 - 1)
    wave /= max(abs(wave.max()), abs(wave.min()))
    wave = (wave * maxv * 0.95).astype(np.int16)

    #wave *= args.amp

    outpath1 = '%s/%s_%s_%s_i%.1f_%s.wav' % (new_args.out_dir, os.path.basename(new_args.init_from), emotion, gender, intensity, file_name)
    #librosa.output.write_wav(outpath1, wave, 44100)
    sf.write(outpath1, wave.T, 44100, format='WAV', endian='LITTLE', subtype='PCM_16')
    if new_args.details:
        outpath2 = '%s/%s_%s_%s_i%0.1f_%s.png' % (new_args.out_dir, os.path.basename(new_args.init_from), emotion, gender, intensity, file_name)
        outpath3 = '%s/%s_%s_%s_i%0.1f_%s.npy' % (new_args.out_dir, os.path.basename(new_args.init_from), emotion, gender, intensity, file_name)
        outpath4 = '%s/%s_%s_%s_i%0.1f_%s.txt' % (new_args.out_dir, os.path.basename(new_args.init_from), emotion, gender, intensity, file_name)
        alignments = alignments.transpose(1,2).squeeze()
        # first step 0 step attention
        alignments[0, 0] = 1.0
        # no_zero_index = (torch.cumsum((torch.argmax(alignments, dim=0) != 0), dim=0) != 0)
        # no_zero_index *= (torch.argmax(alignments, dim=0) == 0)
        # alignments[0, no_zero_index] = 0
        saveAttention(text, alignments, outpath2)
        np.save(outpath3, alignments.cpu().numpy())
        with open(outpath4, 'w') as f:
            f.write(sentence + '\n')
            f.write(text + '\n')
            #f.write(' '.join(['{:.2f}'.format(i*12.5) for i in range(wave_lengths[0])]) + '\n')
    # return outpath1

    def zipFiles(file_list):
        zippath = '%s/%s_%s_%s_i%0.1f_%s.zip' % (new_args.out_dir, os.path.basename(new_args.init_from), emotion, gender, intensity, file_name)
        zipf = zipfile.ZipFile(zippath, 'w', zipfile.ZIP_DEFLATED)
        for files in file_list:
            zipf.write(files)
        zipf.close()
        return zippath

    file_list = []
    file_list.append(outpath1)
    file_list.append(outpath3)
    file_list.append(outpath4)

    zippath = zipFiles(file_list)
    print('zippath', zippath)
    return zippath


@app.route('/', methods=['POST'])
def calc():
    print('request.form : {}'.format(request.form))
    sentence = request.form['sentence'] if 'sentence' in request.form.keys() else None
    emotion = request.form['emotion'] if 'emotion' in request.form.keys() else '10005'
    age = request.form['age'] if 'age' in request.form.keys() else '20003'
    gender = request.form['gender'] if 'gender' in request.form.keys() else '30002'
    intensity = request.form['intensity'] if 'intensity' in request.form.keys() else 1
    intensity = float(intensity)

    assert sentence is not None

    aa = time.time()

    emotion = [int(emo_dict[emotion])]
    age = [int(age_dict[age])]
    gender = [int(gen_dict[gender])]

    if len(sentence) > 80:
        file_name = copy.deepcopy(sentence[:80])
    else:
        file_name = copy.deepcopy(sentence)
    
    if use_GST:
        if '있어?' in sentence[-4:]:
            gst_mel = special_gst_mel
        else:
            gst_mel = None
    else:
        gst_mel = None
        
    sentence = sentence.replace('?', '.')
    sentence = sentence.replace('!', '.')
    if sentence[-1] != '.':
        sentence += '.'
        
    outpath = '%s/%s_%s_%s_i%.1f_%s.zip' % (new_args.out_dir, os.path.basename(new_args.init_from), emotion, gender, intensity, file_name)
    
    if not os.path.exists(outpath):
        filename = generate(sentence, emotion, age, gender, intensity, file_name, gst_mel=gst_mel)
    else:
        print('file already exists {}'.format(outpath))
        filename = outpath

    print('it takes {:.2f}s'.format(time.time() - aa))
    return send_file(filename, mimetype="zip", as_attachment=True, attachment_filename="generated.zip") 


if __name__ == '__main__':
    print('pre-loading takes {}s'.format(time.time() - bb))
    app.run(host='0.0.0.0', port=new_args.port)
