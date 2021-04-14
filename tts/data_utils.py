import random
import os
import re
import numpy as np
import torch
import torch.utils.data
import librosa

import layers
from utils import load_wav_to_torch, load_filepaths_and_text
from text import text_to_sequence, cmudict


class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio, text and speaker ids
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, audiopaths_and_text, hparams, style_ids=None):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.p_arpabet = hparams.p_arpabet

        self.cmudict = None
        if hparams.cmudict_path is not None:
            self.cmudict = cmudict.CMUDict(hparams.cmudict_path)

        self.style_ids = style_ids
        if style_ids is None:
            self.style_ids = self.create_style_lookup_table(self.audiopaths_and_text)

        random.seed(1234)
        random.shuffle(self.audiopaths_and_text)

    def create_style_lookup_table(self, audiopaths_and_text):
        language_ids = list()
        speaker_ids = list()
        emotion_ids = list()
        for x in audiopaths_and_text:
            cur_audiopaths = x[0].split('_')
            language_ids.append(cur_audiopaths[-6])
            speaker_ids.append(cur_audiopaths[-5])
            emotion_ids.append(cur_audiopaths[-3])
        language_ids = np.sort(np.unique(language_ids))
        speaker_ids = np.sort(np.unique(speaker_ids))
        emotion_ids = np.sort(np.unique(emotion_ids))
        language_table = {int(language_ids[i]): i for i in range(len(language_ids))}
        speaker_table = {int(speaker_ids[i]): i for i in range(len(speaker_ids))}
        emotion_table = {int(emotion_ids[i]): i for i in range(len(emotion_ids))}
        
        return (language_table, speaker_table, emotion_table)

    def get_data(self, audiopath_and_text):
        audiopath, text, _ = audiopath_and_text
        cur_audiopath = audiopath.split('_')
        language_id, speaker_id, emotion_id = self.get_style_id(cur_audiopath)
        
        mel, no_sound = self.get_mel(audiopath)
        if no_sound:
            text = ' '
        text = self.get_text(text)
        return (text, mel, language_id, speaker_id, emotion_id)

    def get_style_id(self, cur_audiopath):
        language_id = torch.IntTensor([self.style_ids[0][int(cur_audiopath[-6])]])
        speaker_id = torch.IntTensor([self.style_ids[1][int(cur_audiopath[-5])]])
        emotion_id = torch.IntTensor([self.style_ids[2][int(cur_audiopath[-3])]])
        
        return (language_id, speaker_id, emotion_id)

    def get_mel(self, filepath):
        audio, sampling_rate = load_wav_to_torch(filepath)
        if sampling_rate != self.stft.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(
                sampling_rate, self.stft.sampling_rate))
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        try:
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
            no_sound = False
        except:
            f = open("/data3/sejikpark/.jupyter/workspace/desktop/tts/mellotron/log.txt", 'a')
            f.write(filepath)
            f.write('\n')
            f.close()
            melspec = torch.zeros([80, 5])
            no_sound = True

        return melspec, no_sound

    def get_text(self, text):
        text_norm = torch.IntTensor(
            text_to_sequence(text, self.text_cleaners, self.cmudict, self.p_arpabet))
        return text_norm

    def __getitem__(self, index):
        return self.get_data(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded, gate padded and speaker ids
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        
        language_ids = torch.LongTensor(len(batch))
        speaker_ids = torch.LongTensor(len(batch))
        emotion_ids = torch.LongTensor(len(batch))

        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)
            language_ids[i] = batch[ids_sorted_decreasing[i]][2]
            speaker_ids[i] = batch[ids_sorted_decreasing[i]][3]
            emotion_ids[i] = batch[ids_sorted_decreasing[i]][4]
            
        return (text_padded, input_lengths, mel_padded, gate_padded,
                 output_lengths, language_ids, speaker_ids, emotion_ids)