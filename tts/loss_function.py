import torch
from torch import nn


class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()

    def forward(self, model_output, targets):
        mel_target, gate_target, speaker_ids = targets
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        speaker_ids.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, _, spk_adv_out = model_output
        gate_out = gate_out.view(-1, 1)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + \
            nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
        
        if spk_adv_out is not None:
            spk_target = torch.zeros(spk_adv_out.size(0), spk_adv_out.size(2)).cuda() # spk_adv_out.zero_()
            spk_target.requires_grad = False
            spk_target = spk_target.scatter_(1,speaker_ids[:,None],1)[:, None, :].repeat(1,spk_adv_out.size(1),1)
        
            spk_adv_loss = nn.BCEWithLogitsLoss()(spk_adv_out, spk_target)
        
            return mel_loss + gate_loss + spk_adv_loss, (mel_loss, gate_loss, spk_adv_loss)
        else:
            spk_adv_loss = None
            
            return mel_loss + gate_loss, (mel_loss, gate_loss, spk_adv_loss)
            
            
