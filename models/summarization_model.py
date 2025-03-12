# models/summarization_model.py
# Ahmed Essam

import torch
import torch.nn as nn

class SummarizationModel(nn.Module):
    """Summarization Model - Ahmed Essam"""
    def __init__(self, encoder, src_vocab_size, embed_size, num_classes, device):
        super(SummarizationModel, self).__init__()
        self.encoder = encoder
        self.fc = nn.Linear(embed_size, num_classes)
        self.device = device

    def forward(self, x, mask):
        encoder_output = self.encoder(x, mask)
        pooled_output = encoder_output.mean(dim=1)
        logits = self.fc(pooled_output)
        return logits.squeeze()