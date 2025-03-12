# train.py
# Ahmed Essam

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.transformer_encoder import TransformerEncoder
from models.summarization_model import SummarizationModel
from utils.preprocessing import preprocess_text, TextDataset

# Hyperparameters
embed_size = 64
heads = 8
num_layers = 3
forward_expansion = 4
dropout = 0.1
num_classes = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocess data
file_path = "data/New Text Document (2).txt"
inputs, labels, vocab = preprocess_text(file_path)
dataset = TextDataset(inputs, labels)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Initialize model
encoder = TransformerEncoder(
    src_vocab_size=len(vocab),
    embed_size=embed_size,
    num_layers=num_layers,
    heads=heads,
    device=device,
    forward_expansion=forward_expansion,
    dropout=dropout,
    max_length=inputs.shape[1]
)
model = SummarizationModel(
    encoder=encoder,
    src_vocab_size=len(vocab),
    embed_size=embed_size,
    num_classes=num_classes,
    device=device
).to(device)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch_inputs, batch_labels in dataloader:
        batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)

        # Forward pass
        outputs = model(batch_inputs, mask=None)
        loss = criterion(outputs, batch_labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(dataloader):.4f}")