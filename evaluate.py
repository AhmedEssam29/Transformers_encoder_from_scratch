# evaluate.py
# Ahmed Essam

import torch
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
file_path = "data/text1.txt"
inputs, labels, vocab = preprocess_text(file_path)
dataset = TextDataset(inputs, labels)
dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

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

# Evaluation function
def evaluate(model, dataloader, device):
    """Evaluate model performance - Ahmed Essam"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_inputs, batch_labels in dataloader:
            batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
            outputs = model(batch_inputs, mask=None)
            predicted = torch.round(torch.sigmoid(outputs.squeeze()))
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

    accuracy = correct / total
    print(f"Accuracy: {accuracy:.4f}")
    return accuracy

# Evaluate the model
evaluate(model, dataloader, device)