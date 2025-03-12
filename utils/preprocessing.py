# utils/preprocessing.py
# Ahmed Essam

import re
import torch
from torch.utils.data import Dataset

def preprocess_text(file_path):
    """Preprocess text file into sentences and labels - Ahmed Essam"""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Split text into sentences
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)

    # For demonstration, let's assume the first 3 sentences are important
    labels = [1 if i < 3 else 0 for i in range(len(sentences))]

    # Tokenize sentences (dummy tokenization for demonstration)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for sentence in sentences:
        for word in sentence.split():
            if word not in vocab:
                vocab[word] = len(vocab)

    # Convert sentences to token IDs
    tokenized_sentences = [[vocab.get(word, vocab["<UNK>"]) for word in sentence.split()] for sentence in sentences]

    # Pad sequences to the same length
    max_len = max(len(sentence) for sentence in tokenized_sentences)
    padded_sentences = [sentence + [vocab["<PAD>"]] * (max_len - len(sentence)) for sentence in tokenized_sentences]

    # Convert to PyTorch tensors
    inputs = torch.tensor(padded_sentences, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.float)

    return inputs, labels, vocab


# utils/preprocessing.py
class TextDataset(Dataset):
    """Custom Dataset for text data - Ahmed Essam"""
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]