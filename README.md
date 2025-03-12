# Text Summarization Using Transformer Encoder from Scratch

## Author: Ahmed Essam Abd Elgwad

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)  
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)  
![License](https://img.shields.io/badge/License-MIT-green)  

A **PyTorch-based Transformer Encoder model** for **extractive text summarization**. This project focuses on identifying and extracting the most important sentences from a given text.

---

## 📌 Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Results](#results)
6. [Project Structure](#project-structure)
7. [Contributing](#contributing)
8. [License](#license)
9. [Acknowledgements](#acknowledgements)
10. [Contact](#contact)

---

## 🔍 Overview
Text summarization is the process of condensing a large body of text into a concise version while retaining key information. This project implements a **Transformer Encoder-based model** to perform **extractive summarization**, where sentences are classified as **important (1)** or **not important (0)** for inclusion in the summary.

### ✨ Key Components
- **Transformer Encoder** – Captures contextual relationships between words.
- **Mean Pooling** – Aggregates token-level outputs into a sentence-level representation.
- **Binary Classification** – Determines the importance of each sentence.
- **PyTorch Implementation** – Built from scratch for transparency and flexibility.

---

## 🚀 Features
✅ **Custom Transformer Encoder** – No external pre-trained models used.  
✅ **Extractive Summarization** – Selects key sentences instead of generating new text.  
✅ **Modular Design** – Clean, reusable, and extendable code structure.  
✅ **GPU Support** – Runs efficiently on CUDA-enabled GPUs.  
✅ **Easy Integration** – Plug-and-play architecture for further improvements.  

---

## 🛠 Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/text-summarization-project.git
   cd text-summarization-project
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 📌 Usage
### 🔹 Training the Model
To train the model, run:
```bash
python train.py
```

### 🔹 Evaluating the Model
To evaluate the model, run:
```bash
python evaluate.py
```

### 🔹 Customizing the Input
- Place your text file inside the `data/` directory.
- Update the `file_path` variable in `train.py` and `evaluate.py` to point to your input file.


---

## 📁 Project Structure
```plaintext
text_summarization_project/
│
├── data/
│   └── text1.txt                 # Sample input text file
│
├── models/
│   ├── __init__.py                # Package initializer
│   ├── transformer_encoder.py     # Transformer Encoder implementation
│   └── summarization_model.py     # Summarization model script
│
├── utils/
│   ├── __init__.py                # Package initializer
│   ├── preprocessing.py           # Text preprocessing functions
│   └── evaluation.py              # Evaluation functions
│
├── train.py                        # Model training script
├── evaluate.py                     # Model evaluation script
├── requirements.txt                # List of dependencies
├── README.md                       # Project documentation
└── LICENSE                         # License details
```

---

## Training Results

### Overview
The training process was conducted over 10 epochs, with the model being trained on batches of data. The loss was logged at each batch to monitor the training progress.

### Training Loss
The following table summarizes the loss values recorded at the end of each epoch:

| Epoch | Loss (Last Batch) |
|-------|--------------------|
| 1     | 0.029179930686950684 |
| 2     | 3.5010061264038086 |
| 3     | 1.2415379285812378 |
| 4     | 1.2143275737762451 |
| 5     | 1.410714864730835 |
| 6     | 1.1609588861465454 |
| 7     | 1.4346656799316406 |
| 8     | 1.0334097146987915 |
| 9     | 1.5288965702056885 |
| 10    | 1.0628677606582642 |

### Observations
- The training loss fluctuated significantly across epochs, indicating variability in the training process.
- The lowest loss recorded was 0.029179930686950684 at the end of epoch 1.
- The highest loss recorded was 3.5010061264038086 at the end of epoch 2.
- The model showed some improvement in loss reduction towards the later epochs, but there were still significant fluctuations.

### Conclusion
The training process demonstrated the model's ability to learn from the data, but the variability in loss suggests that further tuning of hyperparameters and training strategies may be necessary to achieve more stable and lower loss values.

### Future Work
- **Hyperparameter Tuning**: Experiment with different learning rates, batch sizes, and other hyperparameters to stabilize the training process.
- **Data Augmentation**: Consider augmenting the training data to provide more diverse examples for the model to learn from.
- **Model Architecture**: Explore modifications to the model architecture to improve its learning capabilities and reduce loss.


---


## 🤝 Contributing
Contributions are welcome! To contribute:
1. **Fork the repository.**
2. **Create a new branch:**  
   ```bash
   git checkout -b feature/YourFeature
   ```
3. **Commit your changes:**  
   ```bash
   git commit -m "Add new feature"
   ```
4. **Push to the branch:**  
   ```bash
   git push origin feature/YourFeature
   ```
5. **Open a Pull Request.**

---

## 📜 License
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 🙌 Acknowledgements
- **[PyTorch](https://pytorch.org/)** – Deep learning framework used for model implementation.
- **[Attention is All You Need](https://arxiv.org/abs/1706.03762)** – Transformer model architecture paper.
- **Ahmed Essam** – Project development and documentation.

---

## 📧 Contact
For questions or feedback, feel free to reach out:
📩 **Email:** ahmedessam2996@gamil.com  
🔗 **GitHub:** [Ahmed Essam](https://github.com/AhmedEssam29)  

---

📌 **If you find this project helpful, don't forget to ⭐ the repository!** 🚀

