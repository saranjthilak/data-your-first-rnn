# 🧠 Your First RNN — From Scratch with Python + NumPy

A hands‑on tutorial-driven project to build your first **Recurrent Neural Network (RNN)** from scratch using Python and NumPy. This RNN performs **character-level language classification**, predicting the language origin of names (e.g., detecting if "Rodrigo" is Portuguese).

---

## 🚀 Project Overview

- **Objective**: Implement a simple RNN from first principles—without high-level frameworks—to understand its inner workings (forward pass, hidden state, and BPTT).
- **Use Case**: Character-level classification: reads a person's name letter by letter and predicts their language of origin.

---

## 📦 Repository Structure
```
├── data/ # Sample name lists per language
├── models/ # Saved model parameters (optional)
├── notebooks/ # Jupyter notebooks for experimentation
│ ├── explore.ipynb # Data analysis & visualization
│ └── rnn_implementation.ipynb # Step-by-step RNN build
├── src/
│ ├── rnn.py # Core RNN implementation (forward, backprop)
│ ├── utils.py # Data loading, encoding, helper functions
│ └── train.py # Training loop using SGD and BPTT
├── requirements.txt # Project dependencies
└── README.md # Project overview and instructions
```

---

## 🛠️ Key Features

- **Character embeddings**: One-hot encoded characters as input vectors.
- **Hidden state propagation**: Maintains and updates hidden layer across time steps.
- **Back-Propagation Through Time (BPTT)**: Manual implementation of gradient propagation over sequences.
- **Cross-entropy loss**: Used with softmax over character predictions.
- **Training loop**: Mini-batch SGD, learning rate decay, and loss tracking.
- **Evaluation**: Accuracy on test examples and confusion matrix to analyze model performance.

---

## ⚙️ Installation

```bash
git clone https://github.com/saranjthilak/data-your-first-rnn.git
cd data-your-first-rnn
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

