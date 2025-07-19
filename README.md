# 🧠 RNN — From Scratch with Python + NumPy

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
```
## 🔄 Usage
1. Prepare Data
Place your .txt files for each language in the data/ folder (e.g. English.txt, Italian.txt).

2. Train the Model
```bash
Copy
Edit
python src/train.py \
  --data_dir data/ \
  --hidden_dim 128 \
  --learning_rate 0.005 \
  --epochs 10
```
3. Monitor Training
Check the loss curve and accuracy output:
```
python-repl
Copy
Edit
Epoch 1/10 — Loss: 3.12 — Accuracy: 12.5%
```
4. Test & Evaluate
```bash
Copy
Edit
python src/train.py --mode test --model_path models/rnn_epoch10.pkl
Generates metrics and a confusion matrix to assess performance.
```
## 🧠 How It Works
One-hot encoding of names as sequence of character vectors.

At each time step t:

hidden_t = tanh(U·x_t + W·hidden_{t-1})

output_t = softmax(V·hidden_t)

Loss: Sum of cross-entropy across time steps.

BPTT: Gradients are computed by unrolling through time and applying chain rule.

Parameter updates: Performed via SGD with optional learning rate decay.

## 📊 Evaluation / Results
Training and validation loss decreased consistently.

Test accuracy ranges between 35–45% depending on data and training length.

Confusion matrix reveals common confusions (e.g. Spanish ↔ Portuguese).

This simple RNN forms a solid foundation for further enhancements like LSTM, GRU, or using PyTorch.



