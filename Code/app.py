import streamlit as st
import pickle
import torch
import numpy as np
import pandas as pd

from transformers import AutoTokenizer, AutoModelForSequenceClassification

import torch.nn as nn
import torch.nn.functional as F

class SimpleTokenizer:
    """
    Minimal word-level tokenizer similar to Keras' Tokenizer.
    0 = PAD, 1 = OOV.
    """

    def __init__(self, num_words: int = 50000, oov_token: str = "<OOV>"):
        self.num_words = num_words
        self.oov_token = oov_token
        self.word_index = {}
        self.oov_index = 1

    def _tokenize(self, text: str):
        import re
        text = text.lower()
        return re.findall(r"\b\w+\b", text)

    def fit_on_texts(self, texts):
        from collections import Counter
        counter = Counter()
        for t in texts:
            tokens = self._tokenize(str(t))
            counter.update(tokens)

        most_common = counter.most_common(max(self.num_words - 2, 0))
        self.word_index = {w: i + 2 for i, (w, _) in enumerate(most_common)}
        self.word_index[self.oov_token] = self.oov_index

    def texts_to_sequences(self, texts):
        sequences = []
        for t in texts:
            tokens = self._tokenize(str(t))
            seq = [self.word_index.get(tok, self.oov_index) for tok in tokens]
            sequences.append(seq)
        return sequences
# ============================================================
# LOAD ARTIFACTS
# ============================================================

@st.cache_resource
def load_label_encoder():
    with open("label_encoder.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_baseline():
    with open("baseline_model.pkl", "rb") as f:
        baseline = pickle.load(f)
    with open("tfidf.pkl", "rb") as f:
        tfidf = pickle.load(f)
    return baseline, tfidf

@st.cache_resource
def load_lstm_model():
    with open("lstm_tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    # Define LSTM class again exactly as in training
    class FineTunedLSTM(nn.Module):
        def __init__(self, vocab_size, embedding_matrix, embedding_dim, lstm_units, num_classes):
            super().__init__()
            self.embedding = nn.Embedding.from_pretrained(
                embedding_matrix,
                freeze=False,
                padding_idx=0,
            )
            self.lstm = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=lstm_units,
                batch_first=True,
                bidirectional=True,
                num_layers=2,
                dropout=0.3,
            )
            self.attn = nn.Linear(lstm_units * 2, 1)
            self.layernorm = nn.LayerNorm(lstm_units * 2)
            self.dropout = nn.Dropout(0.4)
            self.fc = nn.Linear(lstm_units * 2, num_classes)

        def forward(self, x):
            x = self.embedding(x)
            lstm_out, _ = self.lstm(x)
            weights = torch.softmax(self.attn(lstm_out), dim=1)
            context = (weights * lstm_out).sum(dim=1)
            context = self.layernorm(context)
            context = self.dropout(context)
            return self.fc(context)

    # Load embedding matrix size from tokenizer
    vocab_size = len(tokenizer.word_index) + 1
    embedding_dim = 100
    embedding_matrix = torch.randn(vocab_size, embedding_dim)

    num_classes = len(load_label_encoder().classes_)
    model = FineTunedLSTM(vocab_size, embedding_matrix, embedding_dim, 256, num_classes)

    model.load_state_dict(torch.load("best_lstm.pt", map_location="cpu"))
    model.eval()

    return model, tokenizer


@st.cache_resource
def load_bert():
    model = AutoModelForSequenceClassification.from_pretrained("bert_finetuned")
    tokenizer = AutoTokenizer.from_pretrained("bert_finetuned")
    model.eval()
    return model, tokenizer


# ============================================================
# PREDICTION HELPERS
# ============================================================

def baseline_predict(text, model, tfidf, le):
    X = tfidf.transform([text])
    pred_id = model.predict(X)[0]
    pred_label = le.inverse_transform([pred_id])[0]
    return pred_label


def lstm_predict(text, model, tokenizer, le, max_len=400):
    tokens = tokenizer.texts_to_sequences([text])[0]
    if len(tokens) < max_len:
        padded = [0] * (max_len - len(tokens)) + tokens
    else:
        padded = tokens[-max_len:]

    x = torch.tensor([padded], dtype=torch.long)
    logits = model(x)
    probs = F.softmax(logits, dim=1).detach().numpy()[0]
    pred_id = int(np.argmax(probs))
    return le.inverse_transform([pred_id])[0], probs


def bert_predict(text, model, tokenizer, le, max_length=256):
    inputs = tokenizer(text, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
    logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1).detach().numpy()[0]
    pred_id = int(np.argmax(probs))
    return le.inverse_transform([pred_id])[0], probs


# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(page_title="Presidential NLP Model Predictor", layout="wide")

st.title("🇺🇸 Presidential Speech NLP – Live Prediction App")

st.markdown("""
Enter **any political speech text**, and the app will predict which president's
style it resembles using:
- **Baseline TF-IDF + Logistic Regression**
- **LSTM + Attention (GloVe)**
- **DistilBERT Fine-Tuned**
""")

text_input = st.text_area("Enter speech text here:", height=180)

if st.button("Predict"):
    if len(text_input.strip()) == 0:
        st.warning("Please enter some text.")
        st.stop()

    le = load_label_encoder()

    # Load models
    baseline, tfidf = load_baseline()
    lstm_model, lstm_tokenizer = load_lstm_model()
    bert_model, bert_tokenizer = load_bert()

    st.subheader("📌 Predictions")

    # BASELINE
    base_pred = baseline_predict(text_input, baseline, tfidf, le)
    st.write(f"**Baseline:** {base_pred}")

    # LSTM
    lstm_pred, lstm_probs = lstm_predict(text_input, lstm_model, lstm_tokenizer, le)
    st.write(f"**LSTM:** {lstm_pred}")

    # BERT
    bert_pred, bert_probs = bert_predict(text_input, bert_model, bert_tokenizer, le)
    st.write(f"**BERT:** {bert_pred}")

    # Show probability tables
    st.subheader("🔍 Model Probability Distributions")
    df = pd.DataFrame({
        "Class": le.classes_,
        "LSTM Probability": lstm_probs,
        "BERT Probability": bert_probs,
    })
    st.dataframe(df)

# Show confusion matrices
st.markdown("---")
st.subheader("📉 Confusion Matrices")

cols = st.columns(3)
cols[0].image("cm_baseline.png", caption="Baseline Confusion Matrix", use_column_width=True)
cols[1].image("cm_lstm.png", caption="LSTM Confusion Matrix", use_column_width=True)
cols[2].image("cm_bert.png", caption="BERT Confusion Matrix", use_column_width=True)

st.markdown("---")
st.subheader("📊 Combined Error Map")
st.image("cm_combined.png", caption="Combined Model Error Map", use_column_width=True)

# Metrics Table
st.markdown("---")
st.subheader("📈 Metrics Table")
try:
    metrics = pd.read_csv("model_metrics.csv")
    st.dataframe(metrics)
except:
    st.error("model_metrics.csv not found.")
