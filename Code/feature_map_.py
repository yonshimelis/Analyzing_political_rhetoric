#%%


import os
os.environ["TRANSFORMERS_NO_TF"] = "1"  # ensure transformers doesn't try to use TensorFlow

import re
import pickle
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
def clean_text(t):
    t = str(t).lower()
    t = re.sub(r"[^a-zA-Z ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t
def normalize_labels(df, label_col="president", min_count=5):
    """
    Replace all presidents with < min_count speeches into label 'Other'.
    Ensures no unseen labels appear in test set.
    """
    vc = df[label_col].value_counts()
    rare = vc[vc < min_count].index

    df = df.copy()
    df[label_col + "_normalized"] = df[label_col].replace(rare, "Other")
    return df, label_col + "_normalized"

# =====================================================
# 0. DATA LOADING
# =====================================================

DEFAULT_DATA_PATH = "presidential_statements_scraped.csv"


def load_default_dataset():
    """
    Loads the default presidential statements dataset.
    """
    try:
        df = pd.read_csv(DEFAULT_DATA_PATH)
        print(f"Loaded dataset from {DEFAULT_DATA_PATH} with shape {df.shape}")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Could not find {DEFAULT_DATA_PATH}. Make sure it exists in the working directory."
        )


# =====================================================
# 1. FIXED 80/20 TRAIN–TEST SPLIT + OPTIONAL VAL SPLIT
# =====================================================

def create_train_test_split(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create a fixed train/test split (80/20 by default)."""

    strat = df[label_col] if stratify else None
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=strat,
    )
    return train_df, test_df


def create_train_val_test_split(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Optional helper if you still want train/val/test.

    First holds out `test_size` for test, then splits remaining into
    train/val with fraction `val_size` of the remaining portion.
    """

    strat = df[label_col] if stratify else None
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=strat,
    )

    strat_train = train_df[label_col] if stratify else None
    train_df, val_df = train_test_split(
        train_df,
        test_size=val_size,
        random_state=random_state,
        stratify=strat_train,
    )
    return train_df, val_df, test_df


# =====================================================
# 2. TF-IDF FEATURES + LOGISTIC REGRESSION BASELINE
# =====================================================

def build_tfidf(
    train_texts: pd.Series,
    test_texts: pd.Series,
    max_features: int = 5000,
) -> Tuple[TfidfVectorizer, np.ndarray, np.ndarray]:
    """Fit TF-IDF on training and transform train/test splits."""

    tfidf = TfidfVectorizer(max_features=max_features)
    X_train = tfidf.fit_transform(train_texts)
    X_test = tfidf.transform(test_texts)
    return tfidf, X_train, X_test


def train_logreg(X_train, y_train) -> LogisticRegression:
    """Train a simple logistic regression classifier."""
    clf = LogisticRegression(max_iter=500)
    clf.fit(X_train, y_train)
    return clf


def evaluate_model(
    model,
    X,
    y_true,
    title: str = "Model Evaluation",
    figsize: Tuple[int, int] = (8, 6),
):
    """Print classification report and plot confusion matrix."""

    preds = model.predict(X)
    print(f"\n===== {title} =====")
    print(classification_report(y_true, preds))

    cm = confusion_matrix(y_true, preds)
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix: {title}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()


def save_model(obj, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"Saved object to {path}")


def load_model(path: str):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    print(f"Loaded object from {path}")
    return obj


def predict_text_baseline(text: str, tfidf: TfidfVectorizer, model: LogisticRegression):
    """Predict label for a single raw text using the TF-IDF + LogReg baseline."""
    text = "" if text is None else str(text)
    X_vec = tfidf.transform([text])
    return model.predict(X_vec)[0]


def run_full_baseline_pipeline(
    df: Optional[pd.DataFrame] = None,
    text_col: str = "content",
    label_col: str = "president",
    max_features: int = 5000,
) -> Tuple[LogisticRegression, TfidfVectorizer, pd.DataFrame, pd.DataFrame]:
    """Run an end-to-end 80/20 baseline pipeline on the given DataFrame.

    Returns the trained classifier, TF-IDF, train_df, test_df.
    """
    if df is None:
        df = load_default_dataset()

    train_df, test_df = create_train_test_split(
        df,
        text_col=text_col,
        label_col=label_col,
        stratify=False,   # important: avoids stratification error
    )
    X_train, y_train = train_df[text_col], train_df[label_col]
    X_test, y_test = test_df[text_col], test_df[label_col]

    tfidf, X_train_tfidf, X_test_tfidf = build_tfidf(X_train, X_test, max_features=max_features)
    clf = train_logreg(X_train_tfidf, y_train)

    evaluate_model(clf, X_train_tfidf, y_train, title="Baseline – Train")
    evaluate_model(clf, X_test_tfidf, y_test, title="Baseline – Test")

    save_model(clf, "baseline_logreg.pkl")
    save_model(tfidf, "tfidf_vectorizer.pkl")

    return clf, tfidf, train_df, test_df


# =====================================================
# 3. SIMPLE TOKENIZER + PADDING (PURE PYTHON/NUMPY)
# =====================================================

class SimpleTokenizer:
    """
    Minimal word-level tokenizer similar to Keras' Tokenizer.

    - Builds a vocab of up to `num_words` most frequent tokens
    - 0 is reserved for padding
    - 1 is reserved for OOV (out-of-vocabulary)
    """

    def __init__(self, num_words: int = 50000, oov_token: str = "<OOV>"):
        self.num_words = num_words
        self.oov_token = oov_token
        self.word_index: dict[str, int] = {}
        self.oov_index: int = 1

    def _tokenize(self, text: str) -> List[str]:
        # Lowercase and split on word boundaries.
        text = text.lower()
        return re.findall(r"\b\w+\b", text)

    def fit_on_texts(self, texts: List[str]) -> None:
        from collections import Counter

        counter = Counter()
        for t in texts:
            tokens = self._tokenize(str(t))
            counter.update(tokens)

        # Reserve 0 for PAD, 1 for OOV, so actual vocab starts at 2
        most_common = counter.most_common(max(self.num_words - 2, 0))
        self.word_index = {w: i + 2 for i, (w, _) in enumerate(most_common)}
        self.word_index[self.oov_token] = self.oov_index

    def texts_to_sequences(self, texts: List[str]) -> List[List[int]]:
        sequences = []
        for t in texts:
            tokens = self._tokenize(str(t))
            seq = [self.word_index.get(tok, self.oov_index) for tok in tokens]
            sequences.append(seq)
        return sequences


def pad_sequences(
    sequences: List[List[int]],
    maxlen: int = 200,
    padding: str = "pre",
    truncating: str = "pre",
    value: int = 0,
) -> np.ndarray:
    """
    Simple implementation of Keras-like pad_sequences.

    Args:
        sequences: list of variable-length lists of ints
        maxlen: final length
        padding: 'pre' or 'post'
        truncating: 'pre' or 'post'
        value: pad value
    """
    padded = np.full((len(sequences), maxlen), value, dtype=np.int64)

    for idx, seq in enumerate(sequences):
        if len(seq) == 0:
            continue

        if truncating == "pre":
            trunc = seq[-maxlen:]
        else:
            trunc = seq[:maxlen]

        if padding == "pre":
            padded[idx, -len(trunc):] = trunc
        else:
            padded[idx, :len(trunc)] = trunc

    return padded


def build_tokenizer(
    texts: pd.Series,
    num_words: int = 50000,
    oov_token: str = "<OOV>",
) -> SimpleTokenizer:
    """Build a SimpleTokenizer on the given texts."""
    tok = SimpleTokenizer(num_words=num_words, oov_token=oov_token)
    tok.fit_on_texts(texts.astype(str).tolist())
    return tok


def texts_to_padded_sequences(
    tokenizer: SimpleTokenizer,
    texts: pd.Series,
    max_len: int = 200,
) -> np.ndarray:
    """Convert texts to padded integer sequences using SimpleTokenizer."""
    seqs = tokenizer.texts_to_sequences(texts.astype(str).tolist())
    return pad_sequences(seqs, maxlen=max_len)


# =====================================================
# 4. LSTM TEXT CLASSIFIER (PyTorch VERSION)
# =====================================================
def load_glove_embeddings(tokenizer, embedding_dim=100, glove_path="glove.6B.100d.txt"):
    embeddings_index = {}
    with open(glove_path, "r", encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = vector

    vocab_size = len(tokenizer.word_index) + 1
    embedding_matrix = np.random.normal(0, 1, (vocab_size, embedding_dim))

    for word, i in tokenizer.word_index.items():
        if word in embeddings_index:
            embedding_matrix[i] = embeddings_index[word]

    return torch.tensor(embedding_matrix, dtype=torch.float32)
class LSTMDataset(Dataset):
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.sequences = sequences
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_outputs):
        weights = torch.softmax(self.attn(lstm_outputs), dim=1)
        context = (weights * lstm_outputs).sum(dim=1)
        return context


class FineTunedLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_matrix, embedding_dim, lstm_units, num_classes):
        super().__init__()

        # Trainable pretrained embeddings
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix,
            freeze=False,
            padding_idx=0
        )

        # Strong bi-LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_units,
            batch_first=True,
            bidirectional=True,
            num_layers=2,
            dropout=0.3
        )

        # Attention layer
        self.attn = nn.Linear(lstm_units * 2, 1)

        # Extra stabilization
        self.layernorm = nn.LayerNorm(lstm_units * 2)
        self.dropout = nn.Dropout(0.4)

        self.fc = nn.Linear(lstm_units * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)

        # attention weights
        weights = torch.softmax(self.attn(lstm_out), dim=1)
        context = (weights * lstm_out).sum(dim=1)

        context = self.layernorm(context)
        context = self.dropout(context)

        return self.fc(context)

def train_lstm_classifier(
    df: Optional[pd.DataFrame] = None,
    text_col: str = "content",
    label_col: str = "president",
    max_words: int = 50000,
    max_len: int = 400,
    embedding_dim: int = 100,
    lstm_units: int = 256,
    batch_size: int = 64,
    epochs: int = 12,
    learning_rate: float = 1e-3,
    random_state: int = 42,
):
    """
    FINAL CLEAN VERSION: Fine-tuned BiLSTM + Attention + GloVe + OneCycleLR + Early Stopping
    """

    if df is None:
        df = load_default_dataset()

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # --------------------------------------------------------
    # Train/Test Split
    # --------------------------------------------------------
    train_df, test_df = create_train_test_split(df, text_col, label_col, random_state=random_state)

    # Label encoding
    le = LabelEncoder()
    y_train = le.fit_transform(train_df[label_col])
    y_test = le.transform(test_df[label_col])

    # Tokenize
    tokenizer = build_tokenizer(train_df[text_col], num_words=max_words)
    X_train = texts_to_padded_sequences(tokenizer, train_df[text_col], max_len)
    X_test = texts_to_padded_sequences(tokenizer, test_df[text_col], max_len)

    num_classes = len(np.unique(y_train))
    vocab_size = min(max_words, len(tokenizer.word_index) + 1)

    # --------------------------------------------------------
    # Data loaders
    # --------------------------------------------------------
    train_loader = DataLoader(LSTMDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(LSTMDataset(X_test, y_test), batch_size=batch_size)

    # --------------------------------------------------------
    # Load pretrained GloVe
    # --------------------------------------------------------
    embedding_matrix = load_glove_embeddings(tokenizer, embedding_dim=embedding_dim)

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------
    model = FineTunedLSTM(
        vocab_size=vocab_size,
        embedding_matrix=embedding_matrix,
        embedding_dim=embedding_dim,
        lstm_units=lstm_units,
        num_classes=num_classes,
    ).to(device)

    class FocalLoss(nn.Module):
        def __init__(self, gamma=2, alpha=None):
            super().__init__()
            self.gamma = gamma
            self.alpha = alpha

        def forward(self, logits, targets):
            ce = nn.CrossEntropyLoss(reduction="none")(logits, targets)
            pt = torch.exp(-ce)
            focal = ((1 - pt) ** self.gamma) * ce

            if self.alpha is not None:
                focal = self.alpha[targets] * focal

            return focal.mean()

    criterion = FocalLoss(gamma=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1
    )

    # --------------------------------------------------------
    # Training Loop with Early Stopping
    # --------------------------------------------------------
    best_val_loss = float("inf")
    patience = 3
    wait = 0

    training_log = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):

        model.train()
        train_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            out = model(X_batch)
            loss = criterion(out, y_batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        avg_train = train_loss / len(train_loader)
        training_log["train_loss"].append(avg_train)

        # ----------------- VALIDATION -----------------
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                out = model(X_batch)
                val_loss += criterion(out, y_batch).item()

        avg_val = val_loss / len(test_loader)
        training_log["val_loss"].append(avg_val)

        print(f"Epoch {epoch+1}/{epochs} | Train: {avg_train:.4f} | Val: {avg_val:.4f}")

        # ----------------- EARLY STOPPING -----------------
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            wait = 0
            torch.save(model.state_dict(), "best_lstm.pt")
        else:
            wait += 1
            if wait >= patience:
                print("\nEarly stopping triggered.")
                break

    # --------------------------------------------------------
    # Load best checkpoint
    # --------------------------------------------------------
    print("\nLoading best LSTM checkpoint...")
    model.load_state_dict(torch.load("best_lstm.pt"))
    # ============================================================
    # FINAL LSTM EVALUATION: F1, ACCURACY, COHEN KAPPA, CONF MATRIX
    # ============================================================
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        f1_score,
        cohen_kappa_score,
        confusion_matrix
    )
    import seaborn as sns
    import matplotlib.pyplot as plt

    def evaluate_lstm_full(model, data_loader, label_encoder, device):
        model.eval()
        preds, trues = [], []

        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch = X_batch.to(device)
                out = model(X_batch)
                pred = torch.argmax(out, dim=1).cpu().numpy()
                preds.extend(pred.tolist())
                trues.extend(y_batch.numpy().tolist())

        preds = np.array(preds)
        trues = np.array(trues)

        # -------------------------
        # BASIC METRICS
        # -------------------------
        accuracy = accuracy_score(trues, preds)
        f1_macro = f1_score(trues, preds, average="macro", zero_division=0)
        f1_micro = f1_score(trues, preds, average="micro", zero_division=0)
        f1_weighted = f1_score(trues, preds, average="weighted", zero_division=0)

        kappa = cohen_kappa_score(trues, preds)

        print("\n============================")
        print(" OVERALL METRICS")
        print("============================")
        print(f"Accuracy       : {accuracy:.4f}")
        print(f"F1 Macro       : {f1_macro:.4f}")
        print(f"F1 Micro       : {f1_micro:.4f}")
        print(f"F1 Weighted    : {f1_weighted:.4f}")
        print(f"Cohen Kappa    : {kappa:.4f}")

        # -------------------------
        # PER-CLASS METRICS
        # -------------------------
        print("\n============================")
        print(" CLASSIFICATION REPORT")
        print("============================")
        print(classification_report(
            trues,
            preds,
            target_names=label_encoder.classes_,
            zero_division=0
        ))

        # -------------------------
        # CONFUSION MATRIX
        # -------------------------
        cm = confusion_matrix(trues, preds)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=False, cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()

        return {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_micro": f1_micro,
            "f1_weighted": f1_weighted,
            "kappa": kappa,
        }

    print("\n==============================")
    print(" FULL LSTM METRICS REPORT")
    print("==============================")

    train_loader_eval = DataLoader(LSTMDataset(X_train, y_train), batch_size=64)
    test_loader_eval = DataLoader(LSTMDataset(X_test, y_test), batch_size=64)

    print("\n--- TRAIN METRICS ---")
    train_metrics = evaluate_lstm_full(model, train_loader_eval, le, device)

    print("\n--- TEST METRICS ---")
    test_metrics = evaluate_lstm_full(model, test_loader_eval, le, device)

    # Save final model + tokenizer + encoder
    torch.save(model.state_dict(), "final_lstm_model.pt")
    save_model(tokenizer, "lstm_tokenizer.pkl")
    save_model(le, "lstm_label_encoder.pkl")

    print("\nSaved:")
    print("- final_lstm_model.pt")
    print("- lstm_tokenizer.pkl")
    print("- lstm_label_encoder.pkl")

    return model, tokenizer, le, training_log, (train_df, test_df)

def predict_text_lstm(
    text: str,
    model: train_lstm_classifier,
    tokenizer: SimpleTokenizer,
    label_encoder: LabelEncoder,
    max_len: int = 400,
):
    """Predict a single text using the trained PyTorch LSTM model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=max_len)
    x = torch.tensor(pad, dtype=torch.long).to(device)

    with torch.no_grad():
        outputs = model(x)
        pred_id = torch.argmax(outputs, dim=1).cpu().numpy()[0]

    return label_encoder.inverse_transform([pred_id])[0]


# =====================================================
# 5. BERT FINE-TUNING (HuggingFace Transformers, PyTorch)
# =====================================================

class TextClassificationDataset(Dataset):
    """Simple Dataset wrapper for (encodings, labels)."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def prepare_bert_encodings(
    texts: pd.Series,
    tokenizer,
    max_length: int = 256,
):
    """Tokenize texts for BERT-style models."""
    encodings = tokenizer(
        texts.astype(str).tolist(),
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    return encodings


def train_bert_classifier(
    df: Optional[pd.DataFrame] = None,
    text_col: str = "content",
    label_col: str = "president",
    model_name: str = "distilbert-base-uncased",
    max_length: int = 256,
    num_epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    output_dir: str = "bert_finetuned",
    random_state: int = 42,
):
    """Fine-tune a BERT-like model for text classification using 80/20 split.

    Returns
    -------
    model, tokenizer, label_encoder, trainer, (train_df, test_df)
    """

    if df is None:
        df = load_default_dataset()

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    # 80/20 split
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=random_state,
        stratify=df[label_col]
    )
    # Label encoding
    le = LabelEncoder()
    y_train = le.fit_transform(train_df[label_col])
    y_test = le.transform(test_df[label_col])

    num_labels = len(le.classes_)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )

    train_encodings = prepare_bert_encodings(train_df[text_col], tokenizer, max_length=max_length)
    test_encodings = prepare_bert_encodings(test_df[text_col], tokenizer, max_length=max_length)

    train_dataset = TextClassificationDataset(train_encodings, y_train)
    test_dataset = TextClassificationDataset(test_encodings, y_test)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,

        # transformers >= 4.55 style
        eval_strategy="epoch",
        logging_strategy="epoch",

        # How often to save checkpoints
        save_strategy="epoch",  # save every epoch
        save_total_limit=2,  # keep last 2 checkpoints

        learning_rate=learning_rate,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",

        load_best_model_at_end=True,
        metric_for_best_model="accuracy",

        report_to="none",  # disable W&B etc on EC2
    )
    def compute_metrics(eval_pred):
        from sklearn.metrics import accuracy_score, f1_score

        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro"),
            "f1_micro": f1_score(labels, preds, average="micro"),
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate(test_dataset)
    print("\nBERT fine-tune evaluation metrics:", metrics)

    # Save
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    save_model(le, f"{output_dir}/label_encoder.pkl")

    return model, tokenizer, le, trainer, (train_df, test_df)


def predict_text_bert(
    text: str,
    model,
    tokenizer,
    label_encoder: LabelEncoder,
    max_length: int = 256,
):
    """Predict label for a single text using fine-tuned BERT model."""
    text = "" if text is None else str(text)
    inputs = tokenizer(
        [text],
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    outputs = model(**inputs)
    probs = outputs.logits.softmax(dim=-1).detach().cpu().numpy()[0]
    pred_id = int(np.argmax(probs))
    return label_encoder.inverse_transform([pred_id])[0]


# =====================================================
# 6. VISUALIZATION HELPERS
# (These assume you will later create sentiment/tone columns;
# for now they are not used in main)
# =====================================================

def plot_sentiment_distribution(df: pd.DataFrame, sentiment_col: str = "transformer_sentiment"):
    """Bar plot of sentiment label counts."""
    plt.figure(figsize=(6, 4))
    df[sentiment_col].value_counts().plot(kind="bar")
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def plot_party_sentiment(
    df: pd.DataFrame,
    party_col: str = "party",
    sentiment_col: str = "transformer_sentiment",
):
    """Stacked bar plot of sentiment by party."""
    ctab = pd.crosstab(df[party_col], df[sentiment_col])
    ctab.plot(kind="bar", stacked=True, figsize=(10, 5))
    plt.title("Sentiment by Party")
    plt.xlabel("Party")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def plot_tone_distribution(df: pd.DataFrame, tone_col: str = "tone"):
    """Bar plot of tone distribution."""
    plt.figure(figsize=(6, 4))
    df[tone_col].value_counts().plot(kind="bar")
    plt.title("Tone Distribution")
    plt.xlabel("Tone")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def export_dataset(df: pd.DataFrame, path: str = "final_presidential_dataset.csv"):
    """Export dataset to CSV."""
    df.to_csv(path, index=False)
    print(f"Dataset exported to {path}")


# =====================================================
# MAIN EXECUTION BLOCK
# =====================================================

if __name__ == "__main__":
    print("\n==============================")
    print(" LOADING DATASET ")
    print("==============================")
    df = load_default_dataset()
    df["content"] = df["content"].apply(clean_text)

    # Normalize rare presidents
    df, normalized_label_col = normalize_labels(df, label_col="president", min_count=5)
    print(f"Using normalized label column: {normalized_label_col}\n")

    print("\n==============================")
    print(" RUNNING TF-IDF BASELINE ")
    print("==============================")
    clf, tfidf, train_df, test_df = run_full_baseline_pipeline(
        df,
        text_col="content",
        label_col=normalized_label_col,
    )
    print("Baseline model saved as baseline_logreg.pkl")

    print("\n==============================")
    print(" TRAINING PYTORCH LSTM MODEL ")
    print("==============================")
    lstm_model, tokenizer, le, log, splits = train_lstm_classifier(
        df,
        text_col="content",
        label_col=normalized_label_col,
    )
    print("LSTM model saved as pytorch_lstm_model.pt")

    print("\n==============================")
    print(" BERT FINE-TUNING (optional) ")
    print("==============================")
    do_bert = input("Fine-tune BERT? (yes/no): ").strip().lower()

    if do_bert == "yes":
        bert_model, bert_tokenizer, bert_le, trainer, splits = train_bert_classifier(
            df,
            text_col="content",
            label_col=normalized_label_col,
        )
        print("BERT model saved to bert_finetuned/")
    else:
        print("Skipped BERT fine-tuning.")

# %%
