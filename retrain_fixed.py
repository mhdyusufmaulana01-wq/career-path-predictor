"""
RETRAIN SCRIPT DENGAN PERBAIKAN:
1. VOCAB_SIZE: 20,000 → 30,000 (menghilangkan OOV pada kata teknis penting)
2. MAX_LEN: 200 → 300 (representatif untuk semua kelas)
3. Simpan tokenizer & config yang benar
"""

import pandas as pd
import numpy as np
import json, re, os, pickle, warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report

print("=" * 60)
print("CAREER PATH PREDICTOR — RETRAIN (FIXED)")
print("=" * 60)
print(f"TensorFlow: {tf.__version__}")

# ── Load & Clean ──────────────────────────────────────────────
df = pd.read_csv("train_data.csv")
with open("label_mapping.json") as f:
    label_mapping = {int(k): v for k, v in json.load(f).items()}

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

df = df.dropna(subset=["Processed_Text"]).drop_duplicates(subset=["Processed_Text"]).reset_index(drop=True)
df["Clean_Text"] = df["Processed_Text"].apply(clean_text)

X = df["Clean_Text"].values
y = df["Category_Encoded"].values

print(f"Dataset: {len(df):,} rows, {len(label_mapping)} classes")

# ── Split ─────────────────────────────────────────────────────
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
X_val,   X_test,  y_val,  y_test  = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)
print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# ── Tokenizer (FIXED: VOCAB_SIZE 30000, MAX_LEN 300) ─────────
VOCAB_SIZE  = 30000   # ← FIX: was 20000, covers all 28252 unique words
MAX_LEN     = 300     # ← FIX: was 200, better for long resumes
EMBED_DIM   = 128
OOV_TOKEN   = "<OOV>"

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOKEN)
tokenizer.fit_on_texts(X_train)

actual_vocab = len(tokenizer.word_index)
print(f"Vocab size: {actual_vocab:,} unique words | Using: {VOCAB_SIZE:,}")
print(f"Max sequence length: {MAX_LEN}")

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_val_seq   = tokenizer.texts_to_sequences(X_val)
X_test_seq  = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding="post", truncating="post")
X_val_pad   = pad_sequences(X_val_seq,   maxlen=MAX_LEN, padding="post", truncating="post")
X_test_pad  = pad_sequences(X_test_seq,  maxlen=MAX_LEN, padding="post", truncating="post")

# ── Feature Engineering ───────────────────────────────────────
def extract_feats(texts):
    feats = []
    for t in texts:
        ws = t.split()
        nw = len(ws)
        nu = len(set(ws))
        feats.append([
            nw,
            nu,
            nu / nw if nw > 0 else 0,
            sum(len(w) for w in ws) / nw if nw > 0 else 0,
        ])
    return np.array(feats, dtype="float32")

F_train = extract_feats(X_train)
F_val   = extract_feats(X_val)
F_test  = extract_feats(X_test)

feat_min = F_train.min(axis=0)
feat_max = F_train.max(axis=0)

F_train_n = (F_train - feat_min) / (feat_max - feat_min + 1e-8)
F_val_n   = (F_val   - feat_min) / (feat_max - feat_min + 1e-8)
F_test_n  = (F_test  - feat_min) / (feat_max - feat_min + 1e-8)

print(f"Feature shape: {F_train.shape} | min={feat_min} | max={feat_max}")

# ── Class Weights ─────────────────────────────────────────────
classes = np.unique(y_train)
weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weights = dict(zip(classes, weights))

# ── Model Architecture ────────────────────────────────────────
class AttentionLayer(layers.Layer):
    def __init__(self, units=64, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.W = layers.Dense(units, use_bias=False)
        self.V = layers.Dense(1,     use_bias=False)

    def build(self, input_shape):
        self.W.build(input_shape)
        self.V.build((input_shape[0], input_shape[1], self.units))
        super().build(input_shape)

    def call(self, hidden_states):
        score  = self.V(tf.nn.tanh(self.W(hidden_states)))
        attn_w = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(attn_w * hidden_states, axis=1)
        return context, tf.squeeze(attn_w, axis=-1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units})
        return cfg

NUM_CLASSES = len(label_mapping)

def build_model():
    text_input = keras.Input(shape=(MAX_LEN,),  name="text_input")
    x = layers.Embedding(VOCAB_SIZE, EMBED_DIM, name="embedding")(text_input)
    x = layers.SpatialDropout1D(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.1), name="bilstm")(x)
    context, _ = AttentionLayer(units=64, name="attention")(x)
    x_text = layers.Dropout(0.4)(context)

    feat_input = keras.Input(shape=(4,), name="feat_input")
    x_feat = layers.Dense(32, activation="relu", name="feat_dense")(feat_input)
    x_feat = layers.Dropout(0.3)(x_feat)

    merged = layers.Concatenate(name="merge")([x_text, x_feat])
    merged = layers.Dense(256, activation="relu", name="dense_1")(merged)
    merged = layers.BatchNormalization()(merged)
    merged = layers.Dropout(0.4)(merged)
    merged = layers.Dense(128, activation="relu", name="dense_2")(merged)
    merged = layers.Dropout(0.3)(merged)

    output = layers.Dense(NUM_CLASSES, activation="softmax", name="output")(merged)
    model  = keras.Model(inputs=[text_input, feat_input], outputs=output, name="CareerPathClassifier")
    return model

model = build_model()
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# ── Training ──────────────────────────────────────────────────
callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1),
]

print("\nStarting training...")
history = model.fit(
    {"text_input": X_train_pad, "feat_input": F_train_n},
    y_train,
    validation_data=({"text_input": X_val_pad, "feat_input": F_val_n}, y_val),
    epochs=30,
    batch_size=64,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1,
)

# ── Evaluation ────────────────────────────────────────────────
print("\n=== TEST EVALUATION ===")
y_pred_prob = model.predict({"text_input": X_test_pad, "feat_input": F_test_n}, verbose=0)
y_pred = np.argmax(y_pred_prob, axis=1)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc*100:.2f}%")
print()
print(classification_report(y_test, y_pred, target_names=[label_mapping[i] for i in range(NUM_CLASSES)]))

# ── Save Model & Artifacts ────────────────────────────────────
os.makedirs("saved_model", exist_ok=True)

# Save weights
model.save_weights("saved_model/career_path_model.keras")
print("✅ Weights saved: saved_model/career_path_model.keras")

# Save tokenizer
with open("saved_model/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
print("✅ Tokenizer saved: saved_model/tokenizer.pkl")

# Save config (UPDATED)
config = {
    "vocab_size": VOCAB_SIZE,
    "max_len": MAX_LEN,
    "embed_dim": EMBED_DIM,
    "num_classes": NUM_CLASSES,
    "oov_token": OOV_TOKEN,
    "feat_min": feat_min.tolist(),
    "feat_max": feat_max.tolist(),
    "label_mapping": {str(k): v for k, v in label_mapping.items()},
}
with open("saved_model/config.json", "w") as f:
    json.dump(config, f, indent=2)
print("✅ Config saved: saved_model/config.json")
print()
print("=" * 60)
print(f"RETRAIN SELESAI! Test Accuracy: {acc*100:.2f}%")
print("=" * 60)
