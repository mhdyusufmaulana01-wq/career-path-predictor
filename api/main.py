from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np, json, re, pickle, os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from deep_translator import GoogleTranslator

class AttentionLayer(layers.Layer):
    def __init__(self, units=64, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.units = units
        self.W     = layers.Dense(units, use_bias=False)
        self.V     = layers.Dense(1,     use_bias=False)

    def build(self, input_shape):
        self.W.build(input_shape)
        self.V.build((input_shape[0], input_shape[1], self.units))
        super(AttentionLayer, self).build(input_shape)

    def call(self, hidden_states):
        score     = self.V(tf.nn.tanh(self.W(hidden_states)))
        attn_w    = tf.nn.softmax(score, axis=1)
        context   = attn_w * hidden_states
        context   = tf.reduce_sum(context, axis=1)
        return context, tf.squeeze(attn_w, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config

class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, name="focal_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        y_true_oh = tf.one_hot(y_true, depth=36)
        ce_loss = -tf.reduce_sum(y_true_oh * tf.math.log(y_pred), axis=-1)
        p_t     = tf.reduce_sum(y_true_oh * y_pred, axis=-1)
        focal_w = self.alpha * tf.pow(1.0 - p_t, self.gamma)
        return tf.reduce_mean(focal_w * ce_loss)

    def get_config(self):
        config = super().get_config()
        config.update({"gamma": self.gamma, "alpha": self.alpha})
        return config

BASE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE, '../saved_model/config.json')) as f: cfg = json.load(f)


with open(os.path.join(BASE, '../saved_model/tokenizer.pkl'), 'rb') as f: tok = pickle.load(f)
model = keras.models.load_model(
    os.path.join(BASE, '../saved_model/career_path_model.keras'),
    compile=False,
    custom_objects={'AttentionLayer': AttentionLayer}
)
MAX_LEN  = cfg["max_len"]
FMIN     = np.array(cfg["feat_min"])
FMAX     = np.array(cfg["feat_max"])
LMAP     = {int(k): v for k, v in cfg["label_mapping"].items()}

app = FastAPI(title="Career Path Classifier API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class Req(BaseModel):
    text: str
    top_k: int = 3

def clean(t):
    t = re.sub(r"http\S+", "", str(t).lower())
    t = re.sub(r"[^a-z\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

@app.get("/")
def root(): return {"message": "Career Path API v1.0", "docs": "/docs"}

@app.get("/health")
def health(): return {"status": "ok", "classes": len(LMAP)}

@app.post("/predict")
def predict(req: Req):
    if not req.text.strip():
        raise HTTPException(400, "Text kosong")
        
    try:
        translator = GoogleTranslator(source='auto', target='en')
        translated = translator.translate(req.text)
    except Exception:
        translated = req.text
        
    if len(translated.split()) < 80:
        template = "professional summary highly motivated candidate with extensive experience and strong background in analytical thinking project development and delivering results education bachelor degree key skills include teamwork leadership communication and technical development i am proficient in "
        translated = template + translated
        
    c   = clean(translated)
    seq = tok.texts_to_sequences([c])
    pad = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    ws  = c.split(); nw = len(ws); nu = len(set(ws))
    ft  = np.array([[nw, nu, nu/nw if nw else 0,
                     sum(len(w) for w in ws)/nw if nw else 0]], dtype="float32")
    ft  = (ft - FMIN) / (FMAX - FMIN + 1e-8)
    pr  = model.predict({"text_input": pad, "feat_input": ft}, verbose=0)[0]
    ti  = sorted(range(len(pr)), key=lambda i: -pr[i])[:req.top_k]
    return {"input": req.text[:200], "predictions": [{"rank": i+1, "career": LMAP[idx], "confidence": round(float(pr[idx])*100, 2)} for i, idx in enumerate(ti)]}