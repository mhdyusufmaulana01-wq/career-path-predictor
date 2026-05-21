from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np, json, re, pickle, os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers

# ── Custom Attention Layer ────────────────────────────────────
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
        score   = self.V(tf.nn.tanh(self.W(hidden_states)))
        attn_w  = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(attn_w * hidden_states, axis=1)
        return context, tf.squeeze(attn_w, axis=-1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units})
        return cfg

# ── Build Model ───────────────────────────────────────────────
def build_model(cfg):
    text_input = keras.Input(shape=(cfg['max_len'],), name='text_input')
    x = layers.Embedding(cfg['vocab_size'], cfg['embed_dim'], name='embedding')(text_input)
    x = layers.SpatialDropout1D(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.1), name='bilstm')(x)
    context, _ = AttentionLayer(units=64, name='attention')(x)
    x_text = layers.Dropout(0.4)(context)

    feat_input = keras.Input(shape=(4,), name='feat_input')
    x_feat = layers.Dense(32, activation='relu', name='feat_dense')(feat_input)
    x_feat = layers.Dropout(0.3)(x_feat)

    merged = layers.Concatenate(name='merge')([x_text, x_feat])
    merged = layers.Dense(256, activation='relu', name='dense_1')(merged)
    merged = layers.BatchNormalization()(merged)
    merged = layers.Dropout(0.4)(merged)
    merged = layers.Dense(128, activation='relu', name='dense_2')(merged)
    merged = layers.Dropout(0.3)(merged)

    output = layers.Dense(cfg['num_classes'], activation='softmax', name='output')(merged)
    return keras.Model(inputs=[text_input, feat_input], outputs=output, name='CareerPathClassifier')

# ── Load Artifacts ────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE, '../saved_model/config.json')) as f: cfg = json.load(f)
with open(os.path.join(BASE, '../saved_model/tokenizer.pkl'), 'rb') as f: tok = pickle.load(f)
model = build_model(cfg)
# Support both old (.keras) and new (.weights.h5) weight formats
weights_file = cfg.get('weights_file', 'career_path_model.weights.h5')
weights_path = os.path.join(BASE, '../saved_model', weights_file)
if not os.path.exists(weights_path):
    weights_path = os.path.join(BASE, '../saved_model/career_path_model.keras')
model.load_weights(weights_path)

MAX_LEN = cfg['max_len']
FMIN    = np.array(cfg['feat_min'])
FMAX    = np.array(cfg['feat_max'])
LMAP    = {int(k): v for k, v in cfg['label_mapping'].items()}

# ── Translator (optional) ─────────────────────────────────────
try:
    from deep_translator import GoogleTranslator
    HAS_TRANSLATOR = True
except ImportError:
    HAS_TRANSLATOR = False

# ── FastAPI App ───────────────────────────────────────────────
app = FastAPI(title='Career Path Classifier API', version='2.0.0')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

class Req(BaseModel):
    text: str
    top_k: int = 3
    translate: bool = True

def clean(t):
    t = re.sub(r'http\S+', '', str(t).lower())
    t = re.sub(r'[^a-z\s]', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()

@app.get('/')
def root(): return {'message': 'Career Path API v2.0', 'docs': '/docs', 'vocab_size': MAX_LEN}

@app.get('/health')
def health(): return {'status': 'ok', 'classes': len(LMAP), 'vocab_size': cfg['vocab_size'], 'max_len': MAX_LEN}

@app.post('/predict')
def predict(req: Req):
    if not req.text.strip():
        raise HTTPException(400, 'Text tidak boleh kosong')

    # Step 1: Translate if needed
    translated = req.text
    if req.translate and HAS_TRANSLATOR:
        try:
            translated = GoogleTranslator(source='auto', target='en').translate(req.text)
        except Exception:
            translated = req.text

    # Step 2: Clean
    c  = clean(translated)
    ws = c.split()
    nw = max(len(ws), 1)
    nu = len(set(ws))

    # Step 3: Tokenize & Pad
    seq = tok.texts_to_sequences([c])
    pad = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')

    # Step 4: Features (from original cleaned text, before padding)
    avg_len = sum(len(w) for w in ws) / nw
    ft_raw  = np.array([[nw, nu, nu / nw, avg_len]], dtype='float32')
    ft      = np.clip(ft_raw, FMIN, FMAX)
    ft      = (ft - FMIN) / (FMAX - FMIN + 1e-8)

    # Step 5: Predict
    pr = model.predict({'text_input': pad, 'feat_input': ft}, verbose=0)[0]
    ti = sorted(range(len(pr)), key=lambda i: -pr[i])[:req.top_k]

    return {
        'input_original': req.text[:200],
        'input_translated': translated[:200] if translated != req.text else None,
        'predictions': [
            {
                'rank': i + 1,
                'career': LMAP[idx],
                'confidence': round(float(pr[idx]) * 100, 2),
            }
            for i, idx in enumerate(ti)
        ],
    }