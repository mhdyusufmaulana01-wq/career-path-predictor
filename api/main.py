from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np, json, re, pickle, os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

BASE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE, '../saved_model/config.json')) as f: cfg = json.load(f)
with open(os.path.join(BASE, '../saved_model/tokenizer.pkl'), 'rb') as f: tok = pickle.load(f)
model = keras.models.load_model(os.path.join(BASE, '../saved_model/career_path_model.keras'))
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
    c   = clean(req.text)
    seq = tok.texts_to_sequences([c])
    pad = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    ws  = c.split(); nw = len(ws); nu = len(set(ws))
    ft  = np.array([[nw, nu, nu/nw if nw else 0,
                     sum(len(w) for w in ws)/nw if nw else 0]], dtype="float32")
    ft  = (ft - FMIN) / (FMAX - FMIN + 1e-8)
    pr  = model.predict({"text_input": pad, "feat_input": ft}, verbose=0)[0]
    ti  = sorted(range(len(pr)), key=lambda i: -pr[i])[:req.top_k]
    return {"input": req.text[:200], "predictions": [{"rank": i+1, "career": LMAP[idx], "confidence": round(float(pr[idx])*100, 2)} for i, idx in enumerate(ti)]}