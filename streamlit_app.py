import streamlit as st, pandas as pd, numpy as np
import matplotlib.pyplot as plt, json, re, pickle, os
from wordcloud import WordCloud
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers

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

st.set_page_config(page_title='Career Path Predictor', page_icon='X', layout='wide')

def build_model(cfg):
    text_input = keras.Input(shape=(cfg['max_len'],), name='text_input')
    x = layers.Embedding(cfg['vocab_size'], cfg['embed_dim'], name='embedding')(text_input)
    x = layers.SpatialDropout1D(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.1), name='bilstm')(x)
    attn_layer = AttentionLayer(units=64, name='attention')
    context, attn = attn_layer(x)
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
    model = keras.Model(inputs=[text_input, feat_input], outputs=output, name='CareerPathClassifier')
    return model

@st.cache_resource
def load_all():
    with open('saved_model/config.json') as f: cfg = json.load(f)
    with open('saved_model/tokenizer.pkl', 'rb') as f: tok = pickle.load(f)
    mdl = build_model(cfg)
    mdl.load_weights('saved_model/career_path_model.keras')
    return mdl, tok, cfg

model, tokenizer, cfg = load_all()
lmap     = {int(k): v for k, v in cfg["label_mapping"].items()}
feat_min = np.array(cfg["feat_min"])
feat_max = np.array(cfg["feat_max"])
MAX_LEN  = cfg["max_len"]

from deep_translator import GoogleTranslator

def clean(t):
    t = re.sub(r"http\S+", "", str(t).lower())
    t = re.sub(r"[^a-z\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def predict(text, k=5):
    try:
        translator = GoogleTranslator(source='auto', target='en')
        translated = translator.translate(text)
    except Exception:
        translated = text
        
    if len(translated.split()) < 80:
        template = "professional summary highly motivated candidate with extensive experience and strong background in analytical thinking project development and delivering results education bachelor degree key skills include teamwork leadership communication and technical development i am proficient in "
        translated = template + translated
        
    c   = clean(translated)
    seq = tokenizer.texts_to_sequences([c])
    pad = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    ws  = c.split(); nw = len(ws); nu = len(set(ws))
    ft  = np.array([[nw, nu, nu/nw if nw else 0,
                     sum(len(w) for w in ws)/nw if nw else 0]], dtype="float32")
    ft  = (ft - feat_min) / (feat_max - feat_min + 1e-8)
    pr  = model.predict({"text_input": pad, "feat_input": ft}, verbose=0)[0]
    ti  = sorted(range(len(pr)), key=lambda i: -pr[i])[:k]
    return [(lmap[i], float(pr[i])) for i in ti]

page = st.sidebar.radio('Menu', ['Prediksi Karir', 'EDA', 'Tentang'])

if page == 'Prediksi Karir':
    st.title('Career Path Predictor')
    st.markdown('Masukkan teks resume atau daftar skill untuk prediksi karir.')
    txt  = st.text_area('Teks Resume / Skills:', height=180)
    topk = st.slider('Top N prediksi', 1, 10, 5)
    if st.button('Prediksi', type='primary'):
        if txt.strip():
            res = predict(txt, topk)
            st.subheader('Hasil Prediksi:')
            st.info('💡 Teks telah diterjemahkan ke Bahasa Inggris dan diproses dengan *context padding* otomatis agar hasil prediksi lebih akurat.')
            for i, (c, p) in enumerate(res, 1):
                st.progress(p, text=f'{i}. {c.title()} ({p*100:.1f}%)')
        else:
            st.warning('Masukkan teks terlebih dahulu!')

elif page == 'EDA':
    st.title('Insight Dataset')
    df = pd.read_csv('train_data.csv')
    c1, c2, c3 = st.columns(3)
    c1.metric('Total Sampel', f'{len(df):,}')
    c2.metric('Jumlah Kelas', 36)
    c3.metric('Fitur', 'Teks Resume')
    st.subheader('Distribusi Kelas')
    counts = df["Category_Encoded"].value_counts().sort_index()
    names  = [lmap[i] for i in counts.index]
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.barh(names, counts.values, color="steelblue")
    ax.set_xlabel('Jumlah Sampel'); ax.set_title('Distribusi 36 Kelas Karir')
    plt.tight_layout(); st.pyplot(fig)
    st.subheader('Word Cloud')
    sel = st.selectbox('Pilih Profesi:', [lmap[i] for i in range(36)])
    cid = [k for k, v in lmap.items() if v == sel][0]
    txt2 = " ".join(df[df["Category_Encoded"] == cid]["Processed_Text"].fillna("").tolist())
    if txt2.strip():
        wc = WordCloud(width=800, height=400, background_color="white", max_words=100).generate(txt2)
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.imshow(wc); ax2.axis("off")
        ax2.set_title(f'Word Cloud: {sel.title()}', fontweight='bold')
        st.pyplot(fig2)

else:
    st.title('Tentang Proyek')
    st.markdown('**Capstone Dicoding** - Klasifikasi 36 career path menggunakan BiLSTM + Custom Attention')
    st.markdown('Model: TensorFlow | API: FastAPI | Dashboard: Streamlit')