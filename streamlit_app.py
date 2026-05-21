import streamlit as st, pandas as pd, numpy as np
import matplotlib.pyplot as plt, json, re, pickle, os
from wordcloud import WordCloud
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers
import tensorflow as tf

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

st.set_page_config(page_title='Career Path Predictor', page_icon='🎯', layout='wide')

# ── Build Model from Config ───────────────────────────────────
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

@st.cache_resource
def load_all():
    with open('saved_model/config.json') as f: cfg = json.load(f)
    with open('saved_model/tokenizer.pkl', 'rb') as f: tok = pickle.load(f)
    mdl = build_model(cfg)
    # Support both old (.keras) and new (.weights.h5) weight formats
    weights_file = cfg.get('weights_file', 'career_path_model.weights.h5')
    weights_path = f'saved_model/{weights_file}'
    if not os.path.exists(weights_path):
        weights_path = 'saved_model/career_path_model.keras'
    mdl.load_weights(weights_path)
    return mdl, tok, cfg

model, tokenizer, cfg = load_all()
lmap     = {int(k): v for k, v in cfg['label_mapping'].items()}
feat_min = np.array(cfg['feat_min'])
feat_max = np.array(cfg['feat_max'])
MAX_LEN  = cfg['max_len']

# ── Text Translation ──────────────────────────────────────────
try:
    from deep_translator import GoogleTranslator
    HAS_TRANSLATOR = True
except ImportError:
    HAS_TRANSLATOR = False

def clean(t):
    t = re.sub(r'http\S+', '', str(t).lower())
    t = re.sub(r'[^a-z\s]', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()

def predict(text, k=5):
    # Step 1: Translate if Indonesian
    translated = text
    if HAS_TRANSLATOR:
        try:
            translated = GoogleTranslator(source='auto', target='en').translate(text)
        except Exception:
            translated = text

    # Step 2: Clean text
    c = clean(translated)
    ws = c.split()

    # Step 3: Tokenize & Pad (NO repetition trick — model trained correctly now)
    seq = tokenizer.texts_to_sequences([c])
    pad = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')

    # Step 4: Feature extraction (from ORIGINAL text, before any padding)
    nw = max(len(ws), 1)
    nu = len(set(ws))
    avg_len = sum(len(w) for w in ws) / nw if nw > 0 else 0
    ft_raw = np.array([[nw, nu, nu / nw, avg_len]], dtype='float32')
    ft = np.clip(ft_raw, feat_min, feat_max)
    ft = (ft - feat_min) / (feat_max - feat_min + 1e-8)

    # Step 5: Predict
    pr = model.predict({'text_input': pad, 'feat_input': ft}, verbose=0)[0]
    ti = sorted(range(len(pr)), key=lambda i: -pr[i])[:k]
    return [(lmap[i], float(pr[i])) for i in ti], translated

# ── Sidebar Navigation ────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/000000/briefcase.png", width=64)
st.sidebar.title("Career Path Predictor")
st.sidebar.markdown("---")
page = st.sidebar.radio('📌 Menu', ['🎯 Prediksi Karir', '📊 EDA & Insight', 'ℹ️ Tentang'])

# ══════════════════════════════════════════════════════════════
if page == '🎯 Prediksi Karir':
    st.title('🎯 Career Path Predictor')
    st.markdown('Masukkan **resume**, **deskripsi skill**, atau **daftar teknologi** yang Anda kuasai untuk mendapatkan prediksi jalur karir yang paling sesuai.')
    st.markdown('> 💡 Bisa dalam **Bahasa Indonesia** atau **Bahasa Inggris**!')

    col1, col2 = st.columns([3, 1])
    with col1:
        txt = st.text_area('✍️ Teks Resume / Skills Anda:', height=200,
                           placeholder='Contoh: Saya ahli dalam Python, Machine Learning, Pandas, SQL, dan visualisasi data menggunakan Tableau...')
    with col2:
        topk = st.slider('🔢 Top N Prediksi', 1, 10, 5)
        st.markdown("---")
        st.markdown("**Tips Input:**")
        st.markdown("- Sebutkan nama teknologi/tools secara spesifik")
        st.markdown("- Gunakan istilah teknis (figma, react, docker, dll)")
        st.markdown("- Semakin detail, semakin akurat")

    if st.button('🚀 Prediksi Karir Saya!', type='primary', use_container_width=True):
        if txt.strip():
            with st.spinner('Menganalisis profil Anda...'):
                res, translated = predict(txt, topk)

            if HAS_TRANSLATOR and translated.lower() != txt.lower():
                with st.expander("🌐 Teks setelah terjemahan (klik untuk lihat)"):
                    st.write(translated)

            st.subheader('📊 Hasil Prediksi Karir:')
            st.markdown("---")

            # Display top 1 prominently
            top_career, top_conf = res[0]
            col_a, col_b = st.columns([2, 1])
            with col_a:
                st.success(f"### 🏆 Rekomendasi Utama: **{top_career.title()}**")
            with col_b:
                st.metric("Confidence Score", f"{top_conf*100:.1f}%")

            st.markdown("**Top Prediksi Lainnya:**")
            for i, (career, prob) in enumerate(res, 1):
                bar_col, name_col = st.columns([3, 2])
                with bar_col:
                    st.progress(prob, text=f"#{i} — {career.title()}")
                with name_col:
                    st.write(f"`{prob*100:.1f}%`")
        else:
            st.warning('⚠️ Masukkan teks terlebih dahulu!')

# ══════════════════════════════════════════════════════════════
elif page == '📊 EDA & Insight':
    st.title('📊 Exploratory Data Analysis')

    @st.cache_data
    def load_data():
        return pd.read_csv('train_data.csv')

    df = load_data()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('📁 Total Sampel', f'{len(df):,}')
    c2.metric('🏷️ Jumlah Kelas', 36)
    c3.metric('🔤 Vocab Size', f'{cfg["vocab_size"]:,}')
    c4.metric('📏 Max Seq Len', cfg['max_len'])

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📈 Distribusi Kelas", "☁️ Word Cloud", "📉 Visualisasi Lainnya"])

    with tab1:
        st.subheader('Distribusi 36 Kelas Karir')
        counts = df['Category_Encoded'].value_counts().sort_index()
        names  = [lmap[i] for i in counts.index]
        fig, ax = plt.subplots(figsize=(10, 11))
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(counts)))
        ax.barh(names, counts.values, color=colors, edgecolor='white')
        ax.set_xlabel('Jumlah Sampel')
        ax.set_title('Distribusi 36 Kelas Karir')
        plt.tight_layout()
        st.pyplot(fig)

    with tab2:
        st.subheader('Word Cloud per Profesi')
        sel = st.selectbox('Pilih Profesi:', [lmap[i] for i in range(36)])
        cid = [k for k, v in lmap.items() if v == sel][0]
        txt2 = ' '.join(df[df['Category_Encoded'] == cid]['Processed_Text'].fillna('').tolist())
        if txt2.strip():
            wc = WordCloud(width=800, height=400, background_color='white', max_words=100, colormap='viridis').generate(txt2)
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            ax2.imshow(wc); ax2.axis('off')
            ax2.set_title(f'Word Cloud: {sel.title()}', fontweight='bold')
            st.pyplot(fig2)

    with tab3:
        if os.path.exists('viz_training_curves.png'):
            st.subheader("Training History")
            st.image('viz_training_curves.png')
        if os.path.exists('viz_confusion_matrix.png'):
            st.subheader("Confusion Matrix")
            st.image('viz_confusion_matrix.png')
        if os.path.exists('viz_text_length.png'):
            st.subheader("Text Length Distribution")
            st.image('viz_text_length.png')

# ══════════════════════════════════════════════════════════════
else:
    st.title('ℹ️ Tentang Proyek')
    st.markdown("""
    ## Career Path Predictor
    **Proyek Capstone Dicoding** — Klasifikasi 36 jalur karir IT menggunakan teknik NLP mutakhir.

    ### 🏗️ Arsitektur Teknis
    | Komponen | Teknologi |
    |---|---|
    | Model | BiLSTM + Custom Attention Layer |
    | Framework ML | TensorFlow / Keras |
    | API Backend | FastAPI |
    | Dashboard | Streamlit |
    | Deployment | Streamlit Cloud |

    ### 📊 Spesifikasi Model
    | Parameter | Nilai |
    |---|---|
    | Vocabulary Size | 30,000 |
    | Sequence Length | 300 |
    | Embedding Dim | 128 |
    | LSTM Units | 128 (BiDirectional) |
    | Jumlah Kelas | 36 Career Paths |

    ### 🎯 36 Kelas Karir yang Diprediksi
    """)
    cols = st.columns(3)
    for i, (idx, name) in enumerate(sorted(lmap.items())):
        cols[i % 3].write(f"• {name.title()}")