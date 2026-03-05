"""
EmoSense AI — Emotion Recognition Frontend
Run: streamlit run app.py
"""

import os, numpy as np, pandas as pd, torch
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
from collections import Counter

st.set_page_config(page_title="EmoSense AI", page_icon="🌸",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display:ital@0;1&display=swap');

:root {
  --bg:      #f7f3ef;
  --bg2:     #f0ebe4;
  --card:    #faf7f4;
  --border:  #e2d9d0;
  --text:    #3d3530;
  --muted:   #9c8f86;
  --a1:      #c9a0dc;
  --a2:      #f4a7b9;
  --pos:     #7dbf9e;
  --neg:     #e8a0a0;
  --neu:     #b8b0c8;
}

html, body, [class*="css"] { font-family:'DM Sans',sans-serif; color:var(--text); }
.stApp { background:var(--bg); }
.main .block-container { padding:1.4rem 2.2rem 3rem; max-width:1200px; }

[data-testid="stSidebar"] {
  background:var(--card) !important;
  border-right:1px solid var(--border) !important;
}

.hero-wrap { text-align:center; padding:1.4rem 0 1rem; }
.hero-title {
  font-family:'DM Serif Display',serif;
  font-size:2.8rem; font-weight:400; color:var(--text); margin:0; line-height:1.1;
}
.hero-title span {
  background:linear-gradient(135deg,var(--a1),var(--a2));
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.hero-sub { color:var(--muted); font-size:.98rem; margin-top:.45rem; }
.hero-badge {
  display:inline-block; margin-top:.6rem; padding:.2rem .8rem;
  background:rgba(201,160,220,.15); border:1px solid rgba(201,160,220,.4);
  border-radius:999px; font-size:.7rem; color:var(--a1);
  letter-spacing:.07em; text-transform:uppercase;
}

.card {
  background:var(--card); border:1px solid var(--border);
  border-radius:18px; padding:1.4rem; margin-bottom:1.1rem;
  box-shadow:0 2px 12px rgba(0,0,0,.04);
  transition:border-color .2s, box-shadow .2s;
}
.card:hover { border-color:var(--a1); box-shadow:0 4px 20px rgba(0,0,0,.07); }

.mbox {
  background:white; border:1px solid var(--border);
  border-radius:14px; padding:1rem; text-align:center;
  box-shadow:0 1px 6px rgba(0,0,0,.04);
}
.mval { font-family:'DM Serif Display',serif; font-size:1.9rem; line-height:1; color:var(--text); }
.mlbl { font-size:.67rem; color:var(--muted); margin-top:.3rem; text-transform:uppercase; letter-spacing:.07em; }

.sec-lbl { font-size:.68rem; text-transform:uppercase; letter-spacing:.1em; color:var(--muted); margin-bottom:.45rem; font-weight:600; }

.epill {
  display:inline-flex; align-items:center; gap:.3rem;
  padding:.23rem .68rem; border-radius:999px;
  font-size:.75rem; font-weight:600; margin:.17rem;
}

.stTextArea textarea {
  background:white !important; border:1px solid var(--border) !important;
  border-radius:12px !important; color:var(--text) !important;
  font-family:'DM Sans',sans-serif !important; font-size:1rem !important;
}
.stTextArea textarea:focus {
  border-color:var(--a1) !important;
  box-shadow:0 0 0 3px rgba(201,160,220,.15) !important;
}

/* Hide all widget labels — removes ghost boxes */
.stTextArea label,
.stSelectbox label,
.stRadio label { display:none !important; }

/* Main buttons */
.stButton > button {
  background:linear-gradient(135deg,var(--a1),var(--a2)) !important;
  color:white !important; border:none !important; border-radius:10px !important;
  font-family:'DM Sans',sans-serif !important; font-weight:600 !important;
  font-size:.95rem !important; padding:.55rem 1.6rem !important;
  box-shadow:0 2px 10px rgba(201,160,220,.35) !important;
}
.stButton > button:hover {
  transform:translateY(-1px) !important;
  box-shadow:0 4px 16px rgba(201,160,220,.45) !important;
}

/* Sidebar example buttons */
[data-testid="stSidebar"] .stButton > button {
  background:white !important; color:var(--text) !important;
  border:1px solid var(--border) !important; border-radius:8px !important;
  font-size:.82rem !important; font-weight:400 !important;
  padding:.28rem .7rem !important; box-shadow:none !important;
  text-align:left !important; justify-content:flex-start !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
  background:rgba(201,160,220,.12) !important; border-color:var(--a1) !important;
  transform:none !important; box-shadow:none !important;
}

.stTabs [data-baseweb="tab-list"] {
  background:var(--bg2); border-radius:12px; padding:4px; gap:6px;
  border:1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
  background:transparent !important; color:var(--muted) !important;
  border-radius:9px !important; font-family:'DM Sans',sans-serif !important;
  font-weight:500 !important; padding:0.35rem 1rem !important; margin:0 1px !important;
}
.stTabs [aria-selected="true"] {
  background:white !important; color:var(--text) !important;
  box-shadow:0 1px 6px rgba(0,0,0,.08) !important;
}

.stSelectbox [data-baseweb="select"] > div {
  background:white !important; border-color:var(--border) !important;
  border-radius:10px !important; color:var(--text) !important;
}

/* Radio buttons for chart type */
[data-testid="stRadio"] > div { gap:6px !important; }
[data-testid="stRadio"] > div > label {
  display:flex !important;
  background:white; border:1px solid var(--border);
  border-radius:8px; padding:.3rem .8rem;
  font-size:.83rem; cursor:pointer;
  transition:border-color .2s, background .2s;
}
[data-testid="stRadio"] > div > label:hover {
  border-color:var(--a1); background:rgba(201,160,220,.08);
}
[data-testid="stRadio"] > div > label[data-checked="true"] {
  border-color:var(--a1); background:rgba(201,160,220,.15);
  color:var(--a1); font-weight:600;
}

[data-testid="stSlider"] [role="progressbar"] {
  background:linear-gradient(90deg,var(--a1),var(--a2)) !important;
}
[data-testid="stSlider"] [role="slider"] {
  background:var(--a1) !important; border:2px solid white !important;
  box-shadow:0 2px 8px rgba(0,0,0,.15) !important;
}

[data-testid="stExpander"] {
  background:white; border:1px solid var(--border) !important;
  border-radius:12px !important;
}

#MainMenu, footer { visibility:hidden; }
header { visibility:visible !important; }
header [data-testid="stToolbar"] { visibility:hidden; }
.stDeployButton { display:none; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════ EMOTION META ═════════════════════════════════════
EMOTION_META = {
    "admiration":     {"emoji":"🌟","color":"#f4c97a","group":"Positive"},
    "amusement":      {"emoji":"😄","color":"#f9b87a","group":"Positive"},
    "anger":          {"emoji":"😠","color":"#e8a0a0","group":"Negative"},
    "annoyance":      {"emoji":"😤","color":"#e8b0a0","group":"Negative"},
    "approval":       {"emoji":"👍","color":"#a8d8b9","group":"Positive"},
    "caring":         {"emoji":"💙","color":"#a0c4e8","group":"Positive"},
    "confusion":      {"emoji":"😕","color":"#d4b8e8","group":"Ambiguous"},
    "curiosity":      {"emoji":"🔍","color":"#b0cfe8","group":"Positive"},
    "desire":         {"emoji":"✨","color":"#f4a7c0","group":"Positive"},
    "disappointment": {"emoji":"😞","color":"#c8bfb8","group":"Negative"},
    "disapproval":    {"emoji":"👎","color":"#e0a8a8","group":"Negative"},
    "disgust":        {"emoji":"🤢","color":"#c0c898","group":"Negative"},
    "embarrassment":  {"emoji":"😳","color":"#f0b8a0","group":"Negative"},
    "excitement":     {"emoji":"🚀","color":"#a8e0c8","group":"Positive"},
    "fear":           {"emoji":"😨","color":"#b8c0c8","group":"Negative"},
    "gratitude":      {"emoji":"🙏","color":"#a0d8d8","group":"Positive"},
    "grief":          {"emoji":"💔","color":"#b8b0c8","group":"Negative"},
    "joy":            {"emoji":"😊","color":"#f9d89a","group":"Positive"},
    "love":           {"emoji":"❤️", "color":"#f4a7b9","group":"Positive"},
    "nervousness":    {"emoji":"😰","color":"#b8d0e8","group":"Negative"},
    "optimism":       {"emoji":"🌈","color":"#a8e8c8","group":"Positive"},
    "pride":          {"emoji":"🏆","color":"#f4d07a","group":"Positive"},
    "realization":    {"emoji":"💡","color":"#c8b8e8","group":"Ambiguous"},
    "relief":         {"emoji":"😌","color":"#a8e0e0","group":"Positive"},
    "remorse":        {"emoji":"😔","color":"#b8c0c0","group":"Negative"},
    "sadness":        {"emoji":"😢","color":"#a8c0e0","group":"Negative"},
    "surprise":       {"emoji":"😲","color":"#f4b8d8","group":"Ambiguous"},
    "neutral":        {"emoji":"😐","color":"#d0c8c0","group":"Neutral"},
}
GRP_COL = {"Positive":"#a8d8b9","Negative":"#e8a0a0","Ambiguous":"#d4b8e8","Neutral":"#d0c8c0"}

# ══════════════════════════ SESSION STATE ════════════════════════════════════
for k,v in [("last_results",None),("last_text",""),("history",[]),
            ("batch_results",None),("compare_results",[])]:
    if k not in st.session_state: st.session_state[k] = v

# ══════════════════════════ MODEL ════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_model(d):
    from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
    tok = DistilBertTokenizerFast.from_pretrained(d)
    mdl = DistilBertForSequenceClassification.from_pretrained(d)
    mdl.eval(); return mdl, tok

def run_inference(text):
    labels = list(EMOTION_META.keys())
    mdl, tok = (load_model("./emotion_model_final")
                if os.path.exists("./emotion_model_final") else (None,None))
    if mdl:
        inp = tok(text, return_tensors='pt', truncation=True, max_length=64, padding=True)
        with torch.no_grad():
            probs = torch.sigmoid(mdl(**inp).logits).numpy()[0]
    else:
        import hashlib
        rng = np.random.RandomState(int(hashlib.md5(text.encode()).hexdigest(),16)%2**31)
        probs = rng.dirichlet(np.ones(len(labels))*.3)
        probs[:4] *= 3.5
        probs = np.clip(probs/probs.sum(),0,1)
    res = [{"emotion":labels[i],"confidence":float(probs[i])} for i in range(len(labels))]
    res.sort(key=lambda x:x["confidence"],reverse=True)
    return res

# ══════════════════════════ HELPERS ══════════════════════════════════════════
def dominant(results, thr):
    above = [r for r in results if r["confidence"]>=thr]
    return above[0] if above else results[0]

def sentiment(results, thr):
    s,t = 0,0
    for r in results:
        if r["confidence"]>=thr:
            g = EMOTION_META.get(r["emotion"],{}).get("group","Neutral")
            w = r["confidence"]
            s += w if g=="Positive" else (-w if g=="Negative" else 0)
            t += w
    if t==0: return "Neutral",0
    ratio=s/t
    return ("Positive" if ratio>.2 else "Negative" if ratio<-.2 else "Neutral"),ratio

def do_analyze(text, thr):
    results = run_inference(text)
    st.session_state.last_results = results
    st.session_state.last_text    = text
    dom = dominant(results, thr)
    snt,_ = sentiment(results, thr)
    st.session_state.history.append({
        "text":text,"results":results,
        "dominant":dom["emotion"],"sentiment":snt,
        "timestamp":datetime.now().strftime("%H:%M")
    })

# ══════════════════════════ CHARTS ═══════════════════════════════════════════
BG   = "rgba(250,247,244,0)"
FONT = dict(family="DM Sans", color="#3d3530")

def bar_chart(results, thr, top_n):
    top = [r for r in results if r["confidence"]>=thr][:top_n]
    if not top: top = results[:5]
    ems   = [r["emotion"] for r in top]
    confs = [r["confidence"] for r in top]
    cols  = [EMOTION_META.get(e,{}).get("color","#c9a0dc") for e in ems]
    fig = go.Figure(go.Bar(
        x=confs, y=ems, orientation='h',
        marker=dict(color=cols, line=dict(width=0), opacity=.85),
        text=[f"{c:.1%}" for c in confs], textposition='outside',
        textfont=dict(family="DM Sans",size=12,color="#3d3530"),
        hovertemplate="<b>%{y}</b><br>%{x:.2%}<extra></extra>"
    ))
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG, font=FONT,
        height=max(260,len(top)*38), margin=dict(l=5,r=75,t=8,b=8),
        xaxis=dict(range=[0,min(1.25,max(confs)*1.4)], showgrid=True,
                   gridcolor='rgba(0,0,0,.06)', tickformat='.0%',
                   color='#9c8f86', zeroline=False),
        yaxis=dict(autorange='reversed', color='#3d3530',
                   tickfont=dict(size=13), gridcolor='rgba(0,0,0,0)')
    )
    return fig

def radar_chart(results, thr, top_n):
    top = [r for r in results if r["confidence"]>=thr][:top_n]
    if not top: top = results[:6]
    labs = [r["emotion"] for r in top]
    vals = [r["confidence"] for r in top]
    fig = go.Figure(go.Scatterpolar(
        r=vals+[vals[0]], theta=labs+[labs[0]],
        fill='toself', fillcolor='rgba(201,160,220,.18)',
        line=dict(color='#c9a0dc',width=2),
        marker=dict(size=6,color='#f4a7b9'),
        hovertemplate="<b>%{theta}</b><br>%{r:.2%}<extra></extra>"
    ))
    fig.update_layout(
        polar=dict(bgcolor=BG,
                   angularaxis=dict(color='#9c8f86',gridcolor='rgba(0,0,0,.07)'),
                   radialaxis=dict(visible=True,range=[0,max(vals)*1.2],
                                   color='#9c8f86',gridcolor='rgba(0,0,0,.07)',
                                   tickformat='.0%')),
        paper_bgcolor=BG, font=FONT, height=320, margin=dict(l=25,r=25,t=25,b=25)
    )
    return fig

def pie_chart(results, thr):
    groups = {"Positive":0,"Negative":0,"Ambiguous":0,"Neutral":0}
    for r in results:
        if r["confidence"]>=thr:
            g = EMOTION_META.get(r["emotion"],{}).get("group","Neutral")
            groups[g] += r["confidence"]
    labels = [k for k,v in groups.items() if v>0]
    if not labels: return None
    fig = go.Figure(go.Pie(
        labels=labels, values=[groups[k] for k in labels],
        marker=dict(colors=[GRP_COL[k] for k in labels],
                    line=dict(color='#faf7f4',width=2)),
        textfont=dict(family="DM Sans",size=13,color="#3d3530"),
        hovertemplate="<b>%{label}</b><br>%{percent}<extra></extra>", hole=.52
    ))
    fig.update_layout(paper_bgcolor=BG, font=FONT, height=270,
                      margin=dict(l=10,r=10,t=10,b=10),
                      legend=dict(font=dict(size=12,color='#3d3530'),bgcolor='rgba(0,0,0,0)'))
    return fig

def heatmap_chart(results_list, texts, thr):
    labels = list(EMOTION_META.keys())
    cols = [e for e in labels if any(
        r["confidence"]>=.12 for res in results_list for r in res if r["emotion"]==e
    )][:14]
    if not cols: return None
    matrix = []
    for res in results_list:
        cm = {r["emotion"]:r["confidence"] for r in res}
        matrix.append([cm.get(e,0) for e in cols])
    short = [t[:28]+"…" if len(t)>28 else t for t in texts]
    fig = go.Figure(go.Heatmap(
        z=matrix, x=cols, y=short,
        colorscale=[[0,'#f7f3ef'],[.5,'#d4b8e8'],[1,'#c9a0dc']],
        hovertemplate="<b>%{y}</b><br>%{x}: %{z:.2%}<extra></extra>", zmin=0, zmax=1
    ))
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG, font=FONT,
        height=max(240,len(texts)*44), margin=dict(l=8,r=8,t=8,b=75),
        xaxis=dict(color='#9c8f86',tickangle=-35),
        yaxis=dict(color='#3d3530')
    )
    return fig

CFG = {"displayModeBar":False}

# ══════════════════════════ SIDEBAR ══════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:.3rem 0 1.2rem;'>
      <div style='font-family:"DM Serif Display",serif;font-size:1.5rem;color:#3d3530;'>
        🌸 EmoSense</div>
      <div style='color:#9c8f86;font-size:.75rem;margin-top:.2rem;'>Emotion Intelligence</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<p style='font-size:.68rem;text-transform:uppercase;letter-spacing:.1em;color:#9c8f86;font-weight:600;margin-bottom:.2rem;'>⚙️ Settings</p>", unsafe_allow_html=True)
    st.slider("Confidence Threshold", 0.05, 0.95, 0.30, 0.05, key="threshold", label_visibility="hidden")
    st.slider("Max Emotions to Show", 3, 15, 8, 1, key="top_n", label_visibility="hidden")

    st.markdown("<hr style='border:none;border-top:1px solid #e2d9d0;margin:.8rem 0;'>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:.68rem;text-transform:uppercase;letter-spacing:.1em;color:#9c8f86;font-weight:600;margin-bottom:.2rem;'>📊 Chart Style</p>", unsafe_allow_html=True)
    st.radio("chart_style", ["Bar Chart","Radar Chart","Both"],
             key="chart_type", horizontal=False, label_visibility="hidden")

    st.markdown("<hr style='border:none;border-top:1px solid #e2d9d0;margin:.8rem 0;'>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:.68rem;text-transform:uppercase;letter-spacing:.1em;color:#9c8f86;font-weight:600;margin-bottom:.4rem;'>💬 Quick Examples</p>", unsafe_allow_html=True)

    examples = [
        "I'm so proud of what we built together!",
        "This is absolutely terrifying, I can't stop shaking.",
        "Why would they do this? It makes no sense.",
        "I don't know how to feel about this news.",
        "Thank you so much, this made my entire week!",
    ]
    prev = st.session_state.get("_prev_ex", "")
    for ex in examples:
        if st.button(ex, key=f"ex_{ex[:15]}", use_container_width=True):
            st.session_state["main_ta"] = ex
            st.session_state["_prev_ex"] = ex
            st.rerun()

    st.markdown("<hr style='border:none;border-top:1px solid #e2d9d0;margin:.8rem 0;'>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:.68rem;text-transform:uppercase;letter-spacing:.1em;color:#9c8f86;font-weight:600;margin-bottom:.2rem;'>📂 Model</p>", unsafe_allow_html=True)
    if os.path.exists("./emotion_model_final"):
        st.markdown("<div style='color:#7dbf9e;font-size:.84rem;'>✅ Trained model ready</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='color:#e8a070;font-size:.82rem;line-height:1.6;'>⚠️ <b>Demo mode</b><br>Run model.ipynb first.</div>", unsafe_allow_html=True)

    if st.session_state.history:
        st.markdown("<hr style='border:none;border-top:1px solid #e2d9d0;margin:.8rem 0;'>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:.68rem;text-transform:uppercase;letter-spacing:.1em;color:#9c8f86;font-weight:600;margin-bottom:.3rem;'>🕒 Recent</p>", unsafe_allow_html=True)
        for h in st.session_state.history[-4:][::-1]:
            meta = EMOTION_META.get(h["dominant"],{})
            st.markdown(f"""<div style='background:white;border:1px solid #e2d9d0;
              border-left:3px solid {meta.get("color","#c9a0dc")};
              border-radius:10px;padding:.5rem .8rem;margin-bottom:.4rem;font-size:.82rem;'>
              {meta.get("emoji","")} <b style='color:{meta.get("color","#c9a0dc")};'>
              {h["dominant"].capitalize()}</b><br>
              <span style='color:#9c8f86;font-size:.75rem;'>{h["text"][:40]}…</span>
            </div>""", unsafe_allow_html=True)
        if st.button("🗑 Clear History", key="clr_hist"):
            st.session_state.history = []
            st.session_state.last_results = None
            st.rerun()

# live values
thr        = st.session_state.get("threshold", 0.30)
top_n      = st.session_state.get("top_n", 8)
chart_type = st.session_state.get("chart_type", "Bar Chart")

# ══════════════════════════ HERO ═════════════════════════════════════════════
st.markdown("""
<div class='hero-wrap'>
  <h1 class='hero-title'>Emotion <span>Recognition</span> AI</h1>
  <p class='hero-sub'>Uncover the emotional landscape of any text — 28 emotions, one model</p>
  <span class='hero-badge'>GoEmotions · DistilBERT · Multi-label</span>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["🔍 Analyze","🔄 Compare","📦 Batch","📈 Insights"])

# ══════════════════════════ TAB 1 ════════════════════════════════════════════
with tab1:
    L, R = st.columns([1, 1.1], gap="large")

    with L:
        st.markdown("<p style='font-size:.68rem;text-transform:uppercase;letter-spacing:.1em;color:#9c8f86;font-weight:600;margin-bottom:.4rem;'>✍️ Your Text</p>", unsafe_allow_html=True)

        if "main_ta" not in st.session_state:
            st.session_state["main_ta"] = ""
        text_in = st.text_area("Your Text",
                               placeholder="Type or paste anything — a message, review, tweet…",
                               height=155, key="main_ta", label_visibility="hidden")
        wc = len(text_in.split()) if text_in.strip() else 0
        st.markdown(f"<div style='color:#9c8f86;font-size:.77rem;'>{len(text_in)} chars · {wc} words</div>",
                    unsafe_allow_html=True)

        b1, b2 = st.columns([2,1])
        with b1:
            go_btn = st.button("🌸 Analyze Emotions", use_container_width=True, key="go_btn")
        with b2:
            if st.button("Clear", use_container_width=True, key="clr_btn"):
                st.session_state.last_results = None
                st.session_state.last_text    = ""
                st.session_state["main_ta"]   = ""
                st.rerun()

        if go_btn:
            if text_in.strip():
                with st.spinner("Reading emotions…"):
                    do_analyze(text_in.strip(), thr)
            else:
                st.warning("Please enter some text first.")

        # Emotion pills
        st.markdown("<p style='font-size:.68rem;text-transform:uppercase;letter-spacing:.1em;color:#9c8f86;font-weight:600;margin-bottom:.4rem;'>🎭 28 Emotion Classes</p>", unsafe_allow_html=True)
        pills = ""
        for em, meta in EMOTION_META.items():
            c = meta["color"]
            r2,g2,b2 = int(c[1:3],16),int(c[3:5],16),int(c[5:7],16)
            pills += (f"<span class='epill' style='background:rgba({r2},{g2},{b2},.22);"
                      f"color:#3d3530;border:1px solid rgba({r2},{g2},{b2},.5);'>"
                      f"{meta['emoji']} {em}</span>")
        st.markdown(pills, unsafe_allow_html=True)

        if st.session_state.history:
            st.markdown("<p style='font-size:.68rem;text-transform:uppercase;letter-spacing:.1em;color:#9c8f86;font-weight:600;margin:.2rem 0 .4rem;'>🕒 Recent</p>", unsafe_allow_html=True)
            for h in st.session_state.history[-3:][::-1]:
                meta = EMOTION_META.get(h["dominant"],{})
                st.markdown(f"""<div style='background:white;border:1px solid #e2d9d0;
                  border-left:3px solid {meta.get("color","#c9a0dc")};
                  border-radius:10px;padding:.5rem .8rem;margin-bottom:.4rem;font-size:.84rem;'>
                  {meta.get("emoji","")} <b style='color:{meta.get("color","#c9a0dc")};'>
                  {h["dominant"].capitalize()}</b>
                  <span style='color:#9c8f86;font-size:.75rem;'> · {h["timestamp"]}</span><br>
                  <span style='color:#9c8f86;font-size:.75rem;'>{h["text"][:50]}…</span>
                </div>""", unsafe_allow_html=True)

    with R:
        results = st.session_state.last_results
        if results is not None:
            dom_r    = dominant(results, thr)
            dom_meta = EMOTION_META.get(dom_r["emotion"],{"emoji":"🎭","color":"#c9a0dc"})
            snt, _   = sentiment(results, thr)
            snt_col  = {"Positive":"#7dbf9e","Negative":"#e8a0a0","Neutral":"#b8b0c8"}[snt]
            above    = [r for r in results if r["confidence"]>=thr]

            dc = dom_meta["color"]
            dr,dg,db = int(dc[1:3],16),int(dc[3:5],16),int(dc[5:7],16)
            st.markdown(f"""
            <div class='card' style='border-color:{dc};
                 background:linear-gradient(135deg,white,rgba({dr},{dg},{db},.08));
                 text-align:center;padding:1.5rem;'>
              <div style='font-size:3rem;line-height:1;'>{dom_meta["emoji"]}</div>
              <div style='font-family:"DM Serif Display",serif;font-size:1.5rem;color:#3d3530;margin-top:.4rem;'>
                {dom_r["emotion"].capitalize()}</div>
              <div style='color:#9c8f86;font-size:.82rem;margin-top:.25rem;'>
                Dominant emotion · {dom_r["confidence"]:.1%} confidence</div>
            </div>""", unsafe_allow_html=True)

            m1,m2,m3 = st.columns(3)
            with m1:
                st.markdown(f"""<div class='mbox'><div class='mval'>{len(above)}</div>
                  <div class='mlbl'>Detected</div></div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""<div class='mbox'>
                  <div class='mval' style='color:{snt_col};font-size:1.3rem;'>{snt}</div>
                  <div class='mlbl'>Sentiment</div></div>""", unsafe_allow_html=True)
            with m3:
                st.markdown(f"""<div class='mbox'><div class='mval'>{dom_r["confidence"]:.0%}</div>
                  <div class='mlbl'>Top Score</div></div>""", unsafe_allow_html=True)

            st.markdown("<div style='height:.5rem;'></div>", unsafe_allow_html=True)

            if chart_type == "Both":
                ca,cb = st.columns(2)
                with ca:
                    st.markdown("<p class='sec-lbl'>Confidence Scores</p>", unsafe_allow_html=True)
                    st.plotly_chart(bar_chart(results,thr,top_n), use_container_width=True, config=CFG)
                with cb:
                    st.markdown("<p class='sec-lbl'>Radar View</p>", unsafe_allow_html=True)
                    st.plotly_chart(radar_chart(results,thr,top_n), use_container_width=True, config=CFG)
            elif chart_type == "Bar Chart":
                st.markdown("<p class='sec-lbl'>Confidence Scores</p>", unsafe_allow_html=True)
                st.plotly_chart(bar_chart(results,thr,top_n), use_container_width=True, config=CFG)
            else:
                st.markdown("<p class='sec-lbl'>Radar View</p>", unsafe_allow_html=True)
                st.plotly_chart(radar_chart(results,thr,top_n), use_container_width=True, config=CFG)

            p = pie_chart(results, thr)
            if p:
                st.markdown("<p class='sec-lbl'>Sentiment Breakdown</p>", unsafe_allow_html=True)
                st.plotly_chart(p, use_container_width=True, config=CFG)

            with st.expander("📋 Full Emotion Table"):
                df = pd.DataFrame(results)
                df["emoji"] = df["emotion"].apply(lambda e: EMOTION_META.get(e,{}).get("emoji",""))
                df["group"] = df["emotion"].apply(lambda e: EMOTION_META.get(e,{}).get("group",""))
                df["conf%"] = (df["confidence"]*100).round(2)
                df["✓"]    = df["confidence"].apply(lambda c: "✓" if c>=thr else "")
                st.dataframe(df[["emoji","emotion","conf%","group","✓"]].rename(columns={
                    "emoji":"","emotion":"Emotion","conf%":"Conf %","group":"Category","✓":"≥ Thr"
                }), use_container_width=True, hide_index=True)
        else:
            st.markdown("""
            <div style='display:flex;align-items:center;justify-content:center;
                        height:360px;flex-direction:column;gap:1rem;'>
              <div style='font-size:4rem;opacity:.2;'>🌸</div>
              <div style='color:#9c8f86;text-align:center;font-size:.92rem;'>
                Enter text and click <b>Analyze Emotions</b><br>to see results
              </div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════ TAB 2 ════════════════════════════════════════════
with tab2:
    st.markdown("<p style='font-size:.68rem;text-transform:uppercase;letter-spacing:.1em;color:#9c8f86;font-weight:600;margin-bottom:.6rem;'>🔄 Compare Two Texts Side-by-Side</p>", unsafe_allow_html=True)
    ca,cb = st.columns(2, gap="medium")
    with ca:
        st.markdown("<p style='color:#c9a0dc;font-size:.82rem;font-weight:600;margin-bottom:.2rem;'>Text A</p>", unsafe_allow_html=True)
        ct1 = st.text_area("Text A", placeholder="First text…", height=115, label_visibility="hidden")
    with cb:
        st.markdown("<p style='color:#f4a7b9;font-size:.82rem;font-weight:600;margin-bottom:.2rem;'>Text B</p>", unsafe_allow_html=True)
        ct2 = st.text_area("Text B", placeholder="Second text…", height=115, label_visibility="hidden")
    cmp_btn = st.button("🌸 Compare", key="cmp_btn")

    if cmp_btn and ct1.strip() and ct2.strip():
        with st.spinner("Analyzing…"):
            r1,r2 = run_inference(ct1), run_inference(ct2)
        st.session_state.compare_results = [(ct1,r1),(ct2,r2)]

    if st.session_state.compare_results:
        (t1,r1),(t2,r2) = st.session_state.compare_results
        d1,d2 = dominant(r1,thr), dominant(r2,thr)
        m1m = EMOTION_META.get(d1["emotion"],{"emoji":"🎭","color":"#c9a0dc"})
        m2m = EMOTION_META.get(d2["emotion"],{"emoji":"🎭","color":"#f4a7b9"})
        c1,c2 = st.columns(2, gap="medium")
        with c1:
            st.markdown(f"""<div class='card' style='border-color:{m1m["color"]};text-align:center;'>
              <div style='font-size:2.4rem;'>{m1m["emoji"]}</div>
              <div style='font-family:"DM Serif Display",serif;font-size:1.25rem;color:#3d3530;'>
                {d1["emotion"].capitalize()}</div>
              <div style='color:#9c8f86;font-size:.8rem;'>{d1["confidence"]:.1%} · Text A</div>
            </div>""", unsafe_allow_html=True)
            st.plotly_chart(bar_chart(r1,thr,top_n), use_container_width=True, config=CFG)
        with c2:
            st.markdown(f"""<div class='card' style='border-color:{m2m["color"]};text-align:center;'>
              <div style='font-size:2.4rem;'>{m2m["emoji"]}</div>
              <div style='font-family:"DM Serif Display",serif;font-size:1.25rem;color:#3d3530;'>
                {d2["emotion"].capitalize()}</div>
              <div style='color:#9c8f86;font-size:.8rem;'>{d2["confidence"]:.1%} · Text B</div>
            </div>""", unsafe_allow_html=True)
            st.plotly_chart(bar_chart(r2,thr,top_n), use_container_width=True, config=CFG)
        hm = heatmap_chart([r1,r2],["Text A","Text B"],thr)
        if hm:
            st.markdown("<p class='sec-lbl' style='margin-top:.8rem;'>Emotion Matrix</p>", unsafe_allow_html=True)
            st.plotly_chart(hm, use_container_width=True, config=CFG)

# ══════════════════════════ TAB 3 ════════════════════════════════════════════
with tab3:
    st.markdown("<p style='font-size:.68rem;text-transform:uppercase;letter-spacing:.1em;color:#9c8f86;font-weight:600;margin-bottom:.6rem;'>📦 Batch — One Text Per Line (max 20)</p>", unsafe_allow_html=True)
    batch_in = st.text_area("Batch input",
                             placeholder="I love this!\nThis is terrible.\nNot sure…",
                             height=170, label_visibility="hidden")
    bat_btn = st.button("🌸 Analyze All", key="bat_btn")

    if bat_btn and batch_in.strip():
        texts = [t.strip() for t in batch_in.strip().split('\n') if t.strip()][:20]
        prog = st.progress(0, text="Analyzing…")
        all_res = []
        for i,t in enumerate(texts):
            all_res.append(run_inference(t))
            prog.progress((i+1)/len(texts), text=f"{i+1}/{len(texts)}")
        prog.empty()
        st.session_state.batch_results = (texts, all_res)

    if st.session_state.batch_results:
        texts, all_res = st.session_state.batch_results
        rows = []
        for t,res in zip(texts,all_res):
            dm   = dominant(res,thr)
            meta = EMOTION_META.get(dm["emotion"],{})
            snt,_ = sentiment(res,thr)
            rows.append({
                "Text": t[:58]+("…" if len(t)>58 else ""),
                "Dominant": f"{meta.get('emoji','')} {dm['emotion'].capitalize()}",
                "Confidence": f"{dm['confidence']:.1%}",
                "Sentiment": snt,
                "Detected": len([r for r in res if r["confidence"]>=thr])
            })
        st.markdown("<p class='sec-lbl'>Summary</p>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        hm = heatmap_chart(all_res, texts, thr)
        if hm:
            st.markdown("<p class='sec-lbl' style='margin-top:.8rem;'>Emotion Heatmap</p>", unsafe_allow_html=True)
            st.plotly_chart(hm, use_container_width=True, config=CFG)
        csv = pd.DataFrame(rows).to_csv(index=False).encode()
        st.download_button("📥 Download CSV", csv, "emotions.csv", "text/csv")

# ══════════════════════════ TAB 4 ════════════════════════════════════════════
with tab4:
    if not st.session_state.history:
        st.markdown("""
        <div style='text-align:center;padding:3rem;color:#9c8f86;'>
          <div style='font-size:3rem;opacity:.25;'>🌸</div>
          <div style='margin-top:.8rem;'>Run some analyses to see insights here.</div>
        </div>""", unsafe_allow_html=True)
    else:
        hist  = st.session_state.history
        sents = [h["sentiment"] for h in hist]
        pos,neg,neu = sents.count("Positive"),sents.count("Negative"),sents.count("Neutral")

        i1,i2,i3,i4 = st.columns(4)
        for col,val,lbl,clr in [(i1,len(hist),"Analyses","#c9a0dc"),
                                 (i2,pos,"Positive","#7dbf9e"),
                                 (i3,neg,"Negative","#e8a0a0"),
                                 (i4,neu,"Neutral","#b8b0c8")]:
            with col:
                col.markdown(f"""<div class='mbox'>
                  <div class='mval' style='color:{clr};'>{val}</div>
                  <div class='mlbl'>{lbl}</div></div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
        dc  = Counter(h["dominant"] for h in hist)
        ddf = pd.DataFrame(dc.most_common(10), columns=["emotion","count"])
        ddf["color"] = ddf["emotion"].apply(lambda e: EMOTION_META.get(e,{}).get("color","#c9a0dc"))

        st.markdown("<p class='sec-lbl'>Top Emotions This Session</p>", unsafe_allow_html=True)
        fig2 = go.Figure(go.Bar(
            x=ddf["emotion"], y=ddf["count"], marker_color=ddf["color"], marker_opacity=.85,
            text=ddf["count"], textposition='outside', textfont=dict(color="#3d3530"),
            hovertemplate="<b>%{x}</b><br>%{y}<extra></extra>"
        ))
        fig2.update_layout(
            paper_bgcolor=BG, plot_bgcolor=BG, font=FONT, height=270, showlegend=False,
            margin=dict(l=8,r=8,t=8,b=40),
            xaxis=dict(color='#3d3530',gridcolor='rgba(0,0,0,.05)'),
            yaxis=dict(color='#9c8f86',gridcolor='rgba(0,0,0,.05)')
        )
        st.plotly_chart(fig2, use_container_width=True, config=CFG)

        if len(hist)>1:
            st.markdown("<p class='sec-lbl'>Session Emotion Matrix</p>", unsafe_allow_html=True)
            hm = heatmap_chart([h["results"] for h in hist],
                               [h["text"][:28] for h in hist], thr)
            if hm:
                st.plotly_chart(hm, use_container_width=True, config=CFG)
