"""
Sin Analyzer - Visualize how sinful vectors activate on text.

(c) Aritro 'sortira' Shome
"""
import streamlit as st
import torch
import numpy as np
import pandas as pd
from transformer_lens import HookedTransformer
import os

st.set_page_config(page_title="Sin Analyzer", layout="wide")
st.title("Sin Analyzer")

SIN_COLORS = {
    "pride": "#FF6B6B",
    "envy": "#A8E6CF",
    "gluttony": "#FFD93D",
    "sloth": "#6B7280",
    "wrath": "#8B0000",
    "lust": "#FF69B4",
    "greed": "#FFB703",
}

SINS = list(SIN_COLORS.keys())

@st.cache_resource
def load_model():
    return HookedTransformer.from_pretrained("google/gemma-2-2b")

@st.cache_resource
def load_vectors():
    vectors = {}
    vector_path = "data/sinful-vectors-clean"
    
    for sin in SINS:
        vec_file = os.path.join(vector_path, f"{sin}.pt")
        if os.path.exists(vec_file):
            vectors[sin] = torch.load(vec_file)
        else:
            st.error(f"Missing {sin} vector at {vec_file}")
    
    return vectors

model = load_model()
sin_vectors = load_vectors()

st.sidebar.header("Settings")
target_layer = st.sidebar.slider("Target Layer", min_value=0, max_value=model.cfg.n_layers-1, value=17)
percentile_threshold = st.sidebar.slider("Highlight Percentile", min_value=50, max_value=99, value=90)
st.subheader("Input Text")
user_text = st.text_area("Enter a story or text:", height=150, placeholder="The king sat on his throne, blinded by his own pride...")

submit_button = st.button("Analyze", type="primary")

if user_text and submit_button:
    with st.spinner("Processing text..."):
        logits, cache = model.run_with_cache(user_text)
        resid = cache[f"blocks.{target_layer}.hook_resid_pre"]
        resid = resid.squeeze(0)
        
        tokens = model.to_str_tokens(user_text)
        
        projections = {}
        for sin in SINS:
            sin_vec = sin_vectors[sin][target_layer]
            scores = torch.matmul(resid, sin_vec)
            projections[sin] = scores.detach().cpu().numpy()
    
    st.subheader("Sin Heatmap")
    sin_tabs = st.tabs(SINS)
    
    for tab_idx, sin in enumerate(SINS):
        with sin_tabs[tab_idx]:
            scores = projections[sin]
            threshold = np.percentile(scores, percentile_threshold)
            
            min_score = scores.min()
            max_score = scores.max()
            normalized = (scores - min_score) / (max_score - min_score + 1e-8)
            
            color = SIN_COLORS[sin]
            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)
            
            html_parts = []
            for token, score, norm in zip(tokens, scores, normalized):
                is_highlighted = score >= threshold
                opacity = norm if is_highlighted else 0.1
                html_parts.append(f'<span style="background-color:rgba({r},{g},{b},{opacity});padding:2px 4px;border-radius:3px;margin:2px;">{token}</span>')
            
            html_text = '<div style="line-height:2.0;word-wrap:break-word;">' + ''.join(html_parts) + '</div>'
            st.markdown(html_text, unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean Score", f"{scores.mean():.3f}")
            with col2:
                st.metric("Threshold (90th %ile)", f"{threshold:.3f}")
            with col3:
                st.metric("Max Score", f"{scores.max():.3f}")
            with col4:
                st.metric("Highlighted Tokens", int((scores >= threshold).sum()))
    
    st.subheader("Sin-O-Meter")
    
    sin_scores = {}
    for sin in SINS:
        sin_scores[sin] = projections[sin].mean()
    
    total = sum(abs(s) for s in sin_scores.values())
    if total > 0:
        sin_percentages = {sin: abs(sin_scores[sin]) / total * 100 for sin in SINS}
    else:
        sin_percentages = {sin: 100 / len(SINS) for sin in SINS}
    
    sin_df = pd.DataFrame(list(sin_percentages.items()), columns=["Sin", "Percentage"])
    sin_df = sin_df.sort_values("Percentage", ascending=False)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.bar_chart(sin_df.set_index("Sin")["Percentage"])
    
    with col2:
        dominant_sin = max(sin_percentages, key=sin_percentages.get)
        st.metric("Dominant Sin", dominant_sin.upper(), f"{sin_percentages[dominant_sin]:.1f}%")
    
    st.subheader("Token Details")
    
    token_data = {"Token": tokens}
    for sin in SINS:
        token_data[sin] = projections[sin]
    
    df = pd.DataFrame(token_data)
    st.dataframe(df, use_container_width=True)
    
else:
    st.info("Enter some text and click Analyze to see sin activations")
