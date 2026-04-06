"""
Seven Deadly Sins in Gemma 2 2B - Sinclopedia

(c) Aritro 'sortira' Shome
"""
import streamlit as st

st.set_page_config(page_title="Sinclopedia", layout="wide")
st.title("Seven Deadly Sins in Gemma 2 2B Representation Space")

st.markdown("""
## Exploring Sin Vectors in Large Language Models

This application demonstrates how to extract and manipulate "sin vectors" from the hidden 
representations of Gemma 2 2B, enabling both analysis and *steering* of model behavior.

### What are Sin Vectors?

Using a corpus of emotionally-charged stories annotated with the seven deadly sins, we extracted
direction vectors in the model's representation space that correspond to each sin. These vectors
allow us to:

1. **Analyze** - Detect which sins are active in a given text
2. **Steer** - Influence model completions toward or away from sinful behavior

---

## Available Tools

Use the navigation menu to access different features:
""")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Sin Analyzer")
    st.markdown("""
    Detect and visualize which sins activate most strongly on your text.
    
    - See token-level projections for each sin
    - Explore cross-layer activations
    - View the "Sin-O-Meter" composition chart
    
    **Start:** Click "Sin Analyzer" in the menu →
    """)

with col2:
    st.subheader("Sin Steering")
    st.markdown("""
    Steer the model's completions toward or away from specific sins.
    
    - Choose a prompt and a target sin
    - Adjust steering strength (alpha)
    - See how completions change
    
    **Start:** Click "Sin Steering" in the menu →
    """)

st.markdown("---")

st.subheader("How It Works")

st.markdown("""
### Feature Extraction (Sin Analyzer)
1. Text is passed through Gemma 2 2B
2. Residual stream activations are extracted at the target layer
3. Cosine similarity computed against each sin's vector
4. Results are visualized as heatmaps and charts

### Model Steering (Sin Steering)
1. A prompt is given to the model
2. During generation, a steering hook is activated
3. At each step: x_new = x_original + (α · v_sin)
4. This biases the model toward/away from the selected sin
""")

st.markdown("---")

st.subheader("About the Sins")

sins_info = {
    "Pride": "Excessive self-regard, arrogance",
    "Envy": "Desire for what others possess",
    "Gluttony": "Overconsumption, excessive indulgence", 
    "Sloth": "Laziness, lack of effort",
    "Wrath": "Anger, rage, violence",
    "Lust": "Excessive sexual desire",
    "Greed": "Excessive desire for gain"
}

cols = st.columns(4)
for idx, (sin, desc) in enumerate(sins_info.items()):
    with cols[idx % 4]:
        st.markdown(f"**{sin}**  \n{desc}")

st.markdown("---")

st.sidebar.markdown("""
### Navigation
Use the menu at the top-left to switch between pages:
- **Sin Analyzer** - Text analysis
- **Sin Steering** - Model steering

### Settings
Each page has its own configuration panel in the sidebar.
""")
