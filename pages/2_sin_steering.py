"""
Sin Steering - Steer model completions toward/away from sins.

(c) Aritro 'sortira' Shome
"""
import streamlit as st
import torch
from transformer_lens import HookedTransformer
import os

st.set_page_config(page_title="Sin Steering", layout="wide")
st.title("Sin Steering")

st.markdown("""
Steer the model's generation toward or away from specific sins by injecting sin vectors into the residual stream.

**The Math:** x_new = x_original + (α · v_sin)
""")

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
steering_layer = st.sidebar.slider("Steering Layer", min_value=0, max_value=model.cfg.n_layers-1, value=17)
max_new_tokens = st.sidebar.slider("Max Tokens to Generate", min_value=10, max_value=512, value=50)

st.subheader("Prompt")
prompt = st.text_area("Enter a prompt:", height=100, placeholder="The man walked into the crowded room and saw his rival...")

st.subheader("Steering Configuration")

col1, col2 = st.columns(2)

with col1:
    selected_sin = st.selectbox("Select Sin to Steer Toward", options=SINS)

with col2:
    steering_strength = st.slider("Steering Strength (alpha)", min_value=-10.0, max_value=10.0, value=5.0, step=0.5)

generate_button = st.button("Generate", type="primary")

if prompt and generate_button:
    with st.spinner("Generating completion..."):
        sin_vec = sin_vectors[selected_sin][steering_layer]
        
        def steering_hook(resid, hook):
            return resid + steering_strength * sin_vec
        
        try:
            with model.hooks([(f"blocks.{steering_layer}.hook_resid_pre", steering_hook)]):
                completion_text = model.generate(prompt, max_new_tokens=max_new_tokens, temperature=0.7, top_p=0.9, do_sample=True)
            
            st.subheader("Results")
            
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.metric("Sin", selected_sin.upper())
            with col2:
                st.metric("Alpha", f"{steering_strength:.1f}")
            with col3:
                st.metric("Layer", steering_layer)
            
            st.subheader("Steered Completion")
            
            prompt_length = len(prompt)
            generated_part = completion_text[prompt_length:]
            
            html_result = f"""
            <div style="padding: 15px; background-color: #f0f0f0; border-radius: 5px; line-height: 1.8; color: #000;">
            <span>{prompt}</span>
            <span style="font-weight: bold;">{generated_part}</span>
            </div>
            """
            st.markdown(html_result, unsafe_allow_html=True)
            
            st.write("Full completion:")
            st.code(completion_text)
            
        except Exception as e:
            st.error(f"Error during generation: {str(e)}")

else:
    st.info("Enter a prompt and click Generate to steer the model")

st.sidebar.markdown("---")
st.sidebar.subheader("Sin Steering Strength Guide")
st.sidebar.markdown("""
- **Negative Alpha**: Steer *away* from the sin
- **Alpha = 0**: No steering (neutral)
- **Positive Alpha**: Steer *toward* the sin
- **|Alpha| > 5**: Strong effect, may distort output

Examples:
- Wrath (5): Aggressive language
- Pride (5): Arrogant tone
- Sloth (-5): Lazy, unmotivated tone
""")
