import streamlit as st
import numpy as np
import pandas as pd
import json
import pickle
import os
import re
import nltk
from nltk.corpus import stopwords
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Hate Speech Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hide Streamlit elements and set custom CSS for the dark theme
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .stApp {
        background-color: #0d0d0d;
        color: #ffffff;
    }
    .metric-card {
        background-color: #1a1a1a;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.4);
        text-align: center;
        border: 1px solid #333;
    }
    .predict-card-hate {
        background: linear-gradient(135deg, #1f0101, #3b0000);
        padding: 30px;
        border-radius: 12px;
        border-left: 5px solid #ff4b4b;
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.2);
    }
    .predict-card-clean {
        background: linear-gradient(135deg, #011f0c, #003b17);
        padding: 30px;
        border-radius: 12px;
        border-left: 5px solid #00ea65;
        box-shadow: 0 4px 15px rgba(0, 234, 101, 0.2);
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_nltk_resources():
    try:
        nltk.download('stopwords')
        nltk.download('punkt')
    except Exception as e:
        st.warning(f"Could not download NLTK data fully: {e}")

load_nltk_resources()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    try:
        stop_words = set(stopwords.words('english'))
        words = text.split()
        words = [w for w in words if not w in stop_words]
        return " ".join(words)
    except:
        return text

@st.cache_resource
def load_model_and_tokenizer():
    try:
        # Load primary Bi-LSTM model and tokenizer for predictions
        with open('model_bilstm.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('tokenizer_bilstm.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        with open('model_config.json', 'r') as f:
            config = json.load(f)
        return model, tokenizer, config
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

@st.cache_resource
def load_all_models():
    """Load all three models for comparison"""
    try:
        with open('model_bilstm.pkl', 'rb') as f:
            bilstm = pickle.load(f)
        with open('model_lr.pkl', 'rb') as f:
            lr = pickle.load(f)
        with open('model_mlp.pkl', 'rb') as f:
            mlp = pickle.load(f)
        with open('tokenizer_bilstm.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        return {'Bi-LSTM': bilstm, 'Logistic Regression': lr, 'MLP': mlp}, tokenizer
    except Exception as e:
        return None, None

@st.cache_data
def load_metrics():
    try:
        with open('metrics.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        return None

# --- UI LAYOUT ---
st.title("🛡️ Hate Speech Detection ")
st.markdown("**Using Bi-LSTM with Logistic Regression & MLP Comparison** ")

st.markdown("---")

# MAIN COLUMNS
col1, col2 = st.columns([1.5, 1])

# PREDICTION SECTION
with col1:
    st.subheader("Analyze Text")
    st.caption("Using Bi-LSTM model for text analysis")
    user_input = st.text_area("Enter your sentence, comment, or message:", height=150, placeholder="Type here...")
    predict_btn = st.button("Predict ", use_container_width=True)
    
    if predict_btn and user_input.strip() != "":
        model, tokenizer, config = load_model_and_tokenizer()
        if model is None or tokenizer is None or config is None:
            st.error("Model or Tokenizer not correctly loaded. Please run train.py first.")
        else:
            with st.spinner("Analyzing text..."):
                cleaned = clean_text(user_input)
                
                # Transform using TF-IDF vectorizer for Bi-LSTM model
                vec = tokenizer.transform([cleaned])
                
                # Predict using Bi-LSTM model
                prediction_prob = model.predict_proba(vec)[0][1]  # Probability of class 1 (Abusive)
                pred_class = 1 if prediction_prob > 0.5 else 0
                
                if pred_class == 1:
                    conf = prediction_prob * 100
                    st.markdown(f"""
                        <div class="predict-card-hate">
                            <h2 style="margin:0; color:#ff4b4b;">⚠️ Hate Speech / Abusive</h2>
                            <p style="margin-top:10px; font-size:18px;">Confidence: <strong>{conf:.2f}%</strong></p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    conf = (1 - prediction_prob) * 100
                    st.markdown(f"""
                        <div class="predict-card-clean">
                            <h2 style="margin:0; color:#00ea65;">✅ Not Hate Speech</h2>
                            <p style="margin-top:10px; font-size:18px;">Confidence: <strong>{conf:.2f}%</strong></p>
                        </div>
                    """, unsafe_allow_html=True)

# METRICS AND VISUALIZATION
metrics_data = load_metrics()

with col2:
    st.subheader("Model Stats")
    if metrics_data and "Bi-LSTM" in metrics_data:
        m = metrics_data["Bi-LSTM"]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
                <div class="metric-card">
                    <h4 style="margin:0; color:#8e8e8e;">Accuracy</h4>
                    <h2 style="margin:0; color:#ffffff;">{m['Accuracy']*100:.2f}%</h2>
                </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
                <div class="metric-card">
                    <h4 style="margin:0; color:#8e8e8e;">Precision</h4>
                    <h2 style="margin:0; color:#ffffff;">{m['Precision']*100:.2f}%</h2>
                </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
                <div class="metric-card">
                    <h4 style="margin:0; color:#8e8e8e;">Recall</h4>
                    <h2 style="margin:0; color:#ffffff;">{m['Recall']*100:.2f}%</h2>
                </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
                <div class="metric-card">
                    <h4 style="margin:0; color:#8e8e8e;">F1-Score</h4>
                    <h2 style="margin:0; color:#ffffff;">{m['F1-score']*100:.2f}%</h2>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No metrics generated yet. Run train.py first.")

st.markdown("---")

# EXTENDED VISUALIZATIONS
if metrics_data:
    st.subheader("Model Comparisons & Visualizations")
    
    # Model info
    with st.expander("Model Information"):
        st.markdown("""
        **Three-Model Ensemble Approach:**
        - **Bi-LSTM (Primary)**: Deep neural network for bidirectional text analysis
        - **Logistic Regression**: Conservative model to correct false positives
        - **MLP**: Standard neural network baseline comparison
        """)
    
    tcol1, tcol2 = st.columns(2)
    
    with tcol1:
        st.markdown("##### Confusion Matrix (Chosen Model)")
        if "cm" in metrics_data:
            cm = np.array(metrics_data["cm"])
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Reds',
                              labels=dict(x="Predicted", y="Actual", color="Count"),
                              x=['Not Hate (0)', 'Hate (1)'],
                              y=['Not Hate (0)', 'Hate (1)'])
            fig_cm.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color="white",
                margin=dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            
    with tcol2:
        st.markdown("##### Model Comparison (All Metrics)")
        models = [k for k in metrics_data.keys() if k != "cm"]
        
        # Create dataframe with all metrics
        comparison_data = []
        for model_name in models:
            m = metrics_data[model_name]
            comparison_data.append({
                'Model': model_name,
                'Accuracy': m['Accuracy'],
                'Precision': m['Precision'],
                'Recall': m['Recall'],
                'F1-Score': m['F1-score']
            })
        
        df_comp = pd.DataFrame(comparison_data)
        
        fig_bar = go.Figure()
        for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
            fig_bar.add_trace(go.Bar(
                x=df_comp['Model'], 
                y=df_comp[metric], 
                name=metric
            ))
        
        fig_bar.update_layout(
            barmode='group',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color="white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_title="",
            yaxis_title="Score"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")
st.markdown("<p style='text-align: center; color: #888;'>Developed as a stable end-to-end Machine Learning web application.</p>", unsafe_allow_html=True)

# DETAILED METRICS TABLE
st.markdown("---")
st.subheader("Detailed Model Metrics")

if metrics_data:
    models = [k for k in metrics_data.keys() if k != "cm"]
    
    # Create detailed metrics table
    metrics_table_data = []
    for model_name in models:
        m = metrics_data[model_name]
        metrics_table_data.append({
            'Model': model_name,
            'Accuracy': f"{m['Accuracy']*100:.2f}%",
            'Precision': f"{m['Precision']*100:.2f}%",
            'Recall': f"{m['Recall']*100:.2f}%",
            'F1-Score': f"{m['F1-score']*100:.2f}%"
        })
    
    df_metrics = pd.DataFrame(metrics_table_data)
    st.dataframe(df_metrics, use_container_width=True, hide_index=True)
