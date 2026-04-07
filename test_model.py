import pickle
import re
import json
import nltk
from nltk.corpus import stopwords

# Ensure NLTK resources
nltk.download('stopwords', quiet=True)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def test_model():
    # Load model configuration
    with open('model_config.json', 'r') as f:
        config = json.load(f)
    
    # Load all three models and tokenizer
    with open('model_bilstm.pkl', 'rb') as f:
        bilstm_model = pickle.load(f)
    with open('model_lr.pkl', 'rb') as f:
        lr_model = pickle.load(f)
    with open('model_mlp.pkl', 'rb') as f:
        mlp_model = pickle.load(f)
    with open('tokenizer_bilstm.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    
    test_cases = [
        "I like her",
        "the movie is bad",
        "you are a complete idiot",
        "go away"
    ]
    
    print("\n" + "="*70)
    print("TESTING ALL THREE MODELS: Bi-LSTM, Logistic Regression, MLP")
    print("="*70)
    
    for text in test_cases:
        cleaned = clean_text(text)
        
        # Transform using TF-IDF
        vec = tokenizer.transform([cleaned])
        
        # Get predictions from all models
        prob_bilstm = bilstm_model.predict_proba(vec)[0][1]
        label_bilstm = "Hate/Abusive" if prob_bilstm > 0.5 else "Clean"
        
        prob_lr = lr_model.predict_proba(vec)[0][1]
        label_lr = "Hate/Abusive" if prob_lr > 0.5 else "Clean"
        
        prob_mlp = mlp_model.predict_proba(vec)[0][1]
        label_mlp = "Hate/Abusive" if prob_mlp > 0.5 else "Clean"
        
        print(f"\nText: '{text}'")
        print(f"  Cleaned: '{cleaned}'")
        print(f"  ─────────────────────────────────────────")
        print(f"  Bi-LSTM (PRIMARY):      {label_bilstm:15} ({prob_bilstm:.4f})")
        print(f"  Logistic Regression:    {label_lr:15} ({prob_lr:.4f})")
        print(f"  MLP:                    {label_mlp:15} ({prob_mlp:.4f})")
        print("  " + "-"*50)

if __name__ == "__main__":
    test_model()
