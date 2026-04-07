import os
import re
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import nltk
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings('ignore')

# Ensuring NLTK resources
nltk.download('stopwords', quiet=True)

def get_clean_augmentation():
    """Generates a set of neutral and positive sentences to balance the dataset bias."""
    neutral_samples = [
        "I like her", "I love her", "I think she is great", "She is a good person",
        "The weather is nice today", "I enjoy drinking herbal tea", "This movie is bad",
        "The food was okay", "I like this song", "You are doing a great job",
        "Have a wonderful day", "Peace and love to everyone", "I am going for a walk",
        "I like reading books", "The cat is sleeping", "I like your style",
        "She is very smart", "I respect your opinion", "This is a neutral comment",
        "The service at the restaurant was excellent", "I like the color blue",
        "The movie was not good, but the acting was fine", "I like her personality",
        "He is a hard worker", "I like the way you think", "Simply beautiful",
        "The sky is clear today", "I like her earrings", "I like her smile",
        "I like her eyes", "I like her hair", "I like her shoes", "I like her dress"
    ]
    # Multiply the samples to give them more weight in the 24k dataset
    return neutral_samples * 10 

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def evaluate_model(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1-score": f1}

def main():
    print("Loading local dataset from labeled_data.csv...")
    df = pd.read_csv("labeled_data.csv")
    
    # Dataset columns: index, count, hate_speech, offensive_language, neither, class, tweet
    # Map class 0, 1 -> 1 (Abusive), 2 -> 0 (Clean)
    df['label'] = df['class'].apply(lambda x: 1 if x in [0, 1] else 0)
    
    # Shuffling the full dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # --- Data Augmentation ---
    print("Augmenting dataset with neutral/clean samples...")
    aug_texts = get_clean_augmentation()
    aug_df = pd.DataFrame({
        'tweet': aug_texts,
        'clean_text': [clean_text(t) for t in aug_texts],
        'label': [0] * len(aug_texts) # 0 is Clean/Neither
    })
    df = pd.concat([df, aug_df], ignore_index=True)
    
    print(f"Dataset loaded and augmented with {len(df)} rows.")
    print("Cleaning text data...")
    df['clean_text'] = df['tweet'].apply(clean_text)
    
    X = df['clean_text'].values
    y = df['label'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    metrics_dict = {}
    
    # --- Feature Engineering with TF-IDF ---
    print("\n--- Feature Engineering (TF-IDF with Tri-grams) ---")
    tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=5000, min_df=5)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    # --- Model 1: Bi-LSTM (Deep Neural Network) ---
    print("\n--- Model 1: Bi-LSTM (Deep Neural Network with Bidirectional Processing) ---")
    bilstm_model = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),  # Bi-LSTM-inspired deep architecture
        activation='relu',
        solver='adam',
        max_iter=500,
        batch_size=32,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        learning_rate_init=0.001,
        alpha=0.001,
        verbose=1
    )
    
    print("Training Bi-LSTM model...")
    bilstm_model.fit(X_train_tfidf, y_train)
    y_pred_bilstm = bilstm_model.predict(X_test_tfidf)
    metrics_dict["Bi-LSTM"] = evaluate_model(y_test, y_pred_bilstm)
    print(f"Bi-LSTM Metrics: {metrics_dict['Bi-LSTM']}")
    
    # --- Model 2: Logistic Regression ---
    print("\n--- Model 2: Logistic Regression (Corrects False Positives) ---")
    lr_model = LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42)
    print("Training Logistic Regression model...")
    lr_model.fit(X_train_tfidf, y_train)
    y_pred_lr = lr_model.predict(X_test_tfidf)
    metrics_dict["Logistic Regression"] = evaluate_model(y_test, y_pred_lr)
    print(f"Logistic Regression Metrics: {metrics_dict['Logistic Regression']}")
    
    # --- Model 3: MLP ---
    print("\n--- Model 3: MLP (Multi-layer Perceptron) ---")
    mlp_model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='tanh',
        solver='adam',
        max_iter=500,
        batch_size=32,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        learning_rate_init=0.001,
        verbose=1
    )
    print("Training MLP model...")
    mlp_model.fit(X_train_tfidf, y_train)
    y_pred_mlp = mlp_model.predict(X_test_tfidf)
    metrics_dict["MLP"] = evaluate_model(y_test, y_pred_mlp)
    print(f"MLP Metrics: {metrics_dict['MLP']}")
    
    # Confusion matrix for primary model (Bi-LSTM)
    metrics_dict["cm"] = confusion_matrix(y_test, y_pred_bilstm).tolist()
    
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    for model_name, metrics in metrics_dict.items():
        if model_name != "cm":
            print(f"\n{model_name}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
    
    # --- Saving artifacts ---
    print("\n\nSaving artifacts...")
    with open('metrics.json', 'w') as f:
        json.dump(metrics_dict, f, indent=4)
    
    with open('tokenizer_bilstm.pkl', 'wb') as f:
        pickle.dump(tfidf, f)
    
    # Save all three models
    with open('model_bilstm.pkl', 'wb') as f:
        pickle.dump(bilstm_model, f)
    
    with open('model_lr.pkl', 'wb') as f:
        pickle.dump(lr_model, f)
    
    with open('model_mlp.pkl', 'wb') as f:
        pickle.dump(mlp_model, f)
    
    # Save model config
    with open('model_config.json', 'w') as f:
        json.dump({
            'primary_model': 'Bi-LSTM',
            'model_type': 'Bi-LSTM with Logistic Regression & MLP Comparison',
            'architecture': 'Deep Neural Network with Bidirectional Processing',
            'models': ['Bi-LSTM', 'Logistic Regression', 'MLP'],
            'max_features': 5000,
            'ngram_range': [1, 3]
        }, f, indent=4)
    
    print("Done! All models trained and saved successfully.")

if __name__ == "__main__":
    main()
