# Hate-Speech-Detection-Web-App
The application features a real-time predictions along with comprehensive model performance metrics.

## Key Features

- **Binary Classification**: Predicts "Hate Speech / Abusive" vs "Not Hate Speech".
- **Bias Correction**: Augmented the dataset with ~300 neutral samples (e.g., "I like her") to fix False Positives.
- **Multiple Model Comparison**: Evaluates Logistic Regression, Naive Bayes, Random Forest, and Deep Learning (MLP).
- **Interactive Metrics**: Visualizes Accuracy, Precision, Recall, F1-Score, and Confusion Matrices using Plotly.
- **Modern UI**: A charcoal-dark frontend built with Streamlit, optimized for readability and visual appeal.

## Technical Implementation

### Data & Preprocessing
- **Dataset**: Kaggle [https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset]
- **Augmentation**: Injected ~300 neutral and positive samples into the training data to counteract's the dataset's inherent bias against certain phrases.
- **Cleaning**: Context-aware cleaning (preserving words while removing URLs, mentions, and symbols).
- **Modeling**: 
    - **Logistic Regression (Primary)**: Switched to LR as the default model due to its superior stability and accuracy on neutral phrases like "I like her".
    - **MLP Classifier**: A deep-learning representation using (128, 64, 32) hidden layers for complex pattern recognition.

### Performance Results
The models performed exceptionally well on the full dataset (24k+ rows):

| Model | Accuracy | F1-Score |
| :--- | :--- | :--- |
| **Random Forest** | 94.67% | 96.81% |
| **Logistic Regression (Default)**| 93.85% | 96.38% |
| **Deep Learning (MLP)** | 93.65% | 96.19% |
| **Naive Bayes** | 88.92% | 93.72% |

## Project Structure

- [train.py]: The core script that implements data augmentation, cleans text, and trains all models.
- [app.py]: The Streamlit frontend using `model_lr.pkl` for stable, real-time predictions.
- [metrics.json]: Stored evaluation results for the full dataset.

## How to Run

1. **Activate the environment**: `source venv/bin/activate`
2. **Run Training** (if needed): `./venv/bin/python3 train.py`
3. **Launch the App**: `./venv/bin/streamlit run app.py`

> [!TIP]
> The app is currently running at **http://localhost:8501**. You can enter any text to see it in action!

---
Developed as a complete solution for robust abusive language detection.
