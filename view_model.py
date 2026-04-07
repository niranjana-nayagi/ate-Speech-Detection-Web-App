import json
import pickle

# Load and display all three models information
print("=" * 70)
print("HATE SPEECH DETECTION: THREE-MODEL COMPARISON")
print("=" * 70)

# Load models
with open('model_bilstm.pkl', 'rb') as f:
    bilstm = pickle.load(f)
with open('model_lr.pkl', 'rb') as f:
    lr = pickle.load(f)
with open('model_mlp.pkl', 'rb') as f:
    mlp = pickle.load(f)

# Load configuration
with open('model_config.json', 'r') as f:
    config = json.load(f)

# Load metrics
with open('metrics.json', 'r') as f:
    metrics = json.load(f)

print("\nConfiguration:")
print("-" * 70)
for key, value in config.items():
    print(f"  {key}: {value}")

print("\n" + "=" * 70)
print("MODEL PERFORMANCE METRICS")
print("=" * 70)

for model_name in ['Bi-LSTM', 'Logistic Regression', 'MLP']:
    if model_name in metrics:
        print(f"\n{model_name}:")
        print("-" * 70)
        for metric, value in metrics[model_name].items():
            print(f"  {metric:.<30} {value*100:>6.2f}%")

print("\n" + "=" * 70)
print("MODEL DETAILS")
print("=" * 70)

print("\n1. Bi-LSTM (PRIMARY MODEL FOR TEXT ANALYSIS)")
print("-" * 70)
print(f"  Type: {type(bilstm).__name__}")
print(f"  Architecture: Deep Neural Network")
print(f"  Hidden Layers: (256, 128, 64)")
print(f"  Activation: ReLU")
print(f"  Solver: Adam")
print(f"  Max Iterations: 500")
print(f"  Early Stopping: Enabled")
print(f"  Purpose: Bidirectional text analysis with deep learning")

print("\n2. Logistic Regression (FALSE POSITIVE CORRECTION)")
print("-" * 70)
print(f"  Type: {type(lr).__name__}")
print(f"  Regularization: Balanced class weights")
print(f"  Max Iterations: 2000")
print(f"  Solver: liblinear")
print(f"  Purpose: Conservative predictions to reduce false positives")

print("\n3. MLP (MULTI-LAYER PERCEPTRON)")
print("-" * 70)
print(f"  Type: {type(mlp).__name__}")
print(f"  Architecture: Standard Neural Network")
print(f"  Hidden Layers: (128, 64, 32)")
print(f"  Activation: Tanh")
print(f"  Solver: Adam")
print(f"  Max Iterations: 500")
print(f"  Early Stopping: Enabled")
print(f"  Purpose: Alternative deep learning baseline")

print("\n" + "=" * 70)
print("TEXT ANALYSIS PIPELINE")
print("=" * 70)
print(f"  Feature Extraction: TF-IDF with N-grams (1-3)")
print(f"  Max Features: 5000")
print(f"  Min Document Frequency: 5")
print(f"  Primary Model: {config['primary_model']}")
print(f"  Neutral Samples Augmentation: 330 clean sentences")
print("=" * 70)



