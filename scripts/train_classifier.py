import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# Configuration
INPUT_FILE = "embeddings.npz"
MODEL_OUTPUT = "minimalism_classifier.pkl"

def train():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Please run extract_features.py first.")
        return

    print(f"Loading data from {INPUT_FILE}...")
    data = np.load(INPUT_FILE)
    X = data['embeddings']
    labels = data['labels']
    
    # Get unique classes
    classes = np.unique(labels)
    class_to_idx = {name: i for i, name in enumerate(classes)}
    y = np.array([class_to_idx[l] for l in labels])
    
    print(f"Data shape: {X.shape}")
    for cls in classes:
        print(f"{cls}: {np.sum(labels == cls)} samples")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"\nTraining Multi-class Logistic Regression ({len(classes)} styles)...")
    clf = LogisticRegression(class_weight='balanced', max_iter=1000)
    clf.fit(X_train, y_train)
    
    # Save the mapping along with the model
    model_data = {
        'classifier': clf,
        'classes': classes
    }
    
    y_pred = clf.predict(X_test)
    print("\nEvaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    
    print(f"Saving model to {MODEL_OUTPUT}")
    joblib.dump(model_data, MODEL_OUTPUT)
    print("Success!")

if __name__ == "__main__":
    train()
