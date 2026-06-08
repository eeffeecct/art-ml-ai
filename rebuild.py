import subprocess
import sys

def rebuild():
    print("=== STARTING DATABASE REBUILD ===")
    
    print("\nStep 1: Extracting features from all folders...")
    try:
        subprocess.run([sys.executable, "extract_features.py"], check=True)
    except subprocess.CalledProcessError:
        print("Error during feature extraction. Rebuild aborted.")
        return

    print("\nStep 2: Training the multi-class classifier...")
    try:
        subprocess.run([sys.executable, "train_classifier.py"], check=True)
    except subprocess.CalledProcessError:
        print("Error during classifier training. Rebuild aborted.")
        return

    print("\n=== REBUILD SUCCESSFUL! ===")
    print("New styles have been added to the database.")
    print("Please restart the app (python app.py) to apply changes.")

if __name__ == "__main__":
    rebuild()
