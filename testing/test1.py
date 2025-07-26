import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fewshotlib import FewShotClassifier

MODEL_PATH = "testing/model/fewshot_model.pt"  # Change path if needed

def test_model_loading():
    try:
        classifier = FewShotClassifier(model_path=MODEL_PATH)
        print("✅ Model loaded successfully.")

        print(f"Encoder: {type(classifier.encoder)}")
        print(f"Prototypes shape: {classifier.prototypes.shape}")
        if classifier.labels:
            print(f"Loaded {len(classifier.labels)} labels: {classifier.labels}")
        else:
            print("No labels found in checkpoint.")

    except Exception as e:
        print(f"❌ Model loading failed: {e}")

if __name__ == "__main__":
    test_model_loading()
