import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fewshotlib import FewShotClassifier

MODEL_PATH = "testing/model/fewshot_model.pt"  # Change if different
DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")

def test_prediction():
    try:
        classifier = FewShotClassifier(model_path=MODEL_PATH)

        # List image files
        image_files = [os.path.join(DATASET_DIR, f) 
                       for f in os.listdir(DATASET_DIR) 
                       if f.lower().endswith((".png", ".jpg", ".jpeg"))]

        if not image_files:
            print("❌ No images found in dataset folder.")
            return

        results = classifier.predict(image_files)

        print("✅ Predictions:")
        for res in results:
            print(f"  → File: {os.path.basename(res['file'])}, "
                  f"Predicted Index: {res['index']}, Label: {res['label']}")

    except Exception as e:
        print(f"❌ Prediction test failed: {e}")

if __name__ == "__main__":
    test_prediction()
