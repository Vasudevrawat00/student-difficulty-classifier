"""
predict.py

Load the trained model and predict the difficulty of any question you type in.

Run with: python predict.py
"""

import joblib
import numpy as np
from scipy.sparse import hstack

MODEL_FILE = "models/difficulty_model.pkl"
VECTORIZER_FILE = "models/tfidf_vectorizer.pkl"


def predict_difficulty(question_text: str, model, vectorizer) -> str:
    # Turn the question into TF-IDF numbers, using the SAME vectorizer
    # that was fit during training (important: must match training exactly)
    text_features = vectorizer.transform([question_text])

    # Recreate the same simple extra features used in training
    word_count = len(question_text.split())
    has_code = 1 if "```" in question_text else 0
    extra_features = np.array([[word_count, has_code]])

    # Combine, just like during training
    final_features = hstack([text_features, extra_features])

    prediction = model.predict(final_features)[0]
    return prediction


def main():
    print("Loading trained model...")
    model = joblib.load(MODEL_FILE)
    vectorizer = joblib.load(VECTORIZER_FILE)
    print("Model loaded. Type a question to see its predicted difficulty.")
    print("(type 'quit' to exit)\n")

    while True:
        question = input("Enter a question: ").strip()
        if question.lower() == "quit":
            break
        if not question:
            continue

        predicted_band = predict_difficulty(question, model, vectorizer)
        print(f"  -> Predicted difficulty: {predicted_band}\n")


if __name__ == "__main__":
    main()
