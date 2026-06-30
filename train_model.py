"""
train_model.py

A simple, beginner-friendly first model:
1. Load the cleaned questions
2. Turn question text into numbers using TF-IDF
3. Add a couple of simple extra features (word count, has code snippet)
4. Train a basic classifier to predict the difficulty band
5. Check how well it did
6. Save the trained model so we can reuse it later

Run with: python train_model.py
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from scipy.sparse import hstack
import joblib

DATA_FILE = "data/clean_questions.csv"
MODEL_FILE = "models/difficulty_model.pkl"
VECTORIZER_FILE = "models/tfidf_vectorizer.pkl"


def add_simple_features(df: pd.DataFrame) -> pd.DataFrame:
    """A couple of easy, intuitive numeric features."""
    df["word_count"] = df["question"].apply(lambda t: len(str(t).split()))
    df["has_code"] = df["question"].str.contains("```", regex=False).astype(int)
    return df


def main():
    print("Loading data...")
    df = pd.read_csv(DATA_FILE)
    df = df.dropna(subset=["question", "difficulty_band"])
    df = add_simple_features(df)

    # X = the inputs the model will learn from
    # y = the answer we want it to predict (the difficulty band)
    y = df["difficulty_band"]

    print("Splitting into train/test sets...")
    # We hold back 20% of the data as a final "test" the model never sees during training.
    # This is how we honestly check if the model actually learned, vs just memorized.
    train_df, test_df, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Converting question text into TF-IDF numbers...")
    # max_features=3000 means: only keep the 3000 most useful words across all questions,
    # to keep things fast and simple for this first version.
    vectorizer = TfidfVectorizer(max_features=3000, stop_words="english")
    X_train_text = vectorizer.fit_transform(train_df["question"])
    X_test_text = vectorizer.transform(test_df["question"])

    # Combine the TF-IDF numbers with our simple extra features (word_count, has_code)
    X_train_extra = train_df[["word_count", "has_code"]].values
    X_test_extra = test_df[["word_count", "has_code"]].values

    X_train = hstack([X_train_text, X_train_extra])
    X_test = hstack([X_test_text, X_test_extra])

    print("Training the model...")
    # RandomForest = builds many simple decision trees and lets them vote.
    # Good beginner-friendly default: hard to mess up, no scaling needed, reasonably accurate.
    model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    print("Evaluating on the test set (data the model never saw during training)...")
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"\nTest Accuracy: {accuracy:.2%}\n")
    print(classification_report(y_test, predictions))

    print("Saving model and vectorizer...")
    import os
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)
    print(f"Saved to {MODEL_FILE} and {VECTORIZER_FILE}")


if __name__ == "__main__":
    main()
