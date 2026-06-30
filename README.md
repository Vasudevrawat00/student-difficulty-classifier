# Question Difficulty Classifier & Adaptive Test Engine

A system for a student assessment portal that:
1. Predicts the difficulty level of a new exam question (Foundational / Easy / Intermediate / Advanced / Expert) using a trained machine learning model.
2. Serves an adaptive test via API, where question difficulty adjusts in real time based on whether the student answers correctly.

## How it works

- **`clean_data.py`** — Cleans and merges raw question data (handles inconsistent labels, missing values, duplicates).
- **`train_model.py`** — Converts question text into TF-IDF features and trains a RandomForest classifier to predict difficulty band.
- **`predict.py`** — Command-line tool to test the trained model on any custom question.
- **`label_new_questions.py`** — Bulk-labels a new batch of unlabeled questions using the trained model.
- **`recommend.py`** — Rule-based adaptive testing logic: moves the student up/down a difficulty band based on correct/incorrect answers.
- **`api.py`** — FastAPI server exposing everything above as a real API (`/predict`, `/start_test`, `/submit_answer`).

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Note: the `data/` and `models/` folders are excluded from this repo (private question data / large binary files). To run this project, you'll need your own labeled question dataset in `data/clean_questions.csv` with columns: `question_id, question, option_a, option_b, option_c, option_d, correct_answer, domain, topic, difficulty_band`.

## Running it

```bash
python clean_data.py        # clean raw data (if starting from raw Excel)
python train_model.py       # train and save the model
python predict.py           # test the model interactively
uvicorn api:app --reload    # start the API server
```

Then visit `http://127.0.0.1:8000/docs` for interactive API testing.

## Model performance

Baseline RandomForest + TF-IDF model: ~67.5% accuracy on held-out test data (5-class difficulty band prediction).

## Tech stack

Python, pandas, scikit-learn, FastAPI, TF-IDF text vectorization, RandomForest classifier.
