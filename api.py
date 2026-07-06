"""
api.py

A FastAPI server that exposes the trained difficulty model as a web API.

Run with: uvicorn api:app --reload
Then visit http://10.20.0.179:8000/docs to try it interactively.
"""

import joblib
import numpy as np
import pandas as pd
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from scipy.sparse import hstack
from evaluate import calculate_report
from recommend import AdaptiveTest

app = FastAPI(title="Question Difficulty Predictor")

# Allow the HTML page to call the API from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model and vectorizer ONCE when the server starts,
# not on every request -- that would be slow and wasteful.
print("Loading model...")
model = joblib.load("models/difficulty_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
print("Model loaded, ready to serve predictions.")

print("Loading question bank for adaptive testing...")
QUESTIONS_DF = pd.read_csv("data/clean_questions.csv")

# In-memory storage for ongoing test sessions.
# Key = session_id, Value = AdaptiveTest object tracking that student's progress.
# NOTE: this resets if the server restarts -- fine for learning/demo purposes,
# a real production system would use a database instead.
ACTIVE_SESSIONS = {}


# This defines the exact shape of data the API expects to receive.
# FastAPI uses this to validate incoming requests automatically.
class QuestionRequest(BaseModel):
    question: str


# This defines the exact shape of data the API sends back.
class DifficultyResponse(BaseModel):
    question: str
    predicted_band: str


class StartTestRequest(BaseModel):
    domain: str
    start_band: str = "Intermediate"


class QuestionOut(BaseModel):
    question_id: str
    question: str
    option_a: str
    option_b: str
    option_c: str
    option_d: str
    band: str


class StartTestResponse(BaseModel):
    session_id: str
    question: QuestionOut


class SubmitAnswerRequest(BaseModel):
    session_id: str
    answer: str  # "a", "b", "c", or "d"


class SubmitAnswerResponse(BaseModel):
    was_correct: bool
    correct_answer: str
    new_band: str
    next_question: QuestionOut | None  # None means the test ran out of questions


def extract_features(question_text: str):
    text_features = vectorizer.transform([question_text])
    word_count = len(question_text.split())
    has_code = 1 if "```" in question_text else 0
    extra_features = np.array([[word_count, has_code]])
    return hstack([text_features, extra_features])


@app.get("/")
def read_root():
    return {"message": "Question Difficulty Predictor API is running"}


@app.get("/exam")
def serve_exam():
    return FileResponse("exam.html")


@app.post("/predict", response_model=DifficultyResponse)
def predict(request: QuestionRequest):
    features = extract_features(request.question)
    predicted_band = model.predict(features)[0]

    return DifficultyResponse(
        question=request.question,
        predicted_band=predicted_band,
    )


def _question_row_to_out(row, band: str) -> QuestionOut:
    return QuestionOut(
        question_id=row["question_id"],
        question=row["question"],
        option_a=row["option_a"],
        option_b=row["option_b"],
        option_c=row["option_c"],
        option_d=row["option_d"],
        band=band,
    )


@app.post("/start_test", response_model=StartTestResponse)
def start_test(request: StartTestRequest):
    try:
        test = AdaptiveTest(QUESTIONS_DF, domain=request.domain, start_band=request.start_band)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    first_question = test.get_next_question()
    if first_question is None:
        raise HTTPException(status_code=404, detail="No questions available for this domain/band.")

    session_id = str(uuid.uuid4())
    ACTIVE_SESSIONS[session_id] = test

    return StartTestResponse(
        session_id=session_id,
        question=_question_row_to_out(first_question, test.current_band),
    )


@app.get("/get_report/{session_id}")
def get_report(session_id: str):
    test = ACTIVE_SESSIONS.get(session_id)
    if test is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return calculate_report(test)


@app.post("/submit_answer", response_model=SubmitAnswerResponse)
def submit_answer(request: SubmitAnswerRequest):
    test = ACTIVE_SESSIONS.get(request.session_id)
    if test is None:
        raise HTTPException(status_code=404, detail="Session not found. Did you call /start_test first?")

    # We need to know which question was just answered to check correctness.
    # We stored the asked question IDs, but not which one was "current" -- so
    # the client must tell us via session state. Simplify: track last question on the test object.
    if test.last_question is None:
        raise HTTPException(status_code=400, detail="No active question for this session.")

    correct_letter = str(test.last_question["correct_answer"]).strip().lower()
    was_correct = request.answer.strip().lower() == correct_letter

    test.submit_answer(was_correct)
    next_question = test.get_next_question()

    return SubmitAnswerResponse(
        was_correct=was_correct,
        correct_answer=correct_letter,
        new_band=test.current_band,
        next_question=_question_row_to_out(next_question, test.current_band) if next_question is not None else None,
    )