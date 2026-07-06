"""
evaluate.py

Handles MCQ scoring and generates a performance report
after a student completes their adaptive test.

This connects to the AdaptiveTest session which already
tracks every question asked and every answer submitted.
"""


# Performance label thresholds (based on % score)
PERFORMANCE_LABELS = [
    (90, "Outstanding"),
    (75, "Proficient"),
    (60, "Developing"),
    (40, "Beginner"),
    (0,  "Needs Improvement"),
]


def get_performance_label(score_percent: float) -> str:
    """Turn a percentage score into a human-readable label."""
    for threshold, label in PERFORMANCE_LABELS:
        if score_percent >= threshold:
            return label
    return "Needs Improvement"


def calculate_report(session) -> dict:
    """
    Takes a completed AdaptiveTest session and returns
    a full performance report.

    The session must have tracked answers -- we add that
    tracking to AdaptiveTest in recommend.py.
    """
    answers = session.answer_log  # list of dicts, one per question answered

    if not answers:
        return {"error": "No answers recorded in this session."}

    total = len(answers)
    correct = sum(1 for a in answers if a["was_correct"])
    score_percent = round((correct / total) * 100, 1)

    # Breakdown by difficulty band
    band_breakdown = {}
    for a in answers:
        band = a["band"]
        if band not in band_breakdown:
            band_breakdown[band] = {"correct": 0, "total": 0}
        band_breakdown[band]["total"] += 1
        if a["was_correct"]:
            band_breakdown[band]["correct"] += 1

    # Add percentage to each band
    for band in band_breakdown:
        b = band_breakdown[band]
        b["score_percent"] = round((b["correct"] / b["total"]) * 100, 1)

    # Breakdown by topic
    topic_breakdown = {}
    for a in answers:
        topic = a.get("topic", "Unknown")
        if topic not in topic_breakdown:
            topic_breakdown[topic] = {"correct": 0, "total": 0}
        topic_breakdown[topic]["total"] += 1
        if a["was_correct"]:
            topic_breakdown[topic]["correct"] += 1

    for topic in topic_breakdown:
        t = topic_breakdown[topic]
        t["score_percent"] = round((t["correct"] / t["total"]) * 100, 1)

    # Find strongest and weakest topics
    if topic_breakdown:
        strongest = max(topic_breakdown, key=lambda t: topic_breakdown[t]["score_percent"])
        weakest = min(topic_breakdown, key=lambda t: topic_breakdown[t]["score_percent"])
    else:
        strongest = weakest = "N/A"

    return {
        "total_questions": total,
        "correct_answers": correct,
        "score_percent": score_percent,
        "performance_label": get_performance_label(score_percent),
        "final_difficulty_band": session.current_band,
        "band_breakdown": band_breakdown,
        "topic_breakdown": topic_breakdown,
        "strongest_topic": strongest,
        "weakest_topic": weakest,
    }
