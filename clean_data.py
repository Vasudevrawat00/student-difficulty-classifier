"""
clean_data.py
Reads the raw data_proctor.xlsx (one sheet per domain), cleans it up,
and saves a single clean CSV ready for model training.

Run with:  python clean_data.py
"""

import pandas as pd
import re

INPUT_FILE = "data/data_proctor.xlsx"
OUTPUT_FILE = "data/clean_questions.csv"

REQUIRED_COLUMNS = [
    "question_id", "question", "option_a", "option_b", "option_c",
    "option_d", "correct_answer", "domain", "topic", "difficulty_level"
]

# Maps messy label text -> clean band name
BAND_PATTERNS = {
    "Foundational": "Foundational",
    "Easy": "Easy",
    "Intermediate": "Intermediate",
    "Advance": "Advanced",      # catches "Advance" and "Advanced"
    "Expert": "Expert",
}


def clean_band_label(raw_label: str) -> str | None:
    """Turn messy text like 'Expert (L76–L100)' into a clean band name."""
    if not isinstance(raw_label, str):
        return None
    for keyword, clean_name in BAND_PATTERNS.items():
        if keyword.lower() in raw_label.lower():
            return clean_name
    return None  # unrecognized label -> will be dropped


def clean_text(text) -> str:
    """Basic text cleanup: handle non-strings, strip whitespace, fix spacing."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_and_clean():
    xl = pd.ExcelFile(INPUT_FILE)
    all_sheets = []

    for sheet_name in xl.sheet_names:
        df = xl.parse(sheet_name)

        # Keep only rows that have all required columns present (no shifted/corrupt rows)
        df = df.dropna(subset=["question", "difficulty_level", "correct_answer"])

        # Standardize domain name using the sheet name (more reliable than the messy domain column)
        df["domain"] = sheet_name

        # Clean difficulty band labels
        df["difficulty_band"] = df["difficulty_level"].apply(clean_band_label)
        df = df.dropna(subset=["difficulty_band"])

        # Clean question + option text
        for col in ["question", "option_a", "option_b", "option_c", "option_d"]:
            if col in df.columns:
                df[col] = df[col].apply(clean_text)

        # Drop rows where any option ended up empty after cleaning
        df = df[(df["option_a"] != "") & (df["option_b"] != "") &
                 (df["option_c"] != "") & (df["option_d"] != "")]

        all_sheets.append(df)
        print(f"{sheet_name}: kept {len(df)} clean rows")

    merged = pd.concat(all_sheets, ignore_index=True)

    # Remove exact duplicate questions
    before = len(merged)
    merged = merged.drop_duplicates(subset=["question"])
    print(f"\nRemoved {before - len(merged)} duplicate questions")

    return merged


if __name__ == "__main__":
    cleaned = load_and_clean()
    cleaned.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved {len(cleaned)} clean questions to {OUTPUT_FILE}")
    print("\nFinal band distribution:")
    print(cleaned["difficulty_band"].value_counts())
    print("\nFinal domain distribution:")
    print(cleaned["domain"].value_counts())