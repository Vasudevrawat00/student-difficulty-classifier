"""
recommend.py

A simple ADAPTIVE TEST SIMULATOR.

No ML here -- this is rule-based logic:
  - Start the student at a middle difficulty band.
  - After each answer: correct -> move up one band, wrong -> move down one band.
  - Pick the next unseen question from that band (optionally filtered by domain).

Run with: python recommend.py
"""

import pandas as pd
import random

DATA_FILE = "data/clean_questions.csv"

# The bands in order, easiest to hardest. This order matters --
# it's how we know what "one band up" or "one band down" means.
BAND_ORDER = ["Foundational", "Easy", "Intermediate", "Advanced", "Expert"]


class AdaptiveTest:
    def __init__(self, questions_df: pd.DataFrame, domain: str = None, start_band: str = "Intermediate"):
        self.df = questions_df
        if domain:
            # Case-insensitive match, so typing "aiml" still matches "AIML"
            self.df = self.df[self.df["domain"].str.lower() == domain.lower()]
            if len(self.df) == 0:
                raise ValueError(
                    f"No questions found for domain '{domain}'. "
                    f"Check the spelling matches one of the available domains exactly."
                )

        self.current_band_index = BAND_ORDER.index(start_band)
        self.asked_question_ids = set()  # so we never repeat a question
        self.last_question = None  # the most recently served question (used to check answers)

    @property
    def current_band(self) -> str:
        return BAND_ORDER[self.current_band_index]

    def move_up(self):
        # Don't go above Expert (index 4, the last one)
        self.current_band_index = min(self.current_band_index + 1, len(BAND_ORDER) - 1)

    def move_down(self):
        # Don't go below Foundational (index 0, the first one)
        self.current_band_index = max(self.current_band_index - 1, 0)

    def get_next_question(self):
        """Pick a random unseen question from the current difficulty band."""
        candidates = self.df[
            (self.df["difficulty_band"] == self.current_band) &
            (~self.df["question_id"].isin(self.asked_question_ids))
        ]

        if len(candidates) == 0:
            return None  # ran out of questions at this band

        chosen = candidates.sample(n=1).iloc[0]
        self.asked_question_ids.add(chosen["question_id"])
        self.last_question = chosen
        return chosen

    def submit_answer(self, was_correct: bool):
        """Call this after the student answers, to adjust difficulty for next time."""
        if was_correct:
            self.move_up()
        else:
            self.move_down()


def run_simulation():
    print("Loading question bank...")
    df = pd.read_csv(DATA_FILE)

    domain = input(f"Enter a domain to test ({', '.join(df['domain'].unique())}): ").strip()
    num_questions = int(input("How many questions to simulate? ") or "5")

    test = AdaptiveTest(df, domain=domain, start_band="Intermediate")

    print("\nStarting adaptive test simulation.\n")
    for i in range(num_questions):
        question = test.get_next_question()
        if question is None:
            print(f"No more questions available at {test.current_band} band. Stopping early.")
            break

        print(f"--- Question {i + 1} (Band: {test.current_band}) ---")
        print(question["question"])
        print(f"a) {question['option_a']}")
        print(f"b) {question['option_b']}")
        print(f"c) {question['option_c']}")
        print(f"d) {question['option_d']}")

        answer = input("Your answer (a/b/c/d), or 'q' to quit: ").strip().lower()
        if answer == "q":
            break

        correct_letter = str(question["correct_answer"]).strip().lower()
        was_correct = (answer == correct_letter)

        if was_correct:
            print("Correct! Moving to a harder band next.\n")
        else:
            print(f"Incorrect. The correct answer was '{correct_letter}'. Moving to an easier band next.\n")

        test.submit_answer(was_correct)

    print(f"Simulation ended. Final difficulty band reached: {test.current_band}")


if __name__ == "__main__":
    run_simulation()
