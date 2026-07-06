"""
recommend.py

A simple ADAPTIVE TEST SIMULATOR.

No ML here -- this is rule-based logic:
  - Start the student at a middle difficulty band.
  - After each answer: correct -> move up one band, wrong -> move down one band.
  - Pick the next unseen question from that band (optionally filtered by domain).
  - A fixed number of Aptitude questions are mixed into every test, at random
    positions, while still following the same adaptive difficulty logic.

Run with: python recommend.py
"""

import pandas as pd
import random

DATA_FILE = "data/clean_questions.csv"

# The bands in order, easiest to hardest. This order matters --
# it's how we know what "one band up" or "one band down" means.
BAND_ORDER = ["Foundational", "Easy", "Intermediate", "Advanced", "Expert"]

APTITUDE_DOMAIN_NAME = "Aptitude"


class AdaptiveTest:
    def __init__(
        self,
        questions_df: pd.DataFrame,
        domain: str = None,
        start_band: str = "Intermediate",
        total_questions: int = 10,
        num_aptitude_questions: int = None,
    ):
        # Keep the full, unfiltered question bank around so we can pull
        # Aptitude questions separately from the main domain pool.
        self.full_df = questions_df

        self.df = questions_df
        if domain:
            # Case-insensitive match, so typing "aiml" still matches "AIML"
            self.df = self.df[self.df["domain"].str.lower() == domain.lower()]
            if len(self.df) == 0:
                raise ValueError(
                    f"No questions found for domain '{domain}'. "
                    f"Check the spelling matches one of the available domains exactly."
                )

        # Separate pool of Aptitude questions, used for the mixed-in questions
        # regardless of which main domain the test is for. If the test's own
        # domain IS Aptitude, there's nothing to "mix in" separately.
        if domain and domain.lower() == APTITUDE_DOMAIN_NAME.lower():
            self.aptitude_df = None
        else:
            self.aptitude_df = self.full_df[
                self.full_df["domain"].str.lower() == APTITUDE_DOMAIN_NAME.lower()
            ]
            if len(self.aptitude_df) == 0:
                self.aptitude_df = None  # no aptitude questions available, skip mixing

        self.current_band_index = BAND_ORDER.index(start_band)
        self.asked_question_ids = set()
        self.last_question = None
        self.last_question_was_aptitude = False
        self.answer_log = []  # tracks every question + answer for scoring

        self.total_questions = total_questions
        self.question_counter = 0  # how many questions have been served so far

        # Decide which question numbers (1-indexed) will be Aptitude questions.
        # e.g. total_questions=10, num_aptitude_questions=3 -> 3 random positions
        # out of 1..10 will be served from the Aptitude pool instead of the domain pool.
        if self.aptitude_df is not None:
            if num_aptitude_questions is None:
                num_aptitude_questions = random.choice([2, 3])
            num_aptitude_questions = min(num_aptitude_questions, total_questions)
            self.aptitude_positions = set(
                random.sample(range(1, total_questions + 1), num_aptitude_questions)
            )
        else:
            self.aptitude_positions = set()

    @property
    def current_band(self) -> str:
        return BAND_ORDER[self.current_band_index]

    def move_up(self):
        # Don't go above Expert (index 4, the last one)
        self.current_band_index = min(self.current_band_index + 1, len(BAND_ORDER) - 1)

    def move_down(self):
        # Don't go below Foundational (index 0, the first one)
        self.current_band_index = max(self.current_band_index - 1, 0)

    def _pick_from(self, pool: pd.DataFrame):
        """Pick a random unseen question from the given pool at the current band."""
        candidates = pool[
            (pool["difficulty_band"] == self.current_band) &
            (~pool["question_id"].isin(self.asked_question_ids))
        ]
        if len(candidates) == 0:
            return None
        chosen = candidates.sample(n=1).iloc[0]
        self.asked_question_ids.add(chosen["question_id"])
        return chosen

    def get_next_question(self):
        """
        Pick the next question. Usually from the test's own domain, but on
        pre-selected positions, pulls from the Aptitude pool instead --
        still respecting the current adaptive difficulty band.
        """
        self.question_counter += 1
        use_aptitude = (
            self.aptitude_df is not None
            and self.question_counter in self.aptitude_positions
        )

        if use_aptitude:
            chosen = self._pick_from(self.aptitude_df)
            if chosen is None:
                # No aptitude question available at this band -- fall back
                # to the domain pool so the test isn't interrupted.
                chosen = self._pick_from(self.df)
                use_aptitude = False
        else:
            chosen = self._pick_from(self.df)

        if chosen is None:
            return None  # ran out of questions at this band entirely

        self.last_question = chosen
        self.last_question_was_aptitude = use_aptitude
        return chosen

    def submit_answer(self, was_correct: bool):
        """Call this after the student answers, to adjust difficulty for next time."""
        if self.last_question is not None:
            self.answer_log.append({
                "question_id": self.last_question["question_id"],
                "band": self.current_band,
                "topic": self.last_question.get("topic", "Unknown"),
                "domain": APTITUDE_DOMAIN_NAME if self.last_question_was_aptitude else self.last_question.get("domain", "Unknown"),
                "was_correct": was_correct,
            })
        if was_correct:
            self.move_up()
        else:
            self.move_down()


def run_simulation():
    print("Loading question bank...")
    df = pd.read_csv(DATA_FILE)

    domain = input(f"Enter a domain to test ({', '.join(df['domain'].unique())}): ").strip()
    num_questions = int(input("How many questions to simulate? ") or "5")

    test = AdaptiveTest(df, domain=domain, start_band="Intermediate", total_questions=num_questions)

    print("\nStarting adaptive test simulation.\n")
    for i in range(num_questions):
        question = test.get_next_question()
        if question is None:
            print(f"No more questions available at {test.current_band} band. Stopping early.")
            break

        tag = " [APTITUDE]" if test.last_question_was_aptitude else ""
        print(f"--- Question {i + 1} (Band: {test.current_band}){tag} ---")
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
    run_simulation()uvicorn api:app --host 127.0.0.1 --port 8010 --reload