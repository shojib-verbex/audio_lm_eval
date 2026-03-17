"""Utility functions for JMMLU (Japanese MMLU) evaluation.

The parquet version of nlp-waseda/JMMLU has columns:
  question, A, B, C, D, answer (letter)
We normalize these to a consistent format.
"""

ANSWER_TO_IDX = {"A": 0, "B": 1, "C": 2, "D": 3}


def process_docs(dataset):
    """Consolidate choice columns and convert letter answer to integer index."""

    def _map_doc(doc):
        return {
            "question": doc["question"],
            "choices": [doc["A"], doc["B"], doc["C"], doc["D"]],
            "answer_idx": ANSWER_TO_IDX.get(doc["answer"].strip(), 0),
        }

    return dataset.map(_map_doc)
