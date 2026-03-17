"""Utility functions for JCommonsenseQA evaluation."""


def process_docs(dataset):
    """Consolidate choice columns into a single list."""

    def _map_doc(doc):
        return {
            "question": doc["question"],
            "choices": [
                doc["choice0"],
                doc["choice1"],
                doc["choice2"],
                doc["choice3"],
                doc["choice4"],
            ],
            "label": doc["label"],
        }

    return dataset.map(_map_doc)
