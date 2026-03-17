"""Generate per-subject YAML configs for JMMLU evaluation.

Usage:
    cd lmms_eval/tasks/jmmlu
    python _generate_configs.py --base_yaml_path _default_template_yaml --save_prefix_path jmmlu
"""

import argparse
import logging
import os

import yaml
from tqdm import tqdm

eval_logger = logging.getLogger("lm-eval")

# JMMLU subjects and their category mappings
# Based on https://github.com/nlp-waseda/JMMLU
SUBJECTS = {
    # STEM (22 subjects)
    "abstract_algebra": "stem",
    "anatomy": "stem",
    "college_biology": "stem",
    "college_chemistry": "stem",
    "college_computer_science": "stem",
    "college_mathematics": "stem",
    "college_physics": "stem",
    "conceptual_physics": "stem",
    "econometrics": "stem",
    "electrical_engineering": "stem",
    "elementary_mathematics": "stem",
    "high_school_biology": "stem",
    "high_school_chemistry": "stem",
    "high_school_computer_science": "stem",
    "high_school_mathematics": "stem",
    "high_school_physics": "stem",
    "high_school_statistics": "stem",
    "human_aging": "stem",
    "machine_learning": "stem",
    "medical_genetics": "stem",
    "nutrition": "stem",
    "virology": "stem",
    # Humanities (11 subjects)
    "formal_logic": "humanities",
    "high_school_european_history": "humanities",
    "japanese_geography": "humanities",
    "japanese_history": "humanities",
    "japanese_idiom": "humanities",
    "logical_fallacies": "humanities",
    "philosophy": "humanities",
    "prehistory": "humanities",
    "world_history": "humanities",
    "world_religions": "humanities",
    "high_school_geography": "humanities",
    # Social Sciences (15 subjects)
    "business_ethics": "social_sciences",
    "clinical_knowledge": "social_sciences",
    "high_school_macroeconomics": "social_sciences",
    "high_school_microeconomics": "social_sciences",
    "high_school_psychology": "social_sciences",
    "human_sexuality": "social_sciences",
    "international_law": "social_sciences",
    "jurisprudence": "social_sciences",
    "management": "social_sciences",
    "moral_disputes": "social_sciences",
    "professional_medicine": "social_sciences",
    "professional_psychology": "social_sciences",
    "public_relations": "social_sciences",
    "security_studies": "social_sciences",
    "sociology": "social_sciences",
    # Other (8 subjects)
    "computer_security": "other",
    "global_facts": "other",
    "japanese_civics": "other",
    "marketing": "other",
    "miscellaneous": "other",
    "professional_accounting": "other",
    "us_foreign_policy": "other",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_yaml_path", required=True)
    parser.add_argument("--save_prefix_path", default="jmmlu")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    base_yaml_name = os.path.split(args.base_yaml_path)[-1]

    # Add constructor for !function tag used by lmms_eval
    yaml.add_constructor(
        "!function",
        lambda loader, node: f"!function {loader.construct_scalar(node)}",
        Loader=yaml.FullLoader,
    )

    with open(args.base_yaml_path, encoding="utf-8") as f:
        base_yaml = yaml.full_load(f)

    ALL_CATEGORIES = []
    for subject, category in tqdm(SUBJECTS.items()):
        if category not in ALL_CATEGORIES:
            ALL_CATEGORIES.append(category)

        description = (
            f"以下は{' '.join(subject.split('_'))}に関する多肢選択問題（回答付き）です。\n\n"
        )

        yaml_dict = {
            "include": base_yaml_name,
            "tag": f"jmmlu_{category}",
            "task": f"jmmlu_{subject}",
            "task_alias": subject.replace("_", " "),
            "dataset_name": subject,
            "description": description,
        }

        file_save_path = args.save_prefix_path + f"_{subject}.yaml"
        eval_logger.info(f"Saving yaml for subset {subject} to {file_save_path}")
        with open(file_save_path, "w", encoding="utf-8") as yaml_file:
            yaml.dump(
                yaml_dict,
                yaml_file,
                allow_unicode=True,
                default_style='"',
            )

    # Generate category group configs
    for category in ALL_CATEGORIES:
        category_subjects = [
            f"jmmlu_{subject}"
            for subject, cat in SUBJECTS.items()
            if cat == category
        ]
        file_save_path = f"jmmlu_{category}.yaml"
        with open(file_save_path, "w", encoding="utf-8") as yaml_file:
            yaml.dump(
                {
                    "group": f"jmmlu_{category}",
                    "task": category_subjects,
                },
                yaml_file,
                indent=4,
                default_flow_style=False,
                allow_unicode=True,
            )

    # Generate top-level group config
    jmmlu_subcategories = [f"jmmlu_{category}" for category in ALL_CATEGORIES]
    file_save_path = args.save_prefix_path + ".yaml"
    eval_logger.info(f"Saving benchmark config to {file_save_path}")
    with open(file_save_path, "w", encoding="utf-8") as yaml_file:
        yaml.dump(
            {
                "group": "jmmlu",
                "task": jmmlu_subcategories,
            },
            yaml_file,
            indent=4,
            default_flow_style=False,
            allow_unicode=True,
        )
