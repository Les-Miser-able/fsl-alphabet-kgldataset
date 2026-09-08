"""Category configuration and paths shared by collection, training and inference."""
import json
from pathlib import Path
import re
from fsl_sequence import ROOT

CONFIG = ROOT / "vocabulary.json"
RESERVED = {"CON","PRN","AUX","NUL",*(f"COM{i}" for i in range(1,10)),*(f"LPT{i}" for i in range(1,10))}


def load_categories(path=CONFIG):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or not isinstance(data.get("categories"), dict):
        raise ValueError("Expected vocabulary.json schema_version 1 and categories object.")
    categories = data["categories"]
    if not categories:
        raise ValueError("Define at least one vocabulary category.")
    for key, spec in categories.items():
        if not isinstance(spec, dict):
            raise ValueError(f"{key}: category definition must be an object.")
        if not re.fullmatch(r"[a-z][a-z0-9_]*", key) or key.upper() in RESERVED:
            raise ValueError(f"Invalid category ID: {key}")
        labels = spec.get("classes", [])
        if not isinstance(labels, list) or any(not isinstance(v,str) for v in labels) or len(set(labels)) != len(labels):
            raise ValueError(f"{key}: define unique class labels, or [] to discover class folders.")
        for label in labels:
            if not re.fullmatch(r"[A-Z0-9][A-Z0-9_]*", label) or label in RESERVED:
                raise ValueError(f"{key}: invalid class label {label!r}; use uppercase letters, digits and underscores.")
        static = spec.get("recording_only_classes", [])
        if not isinstance(static, list) or (bool(labels) and not set(static) <= set(labels)):
            raise ValueError(f"{key}: recording_only_classes must be a subset of configured classes.")

    return categories


def category_args(parser):
    parser.add_argument("--category", default="alphabet", help="Vocabulary ID from vocabulary.json")
    parser.add_argument("--config", type=Path, default=CONFIG)


def get_category(args, parser):
    try:
        categories = load_categories(args.config)
        if args.category not in categories:
            raise ValueError(f"Unknown category {args.category!r}. Choose: {', '.join(categories)}")
        return categories[args.category]
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))


def paths(category):
    source = ROOT/"data"/"categories"/category
    return dict(source=source, dataset=ROOT/"extracted_data"/"categories"/category,
                models=ROOT/"models"/"categories"/category)


def valid_label(label):
    return bool(re.fullmatch(r"[A-Z0-9][A-Z0-9_]*", label)) and label not in RESERVED


def resolved_vocabulary(spec, source):
    result = dict(spec)
    if not result["classes"]:
        result["classes"] = sorted(p.name for p in source.iterdir()
                                    if p.is_dir() and valid_label(p.name)) if source.exists() else []
    result["static_image_classes"] = sorted(set(result["classes"]) -
                                            set(spec.get("recording_only_classes", [])))
    return result


def check_category_metadata(metadata, category, spec):
    if metadata.get("category") != category:
        raise ValueError(f"Category mismatch: selected {category}, artifact belongs to {metadata.get('category')!r}.")
    if metadata.get("vocabulary_config", metadata.get("vocabulary")) != spec:
        raise ValueError("Vocabulary configuration changed. Rebuild the dataset/retrain or use its original config.")
