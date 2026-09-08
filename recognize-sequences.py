"""Recognize a complete, user-delimited sign with a trained sequence model."""
import argparse
import json
from pathlib import Path
import numpy as np
from fsl_sequence import LANDMARKS, FEATURES, HAND_ORDER, FORMAT, NORMALIZATION, FRAMES, live_clips


from fsl_categories import category_args, get_category, check_category_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    category_args(parser)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=.6)
    args = parser.parse_args()
    spec = get_category(args, parser)
    if not 0 <= args.threshold <= 1:
        parser.error("Threshold must be between 0 and 1.")
    try:
        metadata = json.loads((args.model_dir / "metadata.json").read_text(encoding="utf-8"))
        check_category_metadata(metadata, args.category, spec)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if metadata.get("format") != FORMAT or metadata.get("normalization") != NORMALIZATION:
        parser.error("This is not a compatible sequence model. Train with train-sequences.py.")
    from tensorflow import keras
    model = keras.models.load_model(args.model_dir / "model.keras")
    if tuple(model.input_shape[1:]) != (FRAMES, LANDMARKS, FEATURES) or model.output_shape[-1] != len(metadata["classes"]):
        parser.error("Model dimensions do not match sequence metadata.")
    def predict(sequence, coords, timestamps):
        scores = model(sequence[None], training=False).numpy()[0]
        index = int(np.argmax(scores))
        label = metadata["classes"][index]
        prefix = "" if scores[index] >= args.threshold else "Unsure: "
        return f"{prefix}{label} ({scores[index]:.1%}). SPACE for next sign."
    print(f"Category: {args.category}. SPACE starts/stops a complete sign; C cancels, Q quits.")
    live_clips(predict, args.camera)


if __name__ == "__main__":
    main()
