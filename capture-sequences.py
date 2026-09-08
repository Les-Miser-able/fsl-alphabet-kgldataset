"""Record one labeled gesture per SPACE-delimited clip; saves landmarks only."""
import argparse
from pathlib import Path
import re
import uuid
import numpy as np
from fsl_sequence import ROOT, FORMAT, live_clips
from fsl_categories import category_args, get_category, paths, RESERVED, valid_label


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    category_args(parser)
    parser.add_argument("--label", required=True, type=str.upper)
    parser.add_argument("--signer", required=True, help="Stable anonymous contributor ID, e.g. signer01")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    spec = get_category(args, parser)
    if not valid_label(args.label) or (spec["classes"] and args.label not in spec["classes"]):
        parser.error(f"{args.label} is not a class in {args.category}. Choose: {spec['classes']}")
    if args.signer.upper() in RESERVED:
        parser.error("Use a different signer ID; this name is reserved by Windows.")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.signer):
        parser.error("Signer ID must contain only letters, numbers, underscore or hyphen.")
    folder = (args.output or paths(args.category)["source"]) / args.label

    def save(sequence, coords, timestamps):
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"capture_{args.signer}_{uuid.uuid4().hex}.npz"
        np.savez_compressed(path, coords=coords.astype(np.float32),
                            timestamps=timestamps-timestamps[0],
                            label=args.label, signer=args.signer, category=args.category, format=FORMAT)
        return f"Saved {args.label}: {path.name}. SPACE for another."

    print(f"Recording {args.category}/{args.label} for {args.signer}. Use one or both hands as required by the sign.")
    print("SPACE: begin, perform the COMPLETE sign, SPACE: finish/save. Q: quit.")
    print("For static signs, hold the sign for the whole clip.")
    live_clips(save, args.camera)


if __name__ == "__main__":
    main()
