"""One entry point for category-based FSL data collection and model development."""
import argparse
from collections import Counter
from pathlib import Path
import runpy
import sys
from fsl_categories import CONFIG, load_categories, paths, resolved_vocabulary
from fsl_sequence import ROOT

COMMANDS = {"capture":"capture-sequences.py", "prepare":"prepare-sequences.py",
            "train":"train-sequences.py", "recognize":"recognize-sequences.py"}


def main():
    if len(sys.argv) > 1 and sys.argv[1] in COMMANDS:
        command = sys.argv[1]
        sys.argv = [str(ROOT / COMMANDS[command]), *sys.argv[2:]]
        runpy.run_path(sys.argv[0], run_name="__main__")
        return
    parser = argparse.ArgumentParser(description=__doc__,
        epilog="Use python fsl.py capture|prepare|train|recognize --help for their options.")
    parser.add_argument("command", choices=["categories","init","status"])
    parser.add_argument("--category", help="Limit to one category")
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    try:
        categories = load_categories(args.config)
        if args.category:
            categories = {args.category: categories[args.category]}
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    for category, spec in categories.items():
        p = paths(category)
        resolved = resolved_vocabulary(spec, p["source"])
        print(f"\n{spec.get('display_name',category)} [{category}]: {', '.join(resolved['classes']) or '(add class folders or capture a new label)'}")
        if args.command == "init":
            p["source"].mkdir(parents=True, exist_ok=True)
            for label in spec["classes"]:
                (p["source"]/label).mkdir(exist_ok=True)
            print(f"Source folder ready: {p['source']}")
        elif args.command == "status":
            counts = Counter()
            if p["source"].exists():
                for file in p["source"].glob("*/*"):
                    if file.suffix.lower() in {".npz",".mp4",".avi",".mov",".mkv",".jpg",".jpeg",".png"}:
                        counts[file.parent.name] += 1
            print("Source files by label (not quality-validated):", dict(counts))
            print("Prepared dataset:", p["dataset"] if (p["dataset"]/"manifest.json").exists() else "not prepared")
            runs = sorted(p["models"].glob("*/metadata.json")) if p["models"].exists() else []
            print("Saved model runs:", [str(file.parent) for file in runs])
            print("Next: capture recordings, then prepare, then train --check-only.")


if __name__ == "__main__":
    main()
