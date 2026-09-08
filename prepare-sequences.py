"""Prepare one category from class folders containing images, videos and captured clips."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
from fsl_sequence import ROOT, FRAMES, LANDMARKS, FEATURES, FORMAT, NORMALIZATION, HAND_ORDER, detector, detect_frame, prepare_clip, prepare_still, read_video
from fsl_categories import category_args, get_category, paths, resolved_vocabulary

IMAGES = {".jpg",".jpeg",".png"}
VIDEOS = {".mp4",".avi",".mov",".mkv"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    category_args(parser)
    parser.add_argument("--source", type=Path, help="Category root containing CLASS folders")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-images", action="store_true")
    parser.add_argument("--video-signer", help="Known contributor for ALL imported videos in this run; omit for online supplements")
    args = parser.parse_args()
    config = get_category(args, parser)
    args.source = args.source or paths(args.category)["source"]
    args.output = args.output or paths(args.category)["dataset"]
    spec = resolved_vocabulary(config,args.source)
    if args.output.exists():
        parser.error("Output already exists. Choose a new --output folder to rebuild.")
    if not args.source.is_dir() or not spec["classes"]:
        parser.error("No class folders found. Add CLASS folders or capture labeled clips first.")
    records, seen, rejected = [], {}, []
    with tempfile.TemporaryDirectory(prefix="fsl-category-") as temp:
        def add(sequence,path,label,signer,kind):
            digest=hashlib.sha256(np.asarray(sequence,dtype="<f4").tobytes()).hexdigest()
            if digest in seen:
                if seen[digest]!=label:
                    raise RuntimeError(f"Identical landmarks have conflicting labels: {seen[digest]}, {label}: {path}")
                rejected.append(dict(source=str(path),reason="duplicate landmark sample"))
                return
            seen[digest]=label
            np.save(Path(temp)/f"{len(records)}.npy",sequence)
            records.append(dict(source=str(path.resolve()),label=label,signer=signer,kind=kind,sample_hash=digest))

        landmarker=None
        try:
            for folder in sorted(args.source.iterdir()):
                if not folder.is_dir():
                    continue
                label=folder.name
                if label not in spec["classes"]:
                    print(f"Skipping unconfigured class folder: {folder.name}")
                    continue
                before=len(records)
                for path in sorted(folder.rglob("*")):
                    ext=path.suffix.lower()
                    if not path.is_file() or ext not in IMAGES|VIDEOS|{".npz"}:
                        continue
                    try:
                        if ext in IMAGES:
                            if args.no_images or label not in spec["static_image_classes"]:
                                rejected.append(dict(source=str(path),reason="image disabled or recording-only class"))
                                continue
                            import cv2
                            if landmarker is None:
                                landmarker=detector()
                            frame=cv2.imread(str(path))
                            if frame is None:
                                raise ValueError("Unreadable image")
                            coords=detect_frame(frame,landmarker)
                            add(prepare_still(coords),path,label,None,"still")
                        elif ext in VIDEOS:
                            kind="recording" if args.video_signer else "external_video"
                            add(read_video(path),path,label,args.video_signer,kind)
                        else:
                            with np.load(path,allow_pickle=False) as raw:
                                saved_category=str(raw["category"]) if "category" in raw else None
                                legacy_alphabet=saved_category is None and args.category=="alphabet"
                                if str(raw["format"])!=FORMAT or str(raw["label"])!=label or (saved_category!=args.category and not legacy_alphabet):
                                    raise ValueError("Clip category, label or format mismatch")
                                signer=str(raw["signer"])
                                if not signer:
                                    raise ValueError("Captured clip needs a contributor ID")
                                sequence=prepare_clip(raw["coords"],raw["timestamps"])
                            add(sequence,path,label,signer,"recording")
                    except (OSError,ValueError,KeyError) as error:
                        rejected.append(dict(source=str(path),reason=str(error)))
                print(f"{label}: {len(records)-before} samples kept")
        finally:
            if landmarker is not None:
                landmarker.close()
        if not records:
            for item in rejected[:5]:
                print(f"Rejected {item['source']}: {item['reason']}")
            parser.error("No usable samples. Check source files and rejection messages.")
        args.output.mkdir(parents=True)
        x=np.lib.format.open_memmap(args.output/"X.npy",mode="w+",dtype=np.float32,
                                   shape=(len(records),FRAMES,LANDMARKS,FEATURES))
        for i in range(len(records)):
            x[i]=np.load(Path(temp)/f"{i}.npy")
        x.flush()
        del x
        np.save(args.output/"y.npy",np.array([r["label"] for r in records]))
        manifest=dict(category=args.category,vocabulary=spec,vocabulary_config=config,format=FORMAT,
                      normalization=NORMALIZATION,frames=FRAMES,hand_order=list(HAND_ORDER),
                      features=["x","y","z","present"],records=records,rejected=rejected)
        (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(f"Saved {len(records)} samples to {args.output}; rejected {len(rejected)}.")
    print("Counts:",dict(Counter(r["label"] for r in records)))
    print("Known-signer recordings:",dict(Counter(r["label"] for r in records if r["kind"]=="recording")))
    missing=sorted(set(spec["classes"])-{r["label"] for r in records})
    if missing:
        print("Missing classes:",", ".join(missing))
    print("Still images and unknown-signer videos are training supplements only.")


if __name__=="__main__":
    main()
