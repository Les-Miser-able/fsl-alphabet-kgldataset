"""Train a genuine temporal Conv1D-LSTM; legacy image model is untouched."""
import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import numpy as np
from fsl_sequence import LANDMARKS, FEATURES, HAND_ORDER, ROOT, FORMAT, FRAMES, NORMALIZATION
from fsl_categories import category_args, get_category, paths, check_category_metadata
from fsl_sequence_training import split_dataset, build_model, metrics_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    category_args(parser)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-partial", action="store_true", help="Development only: permit a subset of the configured vocabulary")
    parser.add_argument("--check-only", action="store_true", help="Validate data/splits without TensorFlow or training")
    args = parser.parse_args()
    spec = get_category(args, parser)
    args.data = args.data or paths(args.category)["dataset"]
    args.output = args.output or paths(args.category)["models"] / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("Epochs and batch size must be positive.")
    try:
        x = np.load(args.data / "X.npy", mmap_mode="r", allow_pickle=False)
        labels = np.load(args.data / "y.npy", allow_pickle=False)
        manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
        check_category_metadata(manifest, args.category, spec)
        train, val, test = split_dataset(x, labels, manifest, args.seed, args.allow_partial)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    classes = sorted(set(map(str, labels)))
    mapping = {label: i for i, label in enumerate(classes)}
    y = np.array([mapping[str(label)] for label in labels], dtype=np.int32)
    splits = {"train": train, "validation": val, "test": test}
    for name, ids in splits.items():
        print(f"{name}: {len(ids)} samples; classes {dict(Counter(labels[ids]))}")
    print("Held-out data contains only real clips. Contributor independence applies to recorded contributors.")
    print("Kaggle signer identities are unknown; use a cohort not present in that source for held-out recordings.")
    if args.check_only:
        return
    if args.output.exists():
        parser.error("Output already exists; choose a new --output folder.")
    from tensorflow import keras
    keras.utils.set_random_seed(args.seed)
    model = build_model(len(classes))
    model.summary()
    args.output.mkdir(parents=True)

    class Batches(keras.utils.PyDataset):
        def __init__(self, indices, shuffle=False):
            super().__init__()
            self.indices = indices.copy()
            self.shuffle = shuffle
            self.rng = np.random.default_rng(args.seed)
            self.on_epoch_end()

        def __len__(self):
            return (len(self.indices) + args.batch_size - 1) // args.batch_size

        def __getitem__(self, index):
            ids = self.indices[index*args.batch_size:(index+1)*args.batch_size]
            return np.asarray(x[ids], dtype=np.float32), y[ids]

        def on_epoch_end(self):
            if self.shuffle:
                self.rng.shuffle(self.indices)

    history = model.fit(Batches(train, True), validation_data=Batches(val),
                        epochs=args.epochs, callbacks=[
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=.5, patience=5, min_lr=1e-6)])
    prediction = []
    for start in range(0, len(test), args.batch_size):
        ids = test[start:start+args.batch_size]
        prediction.extend(np.argmax(model(np.asarray(x[ids]), training=False).numpy(), axis=1).tolist())
    report = metrics_report(y[test], np.array(prediction), classes)
    model.save(args.output / "model.keras")
    metadata = dict(category=args.category, vocabulary=manifest["vocabulary"], vocabulary_config=spec, format=FORMAT, normalization=NORMALIZATION, frames=FRAMES,
                    classes=classes, hand_order=list(HAND_ORDER), features=["x","y","z","present"], input_shape=[FRAMES, LANDMARKS, FEATURES], seed=args.seed,
                    dataset=str(args.data.resolve()), split_indices={k:v.tolist() for k,v in splits.items()},
                    split_signers={k:sorted({manifest["records"][int(i)]["signer"] for i in ids
                                            if manifest["records"][int(i)]["kind"]=="recording"})
                                   for k,ids in splits.items()})
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (args.output / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output / "history.json").write_text(json.dumps(history.history, indent=2), encoding="utf-8")
    print(f"Test accuracy: {report['accuracy']:.2%}; macro F1: {report['macro_f1']:.4f}")
    print(f"Saved model and reports to {args.output}")


if __name__ == "__main__":
    main()
