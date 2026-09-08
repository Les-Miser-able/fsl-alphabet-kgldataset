"""Dataset validation, contributor split, and temporal model definition."""
import numpy as np
from fsl_sequence import LANDMARKS, FEATURES, HAND_ORDER, FRAMES, FORMAT, NORMALIZATION, DYNAMIC


def split_dataset(x, labels, manifest, seed=42, allow_partial=False):
    records = manifest.get("records", [])
    if manifest.get("format") != FORMAT or manifest.get("normalization") != NORMALIZATION:
        raise ValueError("Unsupported dataset preprocessing; rebuild with prepare-sequences.py.")
    if x.shape != (len(labels), FRAMES, LANDMARKS, FEATURES) or len(records) != len(labels) or len(labels) == 0:
        raise ValueError("Expected aligned X (N,30,42,4), y (N,), and manifest records.")
    if np.asarray(labels).ndim != 1:
        raise ValueError("Labels must be one-dimensional.")
    for start in range(0, len(x), 256):
        if not np.isfinite(x[start:start+256]).all():
            raise ValueError("Dataset contains non-finite landmarks.")
    classes = sorted(set(map(str, labels)))
    if len(classes) < 2:
        raise ValueError("Collect at least two classes before training a classifier.")
    spec = manifest.get("vocabulary", {"classes":list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
                                      "static_image_classes":list("ABCDEFGHIKLMNOPQRSTUVWXY")})
    allowed = set(spec["classes"])
    static = set(spec.get("static_image_classes", []))
    if not set(classes) <= allowed:
        raise ValueError("Dataset labels do not belong to the selected vocabulary.")
    if not allow_partial and set(classes) != allowed:
        raise ValueError("Training requires all configured classes (A-Z for Alphabet). Collect missing classes, or use --allow-partial for development.")
    hashes = set()
    for label, record in zip(labels, records):
        if record.get("label") != str(label) or record.get("kind") not in {"still", "recording", "external_video"}:
            raise ValueError("Manifest labels or sample types do not match dataset.")
        if record["kind"] == "still" and label not in static:
            raise ValueError("Still images are not approved for this class (including Alphabet J/Z).")
        if record["kind"] == "recording" and not record.get("signer"):
            raise ValueError("Every recording needs a contributor ID.")
        digest = record.get("sample_hash")
        if not digest or digest in hashes:
            raise ValueError("Missing or duplicate sample hashes; rebuild the dataset.")
        hashes.add(digest)
    real = np.array([i for i, r in enumerate(records) if r["kind"] == "recording"], dtype=int)
    still = np.array([i for i, r in enumerate(records) if r["kind"] != "recording"], dtype=int)
    groups = np.array([records[i]["signer"] for i in real])
    signers = np.array(sorted(set(groups)))
    if len(signers) < 3:
        raise ValueError("Need at least 3 independent contributors with real recordings for train/validation/test. More are recommended.")
    wanted = set(classes)
    for label in classes:
        contributors = set(groups[np.asarray(labels)[real] == label])
        if len(contributors) < 3:
            raise ValueError(f"{label} needs real recordings from at least 3 contributors; found {len(contributors)}.")
    n_test = max(1, round(len(signers) * .15))
    n_val = max(1, round(len(signers) * .15))
    rng = np.random.default_rng(seed)
    for _ in range(2000):
        order = rng.permutation(signers)
        test = real[np.isin(groups, order[:n_test])]
        val = real[np.isin(groups, order[n_test:n_test+n_val])]
        train_real = real[np.isin(groups, order[n_test+n_val:])]
        if all(set(np.asarray(labels)[idx]) == wanted for idx in (test, val, train_real)):
            return np.concatenate([train_real, still]), val, test
    raise ValueError("Could not make disjoint contributor splits containing every class. Record each class across more contributors.")


def build_model(num_classes):
    from tensorflow import keras
    from tensorflow.keras import layers
    inputs = keras.Input(shape=(FRAMES, LANDMARKS, FEATURES))
    # Convolve over landmarks within EACH frame; then preserve the time axis.
    x = layers.TimeDistributed(layers.Conv1D(32, 3, padding="same", activation="relu"))(inputs)
    x = layers.TimeDistributed(layers.Conv1D(64, 3, padding="same", activation="relu"))(x)
    x = layers.TimeDistributed(layers.MaxPooling1D(2))(x)
    x = layers.TimeDistributed(layers.Flatten())(x)
    x = layers.Dropout(.3)(x)
    x = layers.LSTM(64)(x)  # This sequence axis is 30 video frames.
    x = layers.Dropout(.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def metrics_report(truth, prediction, classes):
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    np.add.at(cm, (truth, prediction), 1)
    precision = np.divide(np.diag(cm), cm.sum(axis=0), out=np.zeros(len(classes)), where=cm.sum(axis=0)>0)
    recall = np.divide(np.diag(cm), cm.sum(axis=1), out=np.zeros(len(classes)), where=cm.sum(axis=1)>0)
    f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros(len(classes)), where=(precision+recall)>0)
    return dict(accuracy=float(np.trace(cm)/cm.sum()), macro_f1=float(f1.mean()),
                confusion_matrix=cm.tolist(), class_order=list(classes),
                per_class={label: dict(precision=float(precision[i]), recall=float(recall[i]),
                                      f1=float(f1[i]), support=int(cm[i].sum()))
                           for i, label in enumerate(classes)})
