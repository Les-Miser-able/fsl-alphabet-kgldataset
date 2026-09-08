# Category-based FSL collection and training

Project: C:\Users\Russel\Desktop\fsl-extract

Use `python fsl.py` as the entry point. Each category owns its source data, prepared dataset,
model runs, label order and evaluation. The shared Hand Landmarker pipeline supports one
or two hands and 30-frame sequences.

## Folder layout

```text
data/categories/
  alphabet/
    A/                 images, videos and captured .npz clips together
    ...
    J/                 your two existing J clips were moved here
    Z/
  numbers/             empty: add your number class folders
  family/              empty: add your family class folders
  wh_words/            add WHO, WHAT, WHERE, WHEN, WHY, HOW

# Example after you add/capture MOTHER:
data/categories/family/MOTHER/
  online_image.jpg
  online_video.mp4
  capture_signer01_<unique-id>.npz

extracted_data/categories/family/
  X.npy
  y.npy
  manifest.json

models/categories/family/<run>/
  model.keras
  metadata.json
  evaluation.json
  history.json
```

Alphabet images were moved from data/Collated into data/categories/alphabet.
The old image extractor's source path was updated. Original trained models and prepared
image arrays were preserved. The two existing v2 J recordings retain their signer metadata.
The former capture directories may remain empty.

## Category and class definitions

`vocabulary.json` controls categories. Alphabet defines A-Z. WH Words defines WHO, WHAT,
WHERE, WHEN, WHY, HOW. Numbers and Family have `"classes": []`, which means their classes
are discovered from the folders you create. Capturing a new class also creates its folder.

Use uppercase class folder names, digits and underscores, e.g. MOTHER, GRAND_FATHER, 10.
Use the SAME anonymous signer ID for a person across every class, category and session.
Do not create a different ID for each recording.

The `recording_only_classes` list disables still-image supplements for signs requiring
movement. Alphabet starts with J and Z on that list. Add other dynamic signs there after
validating the vocabulary with your FSL teacher. Still images cannot demonstrate a complete
moving sign. Imported images are otherwise repeated across 30 frames as training supplements.
Real clips are required for credible validation and testing of all classes.

## Environment

Open a terminal in the project and use a Python environment with the required packages:
```powershell
cd C:\Users\Russel\Desktop\fsl-extract
python -m pip install -r requirements-sequences.txt
python fsl.py categories
python fsl.py status
```

The existing project environment was previously found to use Python 3.14 and could not be
executed from the agent sandbox. Tests used a bundled Python with NumPy. Camera/MediaPipe
execution and actual TensorFlow training have not been verified in this session.
Use an environment supported by the installed MediaPipe and TensorFlow releases.

The four category folders already exist. `python fsl.py init` can recreate missing category
and configured class folders without deleting anything.

## Capture your own examples

```powershell
python fsl.py capture --category family --label MOTHER --signer signer01
python fsl.py capture --category numbers --label 1 --signer signer01
python fsl.py capture --category alphabet --label J --signer signer01
python fsl.py capture --category wh_words --label WHO --signer signer01
```

SPACE starts recording; perform the complete gesture; SPACE finishes and saves.
For a static sign, hold it throughout the clip. C cancels. Q quits.
Keep all hands needed for the sign visible. Use --camera 1 for a different device.
The camera preview is unmirrored, with colored Left/Right landmark labels.

MOTHER captures save directly in data/categories/family/MOTHER/ alongside your imported files.
Raw video is not saved; the .npz stores coordinates, timestamps, label, signer and category.
Record multiple repetitions per class and multiple independent contributors. Every class
needs at least three known contributors for the training/validation/test split; this is only
a technical minimum, not a recommended final dataset size.

## Add online files manually

Place .jpg/.jpeg/.png images or .mp4/.avi/.mov/.mkv videos directly in the appropriate class
folder. Each video must be trimmed to ONE complete gesture, at most 10 seconds, and have
valid frame-rate metadata. Do not place videos of multiple words in a single class.

Videos without signer information are treated as TRAINING ONLY supplements, as are images.
If ALL imported videos in a preparation run are from one known person, --video-signer ID
can assign that contributor. Do not use one ID for videos from multiple or unknown people.
Captured .npz files use their saved signer IDs independently.

Previously captured two-hand v2 alphabet files lacking category metadata are accepted only
under Alphabet. Single-hand v1 clips cannot be recovered as two-hand data; re-extract the
original images/videos or record again.

## Prepare one category

```powershell
python fsl.py prepare --category family
python fsl.py prepare --category alphabet
```

The source is data/categories/CATEGORY/CLASS/. Both your captures and imported media are
processed together. The output is extracted_data/categories/CATEGORY/.

A rebuild never overwrites an existing dataset:
```powershell
python fsl.py prepare --category family --output extracted_data/categories/family_v2
```

Use --source PATH for another category root containing CLASS folders.
Use --no-images to build from recordings/videos only.
Inspect manifest.json for rejected samples, source types, classes and contributor IDs.
Exact duplicate normalized samples are removed; conflicting duplicate labels stop preparation.

Input X shape is (N,30,42,4): 30 frames, 21 landmarks per hand, and x/y/z/presence.
One shared XY reference preserves hand travel and relative hand positions; Z remains local
to each wrist. Absent hands are zero-filled with presence=0. Brief internal gaps are
interpolated per hand; long gaps and hands entering/leaving view remain marked absent.
The detector can still mislabel hands or lose them under occlusion: review recording quality.

## Check and train separately

```powershell
python fsl.py train --category family --check-only
python fsl.py train --category family --output models/categories/family/run01
```

For a rebuilt dataset:
```powershell
python fsl.py train --category family --data extracted_data/categories/family_v2 --output models/categories/family/run02
```

Without --output, a dated run folder is created under models/categories/CATEGORY.
Each category gets its own model weights, class order and reports.

Known contributors are separated into approximately 70/15/15 percent groups; sample
proportions can differ. Every class needs real known-signer recordings in all three sets.
Imported stills and unknown-signer videos go only to training. Check source balance so a
large online dataset does not overwhelm your recordings. Unknown online signer identities
mean independence across those sources cannot be proven.

By default all classes in the prepared vocabulary must be present. --allow-partial permits
a development subset, but still requires at least two classes and contributor-independent
recordings per class. Add data and PREPARE AGAIN after adding class folders; a prepared
dataset is a snapshot, not a live view of the source folder.

Training uses TimeDistributed Conv1D frame features followed by an LSTM over 30 frames,
up to 60 epochs, early stopping and learning-rate reduction.
Review evaluation.json for accuracy, macro F1, per-class metrics and the confusion matrix.

## Run the selected category

```powershell
python fsl.py recognize --category family --model-dir models/categories/family/run01
```

SPACE starts/stops one sign, then the model predicts within that category.
Category mismatches or changed vocabulary configuration produce an error before inference.
Switch category and model directory together. This is user-delimited recognition, not
automatic continuous sentence segmentation. Softmax scores are not calibrated guarantees;
there is no automatically trained unknown/no-sign class.

## PWA integration boundary

Each model's metadata.json provides its category, output label order, input shape, feature
order, hand order and preprocessing version. Use that mapping with its matching model.
The category selection in the PWA should select the corresponding model and labels.

This project currently saves Keras models. TensorFlow.js conversion and a browser
implementation of the exact sequence preprocessing still need to be added and tested;
the .keras file cannot be loaded directly in a PWA. Existing unrelated model artifacts
are not automatically converted or replaced.

## Verification

```powershell
python -B -m unittest test_sequences test_categories -v
```

Tests use synthetic landmarks to check two-hand slots and masks, trajectory preservation,
dataset preparation, category isolation, capture metadata and contributor splits.
These tests do not establish recognition accuracy. Real recordings and a TensorFlow run
are required before claiming a trained model.
