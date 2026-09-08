# HUDYAT sample PWA

An installable, static web application using the existing `fsl_model.keras` alphabet classifier. All camera processing and inference run on the device.

## Run locally

Use Node 22.13 or newer. Run `npm ci`, then `npm run dev`. Open the printed localhost URL and select Start camera. A camera requires HTTPS or localhost.

Run `npm run build` for the production app in `dist/client`. Serve that directory over HTTPS (or localhost). The manifest and service worker provide installation and offline caching after the assets finish downloading. Development mode does not register the service worker.

## Model

`public/models/alphabet/model.json` and `weights.bin` contain the converted trained weights. `metadata.json` stores the exact label order, mean and standard deviation from the original model. The input is one frame containing 21 landmarks with x/y/z coordinates, normalized per axis. Camera frames are mirrored before extraction, matching the original Python camera script.

This model is not the new 30-frame, two-hand category model. Only Alphabet is enabled. J and Z outputs classify photographed poses and do not establish movement recognition. Scores are not calibrated correctness guarantees. The other categories remain visibly unavailable.

The model's layer weights were transferred directly from the Keras archive into matching TensorFlow.js layers. Twelve inputs were checked against an independent NumPy forward pass over the saved Keras weights: maximum probability difference 5.96e-7; save/reload difference zero. This is numerical conversion validation, not a camera accuracy evaluation or a Keras-runtime comparison.

## Assets and dependencies

MediaPipe Hand Landmarker and its WebAssembly files are served from `public/mediapipe`; TensorFlow.js and MediaPipe JavaScript are bundled locally. No camera frames are uploaded. The full offline asset download is approximately 45 MB. Offline availability depends on successful caching and browser storage retention.

## Validation limits

Production compilation, TypeScript checks, model conversion parity and asset references are checked. Live webcam recognition, browser installation/offline behavior and the optional WebMCP category tool still need device-level testing. Camera preview was not automated because no browser testing was requested.

## Integration

Use the same model, metadata and preprocessing together. Future two-hand sequence models need a separate adapter for their (30,42,4) inputs; replacing these weights alone is insufficient. The category UI stops the camera whenever the category changes.
