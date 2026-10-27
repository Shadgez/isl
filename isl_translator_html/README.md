# ISL Translator — custom HTML frontend

This version replaces the Gradio interface with a custom HTML/CSS/JavaScript frontend while retaining the supplied TensorFlow model, class labels, and MediaPipe hand landmarker in a Python/Flask backend.

## Files required

Keep these files in the project root alongside `app.py`:
- `isl_nn_model.h5`
- `isl_classes.pickle`
- `hand_landmarker.task`

`isl_data.pickle` is included for inspection/training work but is not required to run the web app.

## Run locally (Windows)

Use a supported Python version for the installed TensorFlow and MediaPipe versions. In PowerShell, from this folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:7860`.

## Run locally (Linux/Fedora)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:7860`.

## Notes

- Browser webcam access generally requires `localhost` or HTTPS.
- This is a first-pass frontend conversion. It preserves the feature extraction logic in the supplied `app.py`; it does not guarantee higher model accuracy.
- The browser sends camera frames to the Python backend for inference. Do not expose the app publicly without considering privacy, access controls, and resource limits.
- `isl_data.pickle` was inspected and contains 2,519 feature rows with 84 values per row and letter labels. The original app's feature extraction produces 84 values; matching the training preprocessing is essential for accuracy.
