# Helmet Detection Demo

Simple Flask web demo that runs a YOLOv8 model to detect people wearing helmets. Upload an image and the app highlights helmets and flags people without helmets.

## Project structure

```
helmet-detection-demo/
├── app.py                # Flask app with YOLO inference and auto-cleanup
├── requirements.txt      # Python dependencies
├── models/best.pt        # YOLOv8 weights (place your trained model here)
├── static/
│   ├── uploads/          # Temporary storage for uploaded images
│   └── results/          # Annotated detection outputs
├── templates/index.html  # Single-page UI
└── README.md             # This file
```

## Setup

1. **Create and activate a virtual environment (optional but recommended):**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Add the YOLOv8 weights:**

   Place your `best.pt` file in the `models/` directory (replace the placeholder if necessary).

## Running the app

```bash
# macOS/Linux (bash/zsh)
export FLASK_APP=app.py
flask run

# Windows PowerShell
$env:FLASK_APP = "app.py"
flask run

# Windows Command Prompt
set FLASK_APP=app.py
flask run
```

The app will start on `http://127.0.0.1:5000/`.

Alternatively, run directly with Python to enable the built-in development server (this works the same on every OS):

```bash
python app.py
```

## Usage

1. Open the app in a browser.
2. Upload an image containing people.
3. The YOLOv8 model will detect helmets and display an annotated image. People detected without helmets are marked in red with the label **NO HELMET**.

Uploads and results are automatically cleaned every 10 minutes by a background thread.

## Notes

- The bundled UI mirrors the style of [ElijahAnalysis/indrive](https://github.com/ElijahAnalysis/indrive) for a clean single-page experience.
- Ensure your `best.pt` was trained with class names containing "person" and "helmet" so the annotation logic can differentiate between them.
- If you are running on CPU, inference may take a few seconds depending on the image size.
- When running with PyTorch 2.6 or newer, the app temporarily disables the new `weights_only` safety default while loading trusted YOLO checkpoints so the model initializes without manual steps.
