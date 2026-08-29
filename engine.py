"""
engine.py
Core logic for Diagnoscan — model loading, inference, and LLM-based explanation.
All non-UI logic lives here so app.py only has to handle presentation.
"""

import os
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from dotenv import load_dotenv
from google import genai

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

load_dotenv()  # reads GEMINI_API_KEY from a .env file in the project root

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise EnvironmentError(
        "GEMINI_API_KEY not found. Create a .env file in the project root with:\n"
        "GEMINI_API_KEY=your_key_here"
    )

client = genai.Client(api_key=GEMINI_API_KEY)
GEMINI_MODEL_NAME = "gemini-2.5-flash-lite"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_CONFIG = {
    "Brain Tumor (MRI)": {
        "path": "models/brain_tumor_model.pt",
        "scan_type": "brain MRI scan",
        "class_display_names": {
            "glioma": "Glioma (a tumor arising from glial/supportive brain tissue)",
            "meningioma": "Meningioma (a tumor arising from the membranes covering the brain)",
            "pituitary": "Pituitary tumor (a tumor of the pituitary gland)",
            "notumor": "No tumor detected",
        },
    },
    "Lung Cancer (CT Scan)": {
        "path": "models/lung_cancer_model.pt",
        "scan_type": "chest CT scan",
        "class_display_names": {
            "adenocarcinoma": "Adenocarcinoma (a type of lung cancer)",
            "large.cell.carcinoma": "Large cell carcinoma (a type of lung cancer)",
            "squamous.cell.carcinoma": "Squamous cell carcinoma (a type of lung cancer)",
            "normal": "No cancer detected",
        },
    },
    "Skin Cancer (Dermatoscopic Image)": {
        "path": "models/skin_cancer_model.pt",
        "scan_type": "dermatoscopic skin lesion image",
        "class_display_names": {
            "akiec": "Actinic keratoses / intraepithelial carcinoma (pre-cancerous/early lesion)",
            "bcc": "Basal cell carcinoma (a type of skin cancer)",
            "bkl": "Benign keratosis (non-cancerous)",
            "df": "Dermatofibroma (non-cancerous)",
            "mel": "Melanoma (a serious type of skin cancer)",
            "nv": "Melanocytic nevus / common mole (non-cancerous)",
            "vasc": "Vascular lesion (non-cancerous)",
        },
    },
}

EVAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Simple in-memory cache so each model is only loaded from disk once per session
_model_cache = {}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(disease_key: str):
    """Loads (and caches) the ResNet50 model + class names for the given disease."""
    if disease_key in _model_cache:
        return _model_cache[disease_key]

    config = MODEL_CONFIG[disease_key]
    checkpoint = torch.load(config["path"], map_location=DEVICE)
    class_names = checkpoint["class_names"]

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    _model_cache[disease_key] = (model, class_names)
    return model, class_names


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict(disease_key: str, image: Image.Image) -> dict:
    """
    Runs inference on a single PIL image for the given disease model.
    Returns a dict with the predicted class, confidence, and full probability breakdown.
    """
    model, class_names = load_model(disease_key)

    image = image.convert("RGB")
    input_tensor = EVAL_TRANSFORM(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    pred_idx = int(torch.argmax(probs).item())
    pred_class = class_names[pred_idx]
    display_names = MODEL_CONFIG[disease_key]["class_display_names"]
    pred_class_display = display_names.get(pred_class, pred_class)
    confidence = float(probs[pred_idx].item()) * 100

    all_probs = {
        class_names[i]: round(float(probs[i].item()) * 100, 2)
        for i in range(len(class_names))
    }

    return {
        "predicted_class": pred_class,
        "predicted_class_display": pred_class_display,
        "confidence": round(confidence, 2),
        "all_probabilities": all_probs,
    }


# ---------------------------------------------------------------------------
# LLM explanation
# ---------------------------------------------------------------------------

def explain_result(disease_key: str, prediction: dict) -> str:
    """
    Sends the model's prediction to Gemini and returns a plain-language,
    non-diagnostic explanation for the user.
    """
    scan_type = MODEL_CONFIG[disease_key]["scan_type"]
    pred_class_display = prediction["predicted_class_display"]
    confidence = prediction["confidence"]

    prompt = f"""You are a medical information assistant embedded in an educational
demo app (not a clinical tool). A machine learning model analyzed a {scan_type}
and predicted the finding "{pred_class_display}" with {confidence}% confidence.

Write a short, plain-language explanation for a general audience that:
1. Clearly states whether this finding indicates a tumor/cancer type was detected,
   or that no tumor/cancer was detected — do not just define an anatomical term,
   directly say what the finding means for the person (e.g. "this suggests the
   model detected patterns consistent with X" or "the model did not detect signs of Y").
2. Briefly explains what "{pred_class_display}" generally is, if it is a tumor/cancer type.
3. Notes that this is an AI model's output from a demo application, not a
   medical diagnosis, and confidence percentage reflects model certainty,
   not clinical certainty.
4. Recommends the person consult a qualified medical professional for any
   real evaluation or next steps.
5. Stays calm and reassuring in tone — avoid alarming language.

Do not state or imply a definitive diagnosis. Keep the response under 150 words.
"""

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=prompt,
        )
        return response.text.strip()
    except Exception as e:
        return (
            "Could not generate an explanation right now "
            f"(error contacting Gemini: {e}). "
            "Please consult a medical professional to interpret this result."
        )


# ---------------------------------------------------------------------------
# Convenience wrapper: run prediction + explanation in one call
# ---------------------------------------------------------------------------

def analyze_image(disease_key: str, image: Image.Image) -> dict:
    """Full pipeline: predict, then explain. This is the single function app.py needs to call."""
    prediction = predict(disease_key, image)
    explanation = explain_result(disease_key, prediction)
    return {
        "prediction": prediction,
        "explanation": explanation,
    }
