# 🩺 Diagnoscan

### AI-Powered Multi-Disease Detection, Explained in Plain English

> Upload a scan. Get a prediction. Understand what it actually means — no medical degree required.

Diagnoscan is an end-to-end deep learning system that analyzes medical images across **three different disease domains** — brain tumors, lung cancer, and skin cancer — and pairs every prediction with a clear, LLM-generated explanation so the output isn't just a number, it's something a human can actually understand.

This isn't a single model wearing three hats. It's three independently trained computer vision pipelines, unified under one architecture, talking to a language model that translates machine confidence into human language.
check out the demo here  https://youtu.be/GGOZN8LQyVs

---

## ⚡ What It Actually Does

| Module | Input | Detects |
|---|---|---|
| 🧠 **Brain Tumor** | MRI Scan | Glioma, Meningioma, Pituitary Tumor, or No Tumor |
| 🫁 **Lung Cancer** | CT Scan | Adenocarcinoma, Large Cell Carcinoma, Squamous Cell Carcinoma, or Normal |
| 🔬 **Skin Cancer** | Dermatoscopic Image | 7 lesion classes, including Melanoma and Basal Cell Carcinoma |

For every prediction, Diagnoscan doesn't just spit out a label and a percentage. It sends that result to **Gemini**, which generates a calm, plain-language explanation of what the finding means — while explicitly and consistently reinforcing that this is an educational tool, not a diagnosis.

---

## 🏗️ Architecture

```
                    ┌─────────────────────┐
   Upload Image ──▶ │   Streamlit UI       │
                    │   (app.py)           │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   engine.py          │
                    │   ─────────────────  │
                    │   1. Load ResNet50   │
                    │   2. Run Inference    │
                    │   3. Get Confidence   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Gemini API          │
                    │   Plain-language      │
                    │   explanation         │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Result + Explain    │
                    │   rendered in UI       │
                    └─────────────────────┘
```

**Clean separation of concerns:** `app.py` never touches PyTorch or the Gemini API directly — it just calls one function, `analyze_image()`. All the model logic, inference, and LLM orchestration lives in `engine.py`. This means the UI and the intelligence layer can evolve independently.

---

## 🧠 The Machine Learning

Every model in Diagnoscan uses **transfer learning** on a pretrained **ResNet50** backbone (trained on 14M+ ImageNet images), fine-tuned specifically for its medical imaging domain:

1. **Freeze the pretrained backbone**, replace the final classification layer, and train just that layer on the target dataset.
2. **Unfreeze the entire network** and fine-tune end-to-end at a 10x lower learning rate — letting early layers gently adapt their learned visual features (edges, textures, shapes) to the specific patterns found in medical imagery.
3. **Evaluate on a held-out test set** the model has never seen, using precision, recall, F1-score, and confusion matrices — not just a single accuracy number.

The skin cancer model goes a step further, applying **class-weighted loss** to counteract severe class imbalance in the HAM10000 dataset (benign moles vastly outnumber rarer conditions like dermatofibroma).

### Datasets Used

- 🧠 [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- 🫁 [Chest CT-Scan Images Dataset](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images)
- 🔬 [HAM10000 Skin Cancer Dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

---

## 🚀 Getting Started

### 1. Clone and install dependencies
```bash
git clone <your-repo-url>
cd diagnoscan
pip install -r requirements.txt
```

### 2. Get the trained models

The three trained `.pt` model files are **not included in this repository**. GitHub has a 100MB per-file hard limit and generally isn't well suited to hosting model checkpoints, so they're excluded via `.gitignore`.

**If the `models/` folder is empty when you clone this repo, run the training notebooks first:**

```
training_notebooks/
├── brain_tumor_training.ipynb   → produces brain_tumor_model.pt
├── lung_cancer_training.ipynb   → produces lung_cancer_model.pt
└── skin_cancer_training.ipynb   → produces skin_cancer_model.pt
```

Open each one in Google Colab, switch the runtime to a T4 GPU (Runtime → Change runtime type), and run all cells top to bottom. Each notebook downloads its dataset via the Kaggle API, trains, evaluates, and saves the resulting `.pt` file to your Google Drive — training takes roughly 10-40 minutes per model depending on dataset size. Download the three resulting files from your Drive into a local `models/` folder:

```
models/
├── brain_tumor_model.pt
├── lung_cancer_model.pt
└── skin_cancer_model.pt
```

If you already have these three files from a previous run (or received them separately), just place them here directly — no need to retrain. **The app will not start without all three files present.**

### 3. Set up your Gemini API key
Create a `.env` file in the project root:
```
GEMINI_API_KEY=your_actual_key_here
```

### 4. Run it
```bash
streamlit run app.py
```

That's it — the app opens in your browser, ready to accept uploads.

---

## 📁 Project Structure

```
diagnoscan/
├── app.py                        # Streamlit UI layer
├── engine.py                     # Model loading, inference, LLM explanation logic
├── requirements.txt
├── .env                          # Your Gemini API key (never commit this)
├── models/                       # Empty in this repo — see "Get the trained models" below
│   ├── brain_tumor_model.pt      #   (generated by running the training notebooks)
│   ├── lung_cancer_model.pt
│   └── skin_cancer_model.pt
└── training_notebooks/
    ├── brain_tumor_training.ipynb
    ├── lung_cancer_training.ipynb
    └── skin_cancer_training.ipynb
```

---

## ⚠️ Responsible Use

**Diagnoscan is an educational and portfolio demonstration project. It is not a certified medical device and must never be used for real diagnostic purposes.**

- Model predictions are based on academic benchmark datasets, not clinical-grade, institutionally diverse data.
- Confidence scores reflect model certainty, not clinical certainty.
- The Gemini explanation layer is explicitly prompted to avoid diagnostic language and to always recommend professional medical consultation.
- Real health concerns should always be directed to a qualified healthcare provider.

This project exists to demonstrate applied machine learning engineering — transfer learning, multi-model system design, and LLM orchestration — not to replace medical professionals.

---

## 🛠️ Built With

- **PyTorch & Torchvision** — model training and inference
- **ResNet50** — pretrained CNN backbone
- **Streamlit** — interactive web UI
- **Google Gemini API** — natural language explanation generation
- **scikit-learn** — evaluation metrics
- **Google Colab (T4 GPU)** — training environment

---

## 📌 Notes for Reviewers

If you're looking at this as a portfolio piece: the interesting engineering here isn't any single model — it's the **pipeline discipline**. Three separate training notebooks producing standardized, self-describing checkpoint files (weights + class names bundled together), loaded through a single unified inference engine, feeding a carefully constrained LLM prompt that had to be explicitly engineered *not* to sound like a diagnosis. That constraint — getting an LLM to be genuinely useful without overstepping into something irresponsible — was the harder design problem than the CNNs themselves.

---

<div align="center">

**Developed by Umer Farooq**

</div>
