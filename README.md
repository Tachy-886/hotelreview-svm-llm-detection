# Hotel Review Authenticity Analysis Pipeline

This project implements an **end-to-end hotel review analysis system** that combines:

- A **Linear SVM classifier** trained on the **Ott Deceptive Opinion Spam Dataset**
- A **vision-language model (VLM)** for extracting review text from screenshots
- A **large language model (LLM)** for robust multilingual → English translation

The system is designed to analyze **real hotel review screenshots**, detect **potentially deceptive reviews**, and support downstream decision-making.

------

## 1. Model Overview

### SVM Classifier

- **Model**: Linear Support Vector Machine (LinearSVC)
- **Training Data**:
	**Ott et al. Deceptive Opinion Spam Dataset (Hotel Reviews)**
- **Task**: Binary classification
	- `truthful`
	- `deceptive`
- **Features**: TF-IDF (1–2 grams)
- **Output**:
	- Predicted label (`truthful` / `deceptive`)
	- Decision score (distance to the separating hyperplane)
		- Larger absolute value → higher confidence

The SVM model is trained offline and saved as a reusable `joblib` pipeline.

------

## 2. System Pipeline

The full pipeline follows a **modular, production-style design**:

```
Review Screenshot (PNG/JPG)
        ↓
Vision-Language Model (qwen3-vl-plus)
        ↓
Extracted Review Text (Original Language)
        ↓
LLM Translation Module (English)
        ↓
SVM Authenticity Prediction
        ↓
(Optional) LLM-Based Aggregation & Decision Analysis
```

Each module is isolated and reusable.

------

## 3. Repository Structure

```
project/
├── graph/                     # Put review screenshots here (PNG/JPG)
├── src/
│   ├── llm_image_comment_recognition.py   # Image → review text
│   ├── llm_translate.py                   # Text → English translation
│   ├── model_training.py                  # Train SVM on Ott dataset
│   ├── model_inference.py                 # SVM inference utilities
│   ├── llm_analysis.py                    # High-level LLM decision analysis
│   └── __init__.py
├── model/
│   └── SVM.joblib               # Trained SVM model
├── main.ipynb                   # Main entry notebook
└── README.md
```

------

## 4. Environment Setup

### Python Version

**Recommended**:

```text
Python 3.9 – 3.11
```

------

### Required Libraries

Install the following dependencies to avoid version conflicts:

```bash
pip install \
  numpy==1.24.4 \
  pandas==2.0.3 \
  scikit-learn==1.3.2 \
  joblib==1.3.2 \
  dashscope>=1.14.0
```

> ⚠️ **Important Notes**
>
> - `scikit-learn >= 1.3` is recommended for stable `LinearSVC` behavior.
> - Do **not** use very old NumPy versions (<1.21).
> - `dashscope` is required for all LLM and VLM calls.

------

## 5. How to Use (Quick Start)

### Step 1: Prepare Review Images

Place all hotel review screenshots into the **`graph/`** folder.

Supported formats:

- `.png`
- `.jpg`
- `.jpeg`
- `.webp`

Each image may contain **one or multiple reviews**.

------

### Step 2: Run the Main Pipeline

Open and run:

```text
main.ipynb
```

The notebook will:

1. Extract review text from images
2. Translate reviews into English
3. Apply the trained SVM model
4. Output authenticity predictions and confidence scores

No additional configuration is required.

------

## 6. Training the SVM Model (Optional)

If you want to retrain the model:

```python
from src.model_training import train_and_save_model

train_and_save_model(
    csv_path="data/deceptive-opinion.csv",
    model_save_path="model/SVM.joblib"
)
```

- Training uses a **stratified train/test split**
- Final model is retrained on the full dataset before saving

------

## 7. Notes on Interpretation

- The SVM **decision score is not a probability**
- It represents the **distance to the decision boundary**
	- Positive score → more likely deceptive
	- Negative score → more likely truthful
- Larger magnitude → higher confidence

This score is intentionally used as a **risk signal**, not a calibrated probability.

------

## 8. Intended Use

This project is intended for:

- Academic research
- Coursework and demonstrations
- Experimental decision-support systems

It is **not** a production-grade fraud detection system.

------

## 9. Citation

If you use this project or the trained model, please cite:

> Ott, M., Choi, Y., Cardie, C., & Hancock, J. T.
> *Finding Deceptive Opinion Spam by Any Stretch of the Imagination.*
> ACL 2011.

------

## 10. Summary

- ✔ Trained on a **well-known deceptive review dataset**
- ✔ Clean modular design
- ✔ Vision + Language + Classical ML integration
- ✔ Minimal setup: **drop images → run main