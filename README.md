# AI-Based Skin Disease Detection and Awareness System
### Final Year Project | CNN + MobileNetV2 + Flask + SQLite
# DermaAI - Skin Disease Detection System

AI based healthcare application for detecting skin diseases using deep learning.

Tech Stack:
- ReactJS
- Flask
- TensorFlow
- SQLite
- Python

Features:
- User Authentication
- Image Upload
- AI Disease Prediction
- Scan History
- Result Management

---

## Project Structure
```
dermai_final/
├── train.py                  ← Run ONCE to train the CNN model
├── app.py                    ← Flask web application
├── skin_model.h5             ← Saved model (created after training)
├── requirements.txt          ← All Python packages needed
├── dermai.db                 ← SQLite database (auto created)
├── dataset/                  ← Put HAM10000 files here
│   ├── HAM10000_metadata.csv
│   └── images/
│       └── *.jpg (10,000 images)
├── templates/
│   ├── login.html
│   ├── signup.html
│   ├── index.html            ← Home / upload page
│   ├── history.html          ← All past scans
│   └── detail.html           ← Full report page
└── static/
    └── uploads/              ← Saved scan images
```

---

## Step 1 — Download Dataset from Kaggle

1. Go to: https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection
2. Click Download and extract the ZIP
3. Create a folder called `dataset` inside dermai_final/
4. Create folder `dataset/images/`
5. Copy:
   - HAM10000_metadata.csv  →  dataset/
   - All .jpg files          →  dataset/images/

---

## Step 2 — Install Python packages

```
pip install -r requirements.txt
```

---

## Step 3 — Train the CNN model (Run ONCE)

```
python train.py
```

- Takes 30-60 minutes on GTX 1650
- Saves model as skin_model.h5 when done

---

## Step 4 — Run the web app

```
python app.py
```

Open browser: http://127.0.0.1:5000

---

## Diseases Detected (HAM10000)

| Disease              | Severity    | Type          |
|----------------------|-------------|---------------|
| Melanocytic Nevi     | Non-Severe  | Benign        |
| Melanoma             | Severe      | Malignant     |
| Benign Keratosis     | Non-Severe  | Benign        |
| Basal Cell Carcinoma | Severe      | Malignant     |
| Actinic Keratosis    | Severe      | Pre-Cancerous |
| Vascular Lesion      | Non-Severe  | Benign        |
| Dermatofibroma       | Non-Severe  | Benign        |

---

## Tech Stack

| Layer      | Technology              |
|------------|------------------------|
| Frontend   | HTML, CSS, JavaScript  |
| Backend    | Python + Flask         |
| AI Model   | CNN + MobileNetV2      |
| Dataset    | HAM10000               |
| Database   | SQLite                 |

---

## Features
- User Login & Signup
- Upload skin image
- CNN predicts disease instantly
- Shows confidence score (%)
- Severity: Severe / Non-Severe
- Recommendations and precautions
- Top 3 predictions with confidence bars
- Scan history saved per user
- Full report detail page
- Delete scans
