# Smart Agro AI System

A minimal farmer-facing crop recommendation project with Vercel-ready deployment.

## ✅ What this repo contains
- `index.html` — simple static UI for inputting soil/weather values
- `api/predict.py` — FastAPI serverless endpoint for crop prediction
- `train.py` — train the model from scratch and save artifacts
- `data/` — dataset files used for training
- `artifacts/` — trained model and label encoder
- `vercel.json` — Vercel serverless deployment config

## 🚀 Local setup
1. Install dependencies:
```bash
python -m pip install -r requirements.txt
```
2. Train the model:
```bash
python train.py
```
3. Run locally for development:
```bash
uvicorn api.predict:app --reload --host localhost --port 8000
```
4. Open `index.html` in a browser or use Vercel flow.

## 📦 Vercel deployment
1. Install Vercel CLI if not already installed:
```bash
npm install -g vercel
```
2. Deploy from the repo root:
```bash
vercel --prod
```

## 🧠 Training flow
- Loads `data/Crop_recommendation.csv`
- Encodes crop labels
- Trains a Random Forest classifier
- Saves:
  - `artifacts/model.joblib`
  - `artifacts/label_encoder.joblib`

## 📁 Project structure

```
Smart Agro AI System/
│
├── api/                          # Vercel serverless prediction endpoint
│   └── predict.py
├── artifacts/                    # Saved model artifacts
│   ├── label_encoder.joblib
│   └── model.joblib
├── data/                         # Training data
│   ├── Crop_recommendation.csv
│   └── Crop_Desc.csv
├── index.html                    # Static presentation UI
├── notebooks/                    # Optional analysis notebooks
│   └── models.ipynb
├── train.py                      # Train ML model from scratch
├── requirements.txt              # Python dependencies
├── vercel.json                   # Vercel deployment config
└── README.md                     # Project documentation
```
