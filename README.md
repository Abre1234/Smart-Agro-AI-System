# Smart Agro AI — Crop Recommendation System

AI-powered crop recommendation from soil and weather inputs. Deploy the live demo by connecting this GitHub repo to [Vercel](https://vercel.com).

## Live demo (deploy via GitHub)

1. **Push this repo to GitHub** (include the `artifacts/` folder — required for predictions).
2. Go to [vercel.com/new](https://vercel.com/new) and **Import** your GitHub repository.
3. Use these settings (Vercel usually auto-detects them):
   - **Framework Preset:** Other (or FastAPI)
   - **Root Directory:** `.` (repo root)
   - **Build Command:** leave empty
   - **Output Directory:** leave empty
4. Click **Deploy**. Your live URL will look like: `https://your-project-name.vercel.app`
5. Open the URL — the UI is at `/` and the API at `/api/predict`.

> **Important:** Commit `artifacts/model.joblib`, `artifacts/scaler.joblib`, `artifacts/label_encoder.joblib`, and `artifacts/feature_names.joblib` before deploying. Without them, predictions will fail.

## Project structure

```
Crop_Recommendation_System/
├── app.py                 # FastAPI app (Vercel entrypoint)
├── public/index.html      # Web UI (served at /)
├── artifacts/             # Trained model files (commit these)
├── train.py               # Train model from data/
├── data/                  # Training CSV files
├── requirements.txt       # Production dependencies
└── README.md
```

## Local development

```bash
python -m pip install -r requirements.txt
python train.py
python -m uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) for the UI and [http://127.0.0.1:8000/api/predict](http://127.0.0.1:8000/api/predict) for the API.

### Sample API request

```bash
curl -X POST http://127.0.0.1:8000/api/predict \
  -H "Content-Type: application/json" \
  -d "{\"N\":90,\"P\":42,\"K\":43,\"temperature\":20.8,\"humidity\":82,\"ph\":6.5,\"rainfall\":202.9}"
```

## Retrain the model

```bash
python train.py
```

This writes updated files to `artifacts/`. Commit and push to redeploy on Vercel.

## Tech stack

- **ML:** XGBoost, scikit-learn, engineered NPK/weather features
- **API:** FastAPI
- **Hosting:** Vercel (connected to GitHub)

## License

See [LICENSE](LICENSE).
