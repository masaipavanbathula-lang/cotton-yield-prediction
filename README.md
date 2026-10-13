# Cotton Yield Prediction

Research-grounded full-stack application for pre-harvest cotton yield regression using the supplied 80,000-row dataset, IEEE paper, and notebook implementation.

## Architecture

- `backend/`: FastAPI, pandas, scikit-learn pipelines, optional XGBoost/LightGBM, joblib artifacts.
- `frontend/`: React + TypeScript + Vite responsive research dashboard.
- The supplied CSV is intentionally not committed. Copy `cotton_growth_cycle_data.csv` into `backend/data/cotton_growth_cycle_data.csv` before training or running the API; the path is ignored by Git.

## Research/data decisions

The model uses the 12 defensible pre-harvest fields requested in the supplied brief. `Farm_ID`, `Planting_Date`, `Harvest_Date`, `Harvest_Year`, and `Yield` are excluded from inputs to avoid identifier or post-harvest leakage. Imputation, one-hot encoding, scaling, and model fitting are performed inside a pipeline fitted on training data only. R² is reported as a regression metric, never as classification accuracy. The paper's historical percentages are displayed only as context, not as newly verified results. Synthetic augmentation and ROC/confusion visuals are not claimed because the uploaded implementation does not provide a valid production workflow for them.

## Windows setup

```powershell
cd backend
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts\train_model.py
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

In a second PowerShell:

```powershell
cd frontend
npm install
copy .env.example .env
npm run dev
```

Open `http://localhost:5173`. Swagger is at `http://localhost:8000/docs`.

## Tests

```powershell
cd backend
pytest -q
```

Optional model families can be enabled with `pip install xgboost lightgbm`; training will gracefully use the five built-in scikit-learn models if they are unavailable.

## API

`GET /api/health`, `GET /api/dataset/summary`, `GET /api/dataset/columns`, `GET /api/dataset/records`, `GET /api/dataset/distribution`, `POST /api/predict`, `GET /api/models`, `GET /api/models/metrics`, `GET /api/models/feature-importance`, `POST /api/models/train`, `GET /api/reports/evaluation`, and `POST /api/dataset/upload`.
