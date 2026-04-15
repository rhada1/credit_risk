#  Credit Risk Prediction

A REST API for bank loan approval prediction based on an XGBoost Machine Learning model.

---

##  Description

This project predicts whether a loan application will be **Approved** or **Rejected** based on the applicant's financial profile (income, CIBIL score, loan amount, assets, etc.).

---

##  Project Structure

```
credit-risk-prediction/
├── src/
│   ├── download_data.py            # Download dataset via Kaggle API
│   ├── exploration.py              # Data analysis and visualization
│   ├── preprocess.py               # Data cleaning and encoding
│   ├── train.py                    # ML model training
│   ├── predict.py                  # Prediction script
│   └── main.py                     # FastAPI application
├── models/
│   ├── model.pkl                   # Trained XGBoost model
│   └── scaler.pkl                  # StandardScaler
├
├── .github/
│   └── workflows/
│       └── deploy.yml              # CI/CD GitHub Actions
├── Dockerfile
├── requirements.txt
├── .gitignore
├── .dockerignore
└── README.md
```

---

##  Model Results

| Model | Accuracy | AUC-ROC |
|---|---|---|
| Logistic Regression | 94.03% | 0.9749 |
| Random Forest | 96.72% | 0.9971 |
| **XGBoost**  | **97.54%** | **0.9979** |

---

##  Installation & Setup

### Prerequisites
- Python 3.13
- Docker Desktop
- Kaggle account (to download the dataset)

### 1. Clone the repository
```bash
git clone https://github.com/your_username/credit-risk-prediction.git
cd credit-risk-prediction
```

### 2. Create virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
Create a `.env` file at the root of the project:
```
KAGGLE_API_TOKEN=your_kaggle_token
```

### 5. Run the full pipeline
```bash
python src/download_data.py   # Download dataset
python src/exploration.py     # Explore data
python src/preprocess.py      # Preprocess data
python src/train.py           # Train the model
python src/predict.py         # Test predictions
```

### 6. Run the API
```bash
uvicorn src.main:app --reload
```

API available at `http://localhost:8000`

---

##  Docker

### Build the image
```bash
docker build -t credit-risk-api .
```

### Run the container
```bash
docker run -p 8000:8000 credit-risk-api
```

---

##  API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Home page |
| GET | `/health` | API health check |
| POST | `/predict` | Predict for a single client |
| POST | `/predict/batch` | Predict for multiple clients |
| GET | `/docs` | Swagger UI documentation |

### Sample request `/predict`

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "no_of_dependents": 2,
       "education": 0,
       "self_employed": 0,
       "income_annum": 9600000,
       "loan_amount": 29900000,
       "loan_term": 12,
       "cibil_score": 778,
       "residential_assets_value": 2400000,
       "commercial_assets_value": 17600000,
       "luxury_assets_value": 22700000,
       "bank_asset_value": 8000000
     }'
```

### Sample response

```json
{
  "prediction": "Approved",
  "probability": 0.9823,
  "prob_approved": 0.9823,
  "prob_rejected": 0.0177,
  "prediction_encoded": 0
}
```

---

##  Dataset

**Loan Approval Prediction** — Kaggle
- 4,269 applicants
- 12 features (income, CIBIL score, assets, education, etc.)
- Target variable: `loan_status` (Approved / Rejected)
- Class distribution: 62% Approved / 38% Rejected

---

##  Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.13 | Main language |
| FastAPI | REST API |
| XGBoost | ML model |
| Scikit-learn | Preprocessing |
| Docker | Containerization |
| GitHub Actions | CI/CD |
| Render | Cloud deployment |

---

##  Author

HARRAG Rhada

GitHub: @rhada1 (https://github.com/rhada1)
LinkedIn: Rhada Harrag (https://www.linkedin.com/in/rhada-harrag-991634160/)
