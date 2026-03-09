"""
================================================
STEP 4 — API REST FASTAPI
Projet : Prédiction d'Acceptation de Prêt
Endpoints :
    GET  /               → accueil
    GET  /health         → statut de l'API
    POST /predict        → prédiction pour 1 client
    POST /predict/batch  → prédiction pour plusieurs clients
================================================
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
from contextlib import asynccontextmanager
from typing import List
import joblib
import os
import sys

# Ajouter src/ au path pour importer predict.py
sys.path.append(os.path.dirname(__file__))
from predict import predict_single, predict_batch

# ─── Chemins ────────────────────────────────
MODEL_PATH  = './models/model.pkl'
SCALER_PATH = './models/scaler.pkl'

# ─── Variables globales ──────────────────────
model  = None
scaler = None


# ══════════════════════════════════════════════
# 1. LIFESPAN — Chargement du modèle
# ══════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle au démarrage et libère à l'arrêt."""
    global model, scaler

    # Démarrage
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"❌ Modèle introuvable : {MODEL_PATH}. Lance d'abord train.py")
    if not os.path.exists(SCALER_PATH):
        raise RuntimeError(f"❌ Scaler introuvable : {SCALER_PATH}. Lance d'abord train.py")

    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print(f"✅ Modèle chargé : {type(model).__name__}")
    print(f"✅ Scaler chargé : StandardScaler")

    yield  # L'API tourne ici

    # Arrêt
    print("🛑 API arrêtée")


# ══════════════════════════════════════════════
# 2. CRÉATION DE L'APPLICATION
# ══════════════════════════════════════════════
app = FastAPI(
    title       = "API Prédiction de Risque de Crédit",
    description = "Prédit l'approbation d'un prêt bancaire via un modèle XGBoost",
    version     = "1.0.0",
    lifespan    = lifespan,
)

# ── CORS — autoriser le frontend GitHub Pages
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ══════════════════════════════════════════════
# 3. SCHÉMA DES DONNÉES — Pydantic V2
# ══════════════════════════════════════════════
class LoanRequest(BaseModel):
    """Données d'un client pour la prédiction."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "no_of_dependents":           2,
                "education":                  0,
                "self_employed":              0,
                "income_annum":               9600000,
                "loan_amount":                29900000,
                "loan_term":                  12,
                "cibil_score":                778,
                "residential_assets_value":   2400000,
                "commercial_assets_value":    17600000,
                "luxury_assets_value":        22700000,
                "bank_asset_value":           8000000,
            }
        }
    )

    no_of_dependents:           int   = Field(..., ge=0,   le=5,   description="Nombre de personnes à charge (0-5)")
    education:                  int   = Field(..., ge=0,   le=1,   description="0 = Graduate, 1 = Not Graduate")
    self_employed:              int   = Field(..., ge=0,   le=1,   description="0 = Non, 1 = Oui")
    income_annum:               float = Field(..., gt=0,           description="Revenu annuel en roupies")
    loan_amount:                float = Field(..., gt=0,           description="Montant du prêt demandé")
    loan_term:                  int   = Field(..., ge=2,   le=20,  description="Durée du prêt en années (2-20)")
    cibil_score:                int   = Field(..., ge=300, le=900, description="Score de crédit CIBIL (300-900)")
    residential_assets_value:   float = Field(...,                 description="Valeur des actifs résidentiels")
    commercial_assets_value:    float = Field(...,                 description="Valeur des actifs commerciaux")
    luxury_assets_value:        float = Field(...,                 description="Valeur des actifs de luxe")
    bank_asset_value:           float = Field(...,                 description="Valeur des actifs bancaires")


class LoanResponse(BaseModel):
    """Réponse de l'API après prédiction."""
    prediction:         str
    probability:        float
    prob_approved:      float
    prob_rejected:      float
    prediction_encoded: int


class BatchRequest(BaseModel):
    """Liste de clients pour prédiction batch."""
    clients: List[LoanRequest]


# ══════════════════════════════════════════════
# 4. ENDPOINTS
# ══════════════════════════════════════════════

# ── GET / ────────────────────────────────────
@app.get("/", tags=["Général"])
def root():
    """Page d'accueil de l'API."""
    return {
        "message":     "🏦 API Prédiction de Risque de Crédit",
        "version":     "1.0.0",
        "description": "Prédit l'approbation d'un prêt bancaire",
        "endpoints": {
            "health":        "GET  /health",
            "predict":       "POST /predict",
            "predict_batch": "POST /predict/batch",
            "docs":          "GET  /docs",
        }
    }


# ── GET /health ───────────────────────────────
@app.get("/health", tags=["Général"])
def health():
    """Vérifie que l'API et le modèle sont opérationnels."""
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="❌ Modèle non chargé")
    return {
        "status":     "ok",
        "model":      type(model).__name__,
        "model_path": MODEL_PATH,
    }


# ── POST /predict ─────────────────────────────
@app.post("/predict", response_model=LoanResponse, tags=["Prédiction"])
def predict(request: LoanRequest):
    """
    Prédit l'approbation du prêt pour un seul client.

    - **prediction** : Approved ou Rejected
    - **probability** : confiance de la prédiction (0-1)
    - **prob_approved** : probabilité d'approbation
    - **prob_rejected** : probabilité de rejet
    """
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="❌ Modèle non chargé")
    try:
        result = predict_single(request.model_dump(), model, scaler)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction : {str(e)}")


# ── POST /predict/batch ───────────────────────
@app.post("/predict/batch", tags=["Prédiction"])
def predict_batch_endpoint(request: BatchRequest):
    """
    Prédit l'approbation pour plusieurs clients en une seule requête.
    """
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="❌ Modèle non chargé")
    try:
        clients_data = [c.model_dump() for c in request.clients]
        results      = predict_batch(clients_data, model, scaler)
        return {
            "total":    len(results),
            "approved": sum(1 for r in results if r['prediction'] == 'Approved'),
            "rejected": sum(1 for r in results if r['prediction'] == 'Rejected'),
            "results":  results,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction : {str(e)}")