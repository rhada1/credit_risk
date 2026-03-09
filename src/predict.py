"""
================================================
STEP 3 — SCRIPT DE PRÉDICTION
Projet : Prédiction d'Acceptation de Prêt
Charge model.pkl + scaler.pkl et prédit sur
de nouvelles données jamais vues.
================================================
"""

import pandas as pd
import numpy as np
import joblib
import os

# ─── Chemins ────────────────────────────────
MODEL_PATH  = './models/model.pkl'
SCALER_PATH = './models/scaler.pkl'


# ══════════════════════════════════════════════
# 1. CHARGEMENT DU MODÈLE ET SCALER
# ══════════════════════════════════════════════
def load_model():
    """Charge le modèle et le scaler sauvegardés."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f" Modèle introuvable : {MODEL_PATH}\n   Lance d'abord train.py")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f" Scaler introuvable : {SCALER_PATH}\n   Lance d'abord train.py")

    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    print(f" Modèle chargé  : {MODEL_PATH}")
    print(f" Scaler chargé  : {SCALER_PATH}")
    print(f"   Type modèle    : {type(model).__name__}")

    return model, scaler


# ══════════════════════════════════════════════
# 2. PRÉDICTION SUR UN SEUL CLIENT
# ══════════════════════════════════════════════
def predict_single(client_data: dict, model, scaler) -> dict:
    """
    Prédit l'approbation du prêt pour un seul client.

    Paramètres :
        client_data : dict avec les features du client
        model       : modèle chargé depuis model.pkl
        scaler      : scaler chargé depuis scaler.pkl

    Retourne :
        dict avec prediction, probabilité et détails
    """
    # Convertir en DataFrame
    df = pd.DataFrame([client_data])

    # Appliquer le scaler (même transformation que pendant l'entraînement)
    df_scaled = pd.DataFrame(
        scaler.transform(df),
        columns=df.columns
    )

    # Prédiction
    prediction_encoded = model.predict(df_scaled)[0]
    probabilities      = model.predict_proba(df_scaled)[0]

    # Décoder : 0 = Approved, 1 = Rejected
    prediction_label = 'Approved' if prediction_encoded == 0 else 'Rejected'
    probability      = probabilities[prediction_encoded]

    return {
        'prediction':          prediction_label,
        'probability':         round(float(probability), 4),
        'prob_approved':       round(float(probabilities[0]), 4),
        'prob_rejected':       round(float(probabilities[1]), 4),
        'prediction_encoded':  int(prediction_encoded),
    }


# ══════════════════════════════════════════════
# 3. PRÉDICTION SUR PLUSIEURS CLIENTS
# ══════════════════════════════════════════════
def predict_batch(clients_data: list, model, scaler) -> list:
    """
    Prédit l'approbation pour une liste de clients.

    Paramètres :
        clients_data : liste de dicts
        model        : modèle chargé
        scaler       : scaler chargé

    Retourne :
        liste de dicts avec prédictions
    """
    df = pd.DataFrame(clients_data)

    df_scaled = pd.DataFrame(
        scaler.transform(df),
        columns=df.columns
    )

    predictions_encoded = model.predict(df_scaled)
    probabilities       = model.predict_proba(df_scaled)

    results = []
    for i, (pred, probs) in enumerate(zip(predictions_encoded, probabilities)):
        label = 'Approved' if pred == 0 else 'Rejected'
        results.append({
            'client_index':        i + 1,
            'prediction':          label,
            'probability':         round(float(probs[pred]), 4),
            'prob_approved':       round(float(probs[0]), 4),
            'prob_rejected':       round(float(probs[1]), 4),
            'prediction_encoded':  int(pred),
        })

    return results


# ══════════════════════════════════════════════
# 4. AFFICHAGE DU RÉSULTAT
# ══════════════════════════════════════════════
def display_result(result: dict, client_data: dict = None):
    """Affiche le résultat de prédiction de façon lisible."""
    emoji = 'yes' if result['prediction'] == 'Approved' else 'no'

    print("\n" + "─" * 50)
    print(f"  RÉSULTAT DE LA PRÉDICTION")
    print("─" * 50)

    if client_data:
        print(f"\n   Données client :")
        for key, value in client_data.items():
            print(f"     {key:<30} : {value}")

    print(f"\n  {emoji} Décision        : {result['prediction']}")
    print(f"   Confiance       : {result['probability']*100:.1f}%")
    print(f"   P(Approved)     : {result['prob_approved']*100:.1f}%")
    print(f"   P(Rejected)     : {result['prob_rejected']*100:.1f}%")
    print("─" * 50)


# ══════════════════════════════════════════════
# 5. PIPELINE PRINCIPAL — TEST AVEC 3 CLIENTS
# ══════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 50)
    print("  STEP 3 — SCRIPT DE PRÉDICTION")
    print("=" * 50)

    # Charger le modèle et le scaler
    model, scaler = load_model()

    # ── Client 1 : profil solide → attendu Approved
    client_1 = {
        'no_of_dependents':           2,
        'education':                  0,    # Graduate
        'self_employed':              0,    # No
        'income_annum':         9600000,
        'loan_amount':         29900000,
        'loan_term':                 12,
        'cibil_score':              778,    # bon score
        'residential_assets_value': 2400000,
        'commercial_assets_value':  17600000,
        'luxury_assets_value':      22700000,
        'bank_asset_value':         8000000,
    }

    # ── Client 2 : profil risqué → attendu Rejected
    client_2 = {
        'no_of_dependents':           0,
        'education':                  1,    # Not Graduate
        'self_employed':              1,    # Yes
        'income_annum':         4100000,
        'loan_amount':         12200000,
        'loan_term':                  8,
        'cibil_score':              417,    # mauvais score
        'residential_assets_value': 2700000,
        'commercial_assets_value':  2200000,
        'luxury_assets_value':      8800000,
        'bank_asset_value':         3300000,
    }

    # ── Client 3 : profil intermédiaire
    client_3 = {
        'no_of_dependents':           1,
        'education':                  0,    # Graduate
        'self_employed':              0,    # No
        'income_annum':         6500000,
        'loan_amount':         15000000,
        'loan_term':                 15,
        'cibil_score':              620,    # score moyen
        'residential_assets_value': 5000000,
        'commercial_assets_value':  3000000,
        'luxury_assets_value':      9000000,
        'bank_asset_value':         4000000,
    }

    # ── Prédictions individuelles
    print("\n PRÉDICTIONS INDIVIDUELLES")
    for i, client in enumerate([client_1, client_2, client_3], 1):
        print(f"\n  👤 Client {i} :")
        result = predict_single(client, model, scaler)
        display_result(result, client)

    # ── Prédiction batch
    print("\n PRÉDICTION BATCH (3 clients)")
    batch_results = predict_batch([client_1, client_2, client_3], model, scaler)
    print(f"\n  {'Client':<10} {'Décision':<12} {'Confiance':<12} {'P(Approved)':<14} {'P(Rejected)'}")
    print("  " + "─" * 60)
    for r in batch_results:
        emoji = 'yes' if r['prediction'] == 'Approved' else 'no'
        print(f"  {r['client_index']:<10} {emoji} {r['prediction']:<10} "
              f"{r['probability']*100:<12.1f}% "
              f"{r['prob_approved']*100:<14.1f}% "
              f"{r['prob_rejected']*100:.1f}%")

   