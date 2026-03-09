"""
================================================
FEATURE ENGINEERING - Loan Approval Dataset
================================================
Crée de nouvelles features à partir des colonnes
existantes pour améliorer les performances du modèle.

Ordre dans le pipeline :
load_data() → clean_data() → feature_engineering()
→ encode_features() → split_data()
================================================
"""

import pandas as pd
import numpy as np


def feature_engineering(df):
    """
    Crée 5 nouvelles features financières pertinentes.
    Doit être appelé APRÈS clean_data() et AVANT encode_features().
    """
    df = df.copy()

    print("\n" + "─" * 50)
    print("  FEATURE ENGINEERING")
    print("─" * 50)
    print(f"  Colonnes avant : {df.shape[1]}")

    # ── 1. Ratio dette / revenu ──────────────────
    # Plus ce ratio est élevé → plus le risque est grand
    # Ex: 29900000 / 9600000 = 3.11 → emprunte 3x son revenu annuel
    df['debt_to_income'] = df['loan_amount'] / df['income_annum']

    # ── 2. Total des assets ──────────────────────
    # Somme de toutes les garanties du client
    # Ex: 2.4M + 17.6M + 22.7M + 8M = 50.7M
    df['total_assets'] = (
        df['residential_assets_value'] +
        df['commercial_assets_value'] +
        df['luxury_assets_value'] +
        df['bank_asset_value']
    )

    # ── 3. Ratio assets / loan ───────────────────
    # Capacité de garantie → plus c'est élevé → moins risqué
    # Ex: 50700000 / 29900000 = 1.69 → a 1.69x de garanties
    df['assets_to_loan'] = df['total_assets'] / df['loan_amount']

    # ── 4. Mensualité estimée ────────────────────
    # Montant à rembourser chaque mois
    # Ex: 29900000 / (12 * 12) = 207 638 / mois
    df['monthly_payment'] = df['loan_amount'] / (df['loan_term'] * 12)

    # ── 5. Ratio mensualité / revenu mensuel ─────
    # Charge mensuelle par rapport au revenu
    # Ex: 207638 / (9600000/12) = 0.26 → paie 26% de son revenu
    df['payment_to_income'] = df['monthly_payment'] / (df['income_annum'] / 12)

    print(f"  Colonnes après  : {df.shape[1]}")
    print(f"\n  5 nouvelles features créées :")
    print(f"     → debt_to_income    : loan_amount / income_annum")
    print(f"     → total_assets      : somme des 4 assets")
    print(f"     → assets_to_loan    : total_assets / loan_amount")
    print(f"     → monthly_payment   : loan_amount / (loan_term * 12)")
    print(f"     → payment_to_income : monthly_payment / revenu mensuel")

    # ── Aperçu des nouvelles features ───────────
    new_features = [
        'debt_to_income', 'total_assets',
        'assets_to_loan', 'monthly_payment', 'payment_to_income'
    ]
    print(f"\n   Statistiques des nouvelles features :")
    print(df[new_features].describe().round(3).to_string())

    return df