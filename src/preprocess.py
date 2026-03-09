"""
================================================
PRÉTRAITEMENT DES DONNÉES - Loan Approval Dataset
================================================
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from feature_engineering import feature_engineering
import os

# ─── Chemins ────────────────────────────────
DATA_PATH   = './data/loan_approval_dataset.csv'
OUTPUT_PATH = './data/loan_approval_preprocessed.csv'


def load_data():
    """Charge le dataset brut."""
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    print(f"✅ Dataset chargé : {df.shape[0]} lignes × {df.shape[1]} colonnes")
    print(f"\n📋 Colonnes : {list(df.columns)}")
    print(f"\n❓ Valeurs manquantes :\n{df.isnull().sum()}")
    return df


def clean_data(df):
    """Nettoie les données brutes."""
    df = df.copy()

    # Supprimer colonne id inutile pour le ML
    if 'loan_id' in df.columns:
        df.drop(columns=['loan_id'], inplace=True)
        print("\n🗑️  Colonne 'loan_id' supprimée")

    # Nettoyer les espaces dans les valeurs texte
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip()

    print(f"\n📊 Distribution de la cible :")
    print(df['loan_status'].value_counts())

    return df


def encode_features(df):
    """Encode les variables catégorielles en chiffres."""
    df = df.copy()
    le = LabelEncoder()

    cat_cols = df.select_dtypes(include='object').columns.tolist()

    for col in cat_cols:
        df[col] = le.fit_transform(df[col])
        print(f"✅ Encodé : {col}")

    # Résultat de l'encodage cible
    print(f"\n🎯 Encodage loan_status : Approved=0, Rejected=1")

    return df


def split_data(df):
    """Sépare features/cible puis train/test (80/20)."""
    X = df.drop(columns=['loan_status'])
    y = df['loan_status']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y      # garde la proportion Approved/Rejected
    )

    print(f"\n✂️  Split train/test :")
    print(f"   Train : {X_train.shape[0]} lignes")
    print(f"   Test  : {X_test.shape[0]} lignes")

    return X_train, X_test, y_train, y_test


def run():
    """Pipeline complet de prétraitement."""
    print("=" * 55)
    print("  PRÉTRAITEMENT — Loan Approval Dataset")
    print("=" * 55)

    df = load_data()
    df = clean_data(df)
    df = encode_features(df)

    # Sauvegarder le dataset encodé (avant normalisation)
    os.makedirs('./data', exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n💾 Dataset encodé sauvegardé : {OUTPUT_PATH}")
    print(f"   → {df.shape[0]} lignes × {df.shape[1]} colonnes")

    X_train, X_test, y_train, y_test = split_data(df)

    print("\n" + "=" * 55)
   

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    run()
