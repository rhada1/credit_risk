"""
================================================
STEP 2 — ENTRAÎNEMENT DU MODÈLE ML
Projet : Prédiction d'Acceptation de Prêt
Dataset : Loan Approval Prediction (Kaggle)
Modèles : Logistic Regression, Random Forest, XGBoost
Correction : paramètres de régularisation pour éviter l'overfitting
================================================
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)
from sklearn.model_selection import cross_val_score, train_test_split
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ─── Chemins ────────────────────────────────
PREPROCESSED_PATH = './data/loan_approval_preprocessed.csv'
MODEL_PATH        = './models/model.pkl'
SCALER_PATH       = './models/scaler.pkl'
REPORT_PATH       = './data/training_report.txt'

# ─── Style global ───────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#F8F9FA',
    'axes.facecolor':   '#FFFFFF',
    'axes.grid':        True,
    'grid.alpha':       0.3,
    'font.family':      'DejaVu Sans',
    'axes.titlesize':   13,
    'axes.labelsize':   11,
})


# ══════════════════════════════════════════════
# 1. CHARGEMENT & SPLIT
# ══════════════════════════════════════════════
def load_and_split():
    """Charge le dataset prétraité et sépare train/test."""
    df = pd.read_csv(PREPROCESSED_PATH)

    X = df.drop(columns=['loan_status'])
    y = df['loan_status']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"✅ Dataset chargé    : {df.shape[0]} lignes × {df.shape[1]} colonnes")
    print(f"   Train             : {X_train.shape[0]} lignes")
    print(f"   Test              : {X_test.shape[0]} lignes")
    print(f"   Features          : {X_train.shape[1]} colonnes")

    ratio = y.value_counts(normalize=True) * 100
    print(f"\n📊 Distribution cible :")
    for k, v in ratio.items():
        label = 'Approved' if k == 0 else 'Rejected'
        print(f"   {label} ({k}) : {v:.1f}%")

    return X_train, X_test, y_train, y_test


# ══════════════════════════════════════════════
# 2. NORMALISATION
# ══════════════════════════════════════════════
def scale_data(X_train, X_test):
    """
    Applique StandardScaler fitté UNIQUEMENT sur X_train.
    Évite le data leakage sur X_test.
    """
    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns
    )

    os.makedirs('./models', exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\n💾 Scaler sauvegardé : {SCALER_PATH}")

    return X_train_scaled, X_test_scaled, scaler


# ══════════════════════════════════════════════
# 3. ENTRAÎNEMENT DES MODÈLES
# ══════════════════════════════════════════════
def train_models(X_train, X_test, y_train, y_test):
    """Entraîne et compare 3 modèles ML avec régularisation."""

    # Ratio pour corriger le déséquilibre dans XGBoost
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    models = {
        'Logistic Regression': LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced',
            C=0.1                       # régularisation forte → évite l'overfitting
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced',
            max_depth=10,               # limite la profondeur des arbres
            min_samples_leaf=10,        # minimum 10 exemples par feuille
            max_features='sqrt',        # sqrt des features par arbre
            n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100,
            random_state=42,
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss',
            verbosity=0,
            max_depth=6,                # limite la profondeur
            min_child_weight=10,        # minimum d'exemples par nœud
            subsample=0.8,              # 80% des données par arbre
            colsample_bytree=0.8,       # 80% des features par arbre
            reg_alpha=0.1,              # régularisation L1
            reg_lambda=1.0              # régularisation L2
        ),
    }

    results = {}

    print("\n" + "─" * 55)
    print("  COMPARAISON DES MODÈLES")
    print("─" * 55)

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc       = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall    = recall_score(y_test, y_pred, zero_division=0)
        f1        = f1_score(y_test, y_pred, zero_division=0)
        auc       = roc_auc_score(y_test, y_prob)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')

        results[name] = {
            'model':     model,
            'accuracy':  acc,
            'precision': precision,
            'recall':    recall,
            'f1':        f1,
            'auc':       auc,
            'cv_mean':   cv_scores.mean(),
            'cv_std':    cv_scores.std(),
            'y_pred':    y_pred,
            'y_prob':    y_prob,
        }

        print(f"\n📊 {name}")
        print(f"   Accuracy  : {acc:.4f}")
        print(f"   Precision : {precision:.4f}")
        print(f"   Recall    : {recall:.4f}")
        print(f"   F1 Score  : {f1:.4f}")
        print(f"   AUC-ROC   : {auc:.4f}")
        print(f"   CV AUC    : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return results


# ══════════════════════════════════════════════
# 4. SÉLECTION DU MEILLEUR MODÈLE
# ══════════════════════════════════════════════
def select_best_model(results):
    """Sélectionne le modèle avec le meilleur AUC-ROC."""
    best_name = max(results, key=lambda k: results[k]['auc'])
    best      = results[best_name]

    print("\n" + "─" * 55)
    print(f"  🏆 MEILLEUR MODÈLE : {best_name}")
    print(f"     AUC-ROC   : {best['auc']:.4f}")
    print(f"     Accuracy  : {best['accuracy']:.4f}")
    print(f"     F1 Score  : {best['f1']:.4f}")
    print(f"     CV AUC    : {best['cv_mean']:.4f} ± {best['cv_std']:.4f}")
    print("─" * 55)

    return best_name, best['model']


# ══════════════════════════════════════════════
# 5. SAUVEGARDE DU MODÈLE
# ══════════════════════════════════════════════
def save_model(model, model_name):
    """Sauvegarde le meilleur modèle en fichier .pkl"""
    os.makedirs('./models', exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\n💾 Modèle sauvegardé  : {MODEL_PATH}")
    print(f"   Type              : {model_name}")


# ══════════════════════════════════════════════
# 6. VISUALISATIONS
# ══════════════════════════════════════════════
def plot_results(results, y_test, best_name):
    """Génère les graphiques d'évaluation des modèles."""
    fig = plt.figure(figsize=(18, 12), facecolor='#F0F4F8')
    fig.suptitle(
        f'STEP 2 — Évaluation des Modèles ML\n🏆 Meilleur modèle : {best_name}',
        fontsize=16, fontweight='bold', y=0.98, color='#2C3E50'
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    colors = ['#3498DB', '#2ECC71', '#E74C3C']
    names  = list(results.keys())

    # (a) Comparaison des métriques
    ax1 = fig.add_subplot(gs[0, 0])
    metrics       = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
    x     = np.arange(len(metrics))
    width = 0.25
    for i, (name, color) in enumerate(zip(names, colors)):
        vals = [results[name][m] for m in metrics]
        ax1.bar(x + i * width, vals, width, label=name,
                color=color, alpha=0.85, edgecolor='white')
    ax1.set_title('📊 Comparaison des Métriques', fontweight='bold')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(metric_labels, fontsize=9)
    ax1.set_ylim(0, 1.15)
    ax1.legend(fontsize=8)

    # (b) Matrice de confusion — meilleur modèle
    ax2 = fig.add_subplot(gs[0, 1])
    cm = confusion_matrix(y_test, results[best_name]['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
                xticklabels=['Approved', 'Rejected'],
                yticklabels=['Approved', 'Rejected'],
                linewidths=2, linecolor='white')
    ax2.set_title(f'🔲 Matrice de Confusion\n{best_name}', fontweight='bold')
    ax2.set_ylabel('Réel')
    ax2.set_xlabel('Prédit')

    # (c) Courbes ROC
    ax3 = fig.add_subplot(gs[0, 2])
    for (name, res), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
        ax3.plot(fpr, tpr, color=color, lw=2,
                 label=f"{name} (AUC={res['auc']:.3f})")
    ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    ax3.set_title('📈 Courbes ROC', fontweight='bold')
    ax3.set_xlabel('Taux Faux Positifs')
    ax3.set_ylabel('Taux Vrais Positifs')
    ax3.legend(fontsize=8)

    # (d) Feature Importance — meilleur modèle
    ax4 = fig.add_subplot(gs[1, :2])
    best_model = results[best_name]['model']
    if hasattr(best_model, 'feature_importances_'):
        feature_names = (
            best_model.feature_names_in_
            if hasattr(best_model, 'feature_names_in_')
            else [f'f{i}' for i in range(len(best_model.feature_importances_))]
        )
        feat_df = pd.DataFrame({
            'feature':    feature_names,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=True).tail(10)
        ax4.barh(feat_df['feature'], feat_df['importance'],
                 color='#3498DB', edgecolor='white', alpha=0.85)
        ax4.set_title(f'🔍 Top 10 Features — {best_name}', fontweight='bold')
        ax4.set_xlabel('Importance')

    # (e) CV AUC Scores
    ax5 = fig.add_subplot(gs[1, 2])
    cv_means = [results[n]['cv_mean'] for n in names]
    cv_stds  = [results[n]['cv_std']  for n in names]
    bars = ax5.bar(names, cv_means, color=colors, edgecolor='white',
                   alpha=0.85, width=0.5, yerr=cv_stds, capsize=5)
    for bar, val in zip(bars, cv_means):
        ax5.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.01,
                 f'{val:.4f}', ha='center', fontweight='bold', fontsize=10)
    ax5.set_title('🔁 CV AUC Score (5-fold)', fontweight='bold')
    ax5.set_ylabel('AUC Moyen')
    ax5.set_ylim(0, 1.15)
    ax5.tick_params(axis='x', rotation=15)

    plt.savefig('./data/fig3_model_evaluation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Figure sauvegardée  → data/fig3_model_evaluation.png")


# ══════════════════════════════════════════════
# 7. RAPPORT TEXTE
# ══════════════════════════════════════════════
def save_report(results, best_name, y_test):
    """Sauvegarde le rapport complet dans un fichier texte."""
    lines = []
    lines.append("=" * 55)
    lines.append("  STEP 2 — RAPPORT D'ENTRAÎNEMENT")
    lines.append("=" * 55)

    for name, res in results.items():
        lines.append(f"\n📊 {name}")
        lines.append(f"   Accuracy  : {res['accuracy']:.4f}")
        lines.append(f"   Precision : {res['precision']:.4f}")
        lines.append(f"   Recall    : {res['recall']:.4f}")
        lines.append(f"   F1 Score  : {res['f1']:.4f}")
        lines.append(f"   AUC-ROC   : {res['auc']:.4f}")
        lines.append(f"   CV AUC    : {res['cv_mean']:.4f} ± {res['cv_std']:.4f}")

    lines.append(f"\n{'─' * 55}")
    lines.append(f"🏆 MEILLEUR MODÈLE : {best_name}")
    lines.append(f"{'─' * 55}")
    lines.append(f"\n📋 Classification Report — {best_name} :")
    lines.append(classification_report(
        y_test, results[best_name]['y_pred'],
        target_names=['Approved', 'Rejected']
    ))

    rapport = "\n".join(lines)
    print(rapport)

    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(rapport)
    print(f"\n💾 Rapport sauvegardé : {REPORT_PATH}")


# ══════════════════════════════════════════════
# 8. PIPELINE PRINCIPAL
# ══════════════════════════════════════════════
def run():
    print("=" * 55)
    print("  STEP 2 — ENTRAÎNEMENT DU MODÈLE ML")
    print("=" * 55)

    X_train, X_test, y_train, y_test = load_and_split()
    X_train, X_test, scaler          = scale_data(X_train, X_test)
    results                           = train_models(X_train, X_test, y_train, y_test)
    best_name, best_model             = select_best_model(results)

    save_model(best_model, best_name)
    plot_results(results, y_test, best_name)
    save_report(results, best_name, y_test)

    print("\n" + "=" * 55)
    print("  ✅ STEP 2 TERMINÉ — Prêt pour predict.py")
    print("=" * 55)

    return best_model, scaler


if __name__ == "__main__":
    run()