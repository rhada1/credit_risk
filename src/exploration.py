"""
================================================
STEP 1 — EXPLORATION DU DATASET
Projet : Prédiction d'Acceptation de Prêt
Dataset : Loan Approval Prediction (Kaggle)
================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ─── Chemins ────────────────────────────────
DATA_PATH = './data/loan_approval_dataset.csv'

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
COLORS  = {'Approved': '#2ECC71', 'Rejected': '#E74C3C'}
PALETTE = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12',
           '#9B59B6', '#1ABC9C', '#E67E22', '#34495E']


# ══════════════════════════════════════════════
# 1. CHARGEMENT & APERÇU GÉNÉRAL
# ══════════════════════════════════════════════
def load_and_overview(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip()

    # ─── Construire le rapport ───────────────────
    lines = []
    lines.append("=" * 60)
    lines.append("  STEP 1 — EXPLORATION DU LOAN APPROVAL DATASET")
    lines.append("=" * 60)
    lines.append(f"\n Dimensions         : {df.shape[0]} lignes × {df.shape[1]} colonnes")
    lines.append(f" Variable cible     : loan_status")
    lines.append(f"\n Colonnes & types :")
    lines.append(df.dtypes.to_string())
    lines.append(f"\n Statistiques descriptives :")
    lines.append(df.describe().round(2).to_string())
    lines.append(f"\n Valeurs manquantes :")
    missing = df.isnull().sum()
    lines.append("   Aucune valeur manquante " if missing.sum() == 0 else missing[missing > 0].to_string())
    lines.append(f"\n Distribution de la cible :")
    lines.append(df['loan_status'].value_counts().to_string())
    lines.append(df['loan_status'].value_counts(normalize=True).mul(100).round(1).astype(str).add('%').to_string())

    rapport = "\n".join(lines)

    # ─── Afficher dans le terminal ───────────────
    print(rapport)

    # ─── Sauvegarder dans un fichier texte ───────
    with open('./data/exploration_report.txt', 'w', encoding='utf-8') as f:
        f.write(rapport)
    print("\n Rapport sauvegardé → data/exploration_report.txt")

    return df


# ══════════════════════════════════════════════
# 2. FIGURE 1 — VUE D'ENSEMBLE
# ══════════════════════════════════════════════
def plot_overview(df):
    fig = plt.figure(figsize=(18, 14), facecolor='#F0F4F8')
    fig.suptitle(
        'STEP 1 — Exploration du Loan Approval Dataset\nVue d\'ensemble',
        fontsize=17, fontweight='bold', y=0.98, color='#2C3E50'
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # (a) Distribution cible
    ax1 = fig.add_subplot(gs[0, 0])
    counts = df['loan_status'].value_counts()
    bars = ax1.bar(counts.index, counts.values,
                   color=[COLORS.get(k, '#3498DB') for k in counts.index],
                   edgecolor='white', linewidth=2, width=0.5)
    for bar, val in zip(bars, counts.values):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 10,
                 f'{val}\n({val/len(df)*100:.0f}%)',
                 ha='center', fontweight='bold', fontsize=11)
    ax1.set_title('🎯 Distribution de la Cible', fontweight='bold')
    ax1.set_ylabel('Nombre de demandes')
    ax1.set_ylim(0, counts.max() * 1.2)

    # (b) Revenu annuel
    ax2 = fig.add_subplot(gs[0, 1])
    if 'income_annum' in df.columns:
        for status, color in COLORS.items():
            if status in df['loan_status'].values:
                subset = df[df['loan_status'] == status]['income_annum']
                ax2.hist(subset, bins=25, alpha=0.65,
                         color=color, label=status, edgecolor='white')
        ax2.set_title('💰 Revenu Annuel par Statut', fontweight='bold')
        ax2.set_xlabel('Revenu annuel')
        ax2.set_ylabel('Fréquence')
        ax2.legend(fontsize=9)

    # (c) Montant du prêt
    ax3 = fig.add_subplot(gs[0, 2])
    if 'loan_amount' in df.columns:
        for status, color in COLORS.items():
            if status in df['loan_status'].values:
                subset = df[df['loan_status'] == status]['loan_amount']
                ax3.hist(subset, bins=25, alpha=0.65,
                         color=color, label=status, edgecolor='white')
        ax3.set_title('🏦 Montant du Prêt par Statut', fontweight='bold')
        ax3.set_xlabel('Montant du prêt')
        ax3.set_ylabel('Fréquence')
        ax3.legend(fontsize=9)

    # (d) CIBIL Score
    ax4 = fig.add_subplot(gs[1, 0])
    if 'cibil_score' in df.columns:
        bp = ax4.boxplot(
            [df[df['loan_status'] == s]['cibil_score'].dropna()
             for s in df['loan_status'].unique()],
            labels=df['loan_status'].unique(),
            patch_artist=True,
            medianprops=dict(color='white', linewidth=2)
        )
        colors_list = [COLORS.get(s, '#3498DB')
                       for s in df['loan_status'].unique()]
        for patch, color in zip(bp['boxes'], colors_list):
            patch.set_facecolor(color)
        ax4.set_title(' CIBIL Score × Statut du Prêt', fontweight='bold')
        ax4.set_ylabel('CIBIL Score')

    # (e) Niveau d'éducation
    ax5 = fig.add_subplot(gs[1, 1])
    if 'education' in df.columns:
        edu_risk = df.groupby(['education', 'loan_status']).size().unstack(fill_value=0)
        pct = edu_risk.div(edu_risk.sum(axis=1), axis=0) * 100
        pct.plot(kind='bar', ax=ax5,
                 color=[COLORS.get(c, '#3498DB') for c in pct.columns],
                 edgecolor='white', linewidth=1.5)
        ax5.set_title('🎓 Éducation × Taux Approbation (%)', fontweight='bold')
        ax5.set_ylabel('Pourcentage (%)')
        ax5.tick_params(axis='x', rotation=0)
        ax5.set_ylim(0, 110)

    # (f) Statut emploi
    ax6 = fig.add_subplot(gs[1, 2])
    if 'self_employed' in df.columns:
        emp_risk = df.groupby(['self_employed', 'loan_status']).size().unstack(fill_value=0)
        pct2 = emp_risk.div(emp_risk.sum(axis=1), axis=0) * 100
        pct2.plot(kind='bar', ax=ax6,
                  color=[COLORS.get(c, '#3498DB') for c in pct2.columns],
                  edgecolor='white', linewidth=1.5)
        ax6.set_title('💼 Auto-Entrepreneur × Approbation (%)', fontweight='bold')
        ax6.set_ylabel('Pourcentage (%)')
        ax6.tick_params(axis='x', rotation=0)
        ax6.set_ylim(0, 110)

    plt.savefig('./data/fig1_overview.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n Figure 1 sauvegardée → data/fig1_overview.png")


# ══════════════════════════════════════════════
# 3. FIGURE 2 — CORRÉLATIONS & FEATURES
# ══════════════════════════════════════════════
def plot_correlations(df):
    fig2, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor='#F0F4F8')
    fig2.suptitle(
        'STEP 1 — Corrélations & Analyse des Features',
        fontsize=16, fontweight='bold', y=0.98, color='#2C3E50'
    )

    # Encoder pour la heatmap
    df_enc = df.copy()
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for col in df_enc.select_dtypes(include='object').columns:
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))

    # (a) Heatmap corrélation
    ax = axes[0, 0]
    corr = df_enc.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, ax=ax, mask=mask, annot=True, fmt='.2f',
                cmap='RdYlGn', center=0, square=True,
                linewidths=0.5, annot_kws={'size': 7},
                cbar_kws={'shrink': 0.8})
    ax.set_title(' Matrice de Corrélation', fontweight='bold')
    ax.tick_params(axis='x', rotation=45, labelsize=7)
    ax.tick_params(axis='y', rotation=0, labelsize=7)

    # (b) CIBIL Score vs Montant prêt
    ax = axes[0, 1]
    if 'cibil_score' in df.columns and 'loan_amount' in df.columns:
        for status, color in COLORS.items():
            if status in df['loan_status'].values:
                sub = df[df['loan_status'] == status]
                ax.scatter(sub['cibil_score'], sub['loan_amount'],
                           c=color, alpha=0.4, s=15, label=status)
        ax.set_title(' CIBIL Score vs Montant du Prêt', fontweight='bold')
        ax.set_xlabel('CIBIL Score')
        ax.set_ylabel('Montant du Prêt')
        ax.legend()

    # (c) Nombre de dépendants
    ax = axes[1, 0]
    if 'no_of_dependents' in df.columns:
        dep_risk = df.groupby(['no_of_dependents', 'loan_status']).size().unstack(fill_value=0)
        pct = dep_risk.div(dep_risk.sum(axis=1), axis=0) * 100
        pct.plot(kind='bar', ax=ax,
                 color=[COLORS.get(c, '#3498DB') for c in pct.columns],
                 edgecolor='white')
        ax.set_title(' Dépendants × Taux Approbation (%)', fontweight='bold')
        ax.set_ylabel('Pourcentage (%)')
        ax.tick_params(axis='x', rotation=0)

    # (d) Durée du prêt
    ax = axes[1, 1]
    if 'loan_term' in df.columns:
        for status, color in COLORS.items():
            if status in df['loan_status'].values:
                subset = df[df['loan_status'] == status]['loan_term']
                ax.hist(subset, bins=15, alpha=0.65,
                        color=color, label=status, edgecolor='white')
        ax.set_title(' Durée du Prêt × Statut', fontweight='bold')
        ax.set_xlabel('Durée (années)')
        ax.set_ylabel('Fréquence')
        ax.legend()

    plt.tight_layout()
    plt.savefig('./data/fig2_correlations.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(" Figure 2 sauvegardée → data/fig2_correlations.png")


# ══════════════════════════════════════════════
# 4. PIPELINE PRINCIPAL
# ══════════════════════════════════════════════
def run():
    df = load_and_overview(DATA_PATH)
    plot_overview(df)
    plot_correlations(df)

    print("\n" + "=" * 60)
    print("  STEP 1 TERMINÉ — Prêt pour preprocess.py")
    print("=" * 60)

    return df


if __name__ == "__main__":
    run()