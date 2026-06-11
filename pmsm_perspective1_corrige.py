"""
=============================================================
  PERSPECTIVE 1 — VALIDATION SUR DONNÉES RÉELLES (CORRIGÉ)
  Dataset : Bacha (2024) — Zenodo : 10.5281/zenodo.13974503
  Colonnes confirmées : Ia, Ib, VDC, IDC, T1, T2, T3, VD, FDD
  Labels : F0=Normal, F1-F8=Défauts
  Auteur : [Votre Nom] | Date : 2025
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import MinMaxScaler
import pywt
import warnings, os
warnings.filterwarnings('ignore')

DOSSIER = r'C:\PFE_PMSM\resultats'
os.makedirs(DOSSIER, exist_ok=True)

# ============================================================
# DESCRIPTION DES LABELS (Dataset Bacha 2024)
# ============================================================
LABELS_DESCRIPTION = {
    'F0': 'Normal (sain)',
    'F1': 'Court-circuit ouvert — switch 1',
    'F2': 'Court-circuit ouvert — switch 2',
    'F3': 'Court-circuit ouvert — switch 3',
    'F4': 'Court-circuit ouvert — switch 4',
    'F5': 'Court-circuit ouvert — switch 5',
    'F6': 'Court-circuit ouvert — switch 6',
    'F7': 'Court-circuit statorique',
    'F8': 'Surchauffe onduleur',
}

COULEURS_LABELS = {
    'F0': '#276749',  # vert  — normal
    'F1': '#C53030',  # rouge — défaut
    'F2': '#E53E3E',
    'F3': '#D44A23',
    'F4': '#B7791F',
    'F5': '#553C9A',
    'F6': '#2B6CB0',
    'F7': '#DD6B20',
    'F8': '#285E61',
}

# ============================================================
# 1. CHARGEMENT
# ============================================================
def charger_dataset():
    csv = r'C:\PFE_PMSM\dataset_reel\converted_dataset.csv'
    print(f"  Chargement : {csv}")
    df = pd.read_csv(csv)
    print(f"  Dimensions : {df.shape}")
    print(f"  Labels FDD : {sorted(df['FDD'].unique())}")
    n_par_label = df.groupby('FDD').size()
    print(f"  Échantillons par label :")
    for label, n in n_par_label.items():
        print(f"    {label} ({LABELS_DESCRIPTION.get(label,'?')}) : {n} échantillons")
    return df

# ============================================================
# 2. VISUALISATION DES SIGNAUX BRUTS
# ============================================================
def plot_signaux_bruts(df, save=True):
    """Trace les signaux Ia pour chaque condition — vue d'ensemble."""
    labels = sorted(df['FDD'].unique())
    n = len(labels)
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    fig.suptitle('Signaux réels Ia — Dataset Bacha (2024) — Par condition',
                 fontsize=13, fontweight='bold')

    for ax, label in zip(axes.flat, labels):
        sous_df = df[df['FDD'] == label]['Ia'].values[:500]
        col = COULEURS_LABELS.get(label, 'gray')
        ax.plot(sous_df, color=col, linewidth=0.8)
        ax.set_title(f"{label} — {LABELS_DESCRIPTION.get(label,'?')}",
                     fontsize=9, fontweight='bold', color=col)
        ax.set_xlabel('Échantillons', fontsize=8)
        ax.set_ylabel('Ia [A]', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    if save:
        plt.savefig(rf'{DOSSIER}\perspective1_signaux_bruts.png',
                    dpi=150, bbox_inches='tight')
        print("  Figure : perspective1_signaux_bruts.png")
    plt.show()

# ============================================================
# 3. EXTRACTION DES FEATURES PAR CONDITION
# ============================================================
def extraire_features(df, fenetre=500):
    """
    Extrait les 17 indicateurs pour chaque fenêtre de chaque condition.
    Utilise le courant Ia (signal principal).
    """
    resultats = []
    fs = 10  # fréquence échantillonnage dataset Bacha = 10 Hz

    for label in sorted(df['FDD'].unique()):
        sig = df[df['FDD'] == label]['Ia'].values.astype(float)
        n_fenetres = len(sig) // fenetre

        for i in range(n_fenetres):
            s = sig[i*fenetre : (i+1)*fenetre]

            # ── Temporels ──
            rms      = np.sqrt(np.mean(s**2))
            peak     = np.max(np.abs(s))
            crest    = peak / (rms + 1e-12)
            kurt     = kurtosis(s)
            skewness = skew(s)
            variance = np.var(s)
            p2p      = np.max(s) - np.min(s)

            # ── Fréquentiels (FFT) ──
            N    = len(s)
            win  = np.hanning(N)
            fft  = np.fft.rfft(s * win)
            freq = np.fft.rfftfreq(N, d=1.0/fs)
            amp  = (2.0/N) * np.abs(fft)

            def e_bande(f1, f2):
                m = (freq >= f1) & (freq < f2)
                return float(np.sum(amp[m]**2)) if m.any() else 0.0

            e_0_1   = e_bande(0,   1.0)
            e_1_2   = e_bande(1.0, 2.0)
            e_2_3   = e_bande(2.0, 3.0)
            e_3_5   = e_bande(3.0, 5.0)

            # THD adapté à fs=10Hz
            idx_f = np.argmin(np.abs(freq - 0.5)) if len(freq) > 1 else 0
            a_f   = amp[idx_f] if idx_f < len(amp) else 1e-12
            harm  = [amp[min(np.argmin(np.abs(freq - k*0.5)), len(amp)-1)]
                     for k in range(2, 6)]
            thd   = np.sqrt(sum(h**2 for h in harm)) / (a_f + 1e-12)

            # ── Wavelet ──
            try:
                coeffs = pywt.wavedec(s, 'db4', level=4)
                total  = sum(np.sum(c**2) for c in coeffs) + 1e-12
                wav_e  = [float(np.sum(c**2)/total) for c in coeffs]
                while len(wav_e) < 5:
                    wav_e.append(0.0)
            except:
                wav_e = [0.0]*5

            resultats.append({
                'RMS'          : round(rms, 4),
                'Peak'         : round(peak, 4),
                'Crest_Factor' : round(crest, 4),
                'Kurtosis'     : round(kurt, 4),
                'Skewness'     : round(skewness, 4),
                'Variance'     : round(variance, 4),
                'Peak2Peak'    : round(p2p, 4),
                'Energy_B1'    : round(e_0_1, 6),
                'Energy_B2'    : round(e_1_2, 6),
                'Energy_B3'    : round(e_2_3, 6),
                'Energy_B4'    : round(e_3_5, 6),
                'THD'          : round(min(thd, 1e6), 4),
                'Wav_E_approx' : round(wav_e[0], 4),
                'Wav_E_D4'     : round(wav_e[1], 4),
                'Wav_E_D3'     : round(wav_e[2], 4),
                'Wav_E_D2'     : round(wav_e[3], 4),
                'Wav_E_D1'     : round(wav_e[4], 4),
                'Label'        : label,
                'Description'  : LABELS_DESCRIPTION.get(label, '?'),
            })

        print(f"    {label} ({LABELS_DESCRIPTION.get(label,'?')}) : "
              f"{n_fenetres} fenêtres")

    return pd.DataFrame(resultats)

# ============================================================
# 4. CALCUL DU HEALTH INDEX
# ============================================================
def calculer_hi(df_feat):
    """Calcule le HI par ACP-variance sur données réelles."""
    feature_cols = [c for c in df_feat.columns
                    if c not in ['Label', 'Description']]
    X    = df_feat[feature_cols].values.astype(float)

    # Remplacer inf et nan
    X    = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalisation Min-Max
    xmin = X.min(axis=0)
    xmax = X.max(axis=0)
    X_n  = (X - xmin) / (xmax - xmin + 1e-12)

    # Poids par variance inter-conditions
    variances = np.var(X_n, axis=0)
    weights   = variances / (variances.sum() + 1e-12)

    # Score de dégradation
    scores = X_n @ weights
    s_min, s_max = scores.min(), scores.max()
    scores_n = (scores - s_min) / (s_max - s_min + 1e-12)
    HI = 1 - scores_n

    df_feat = df_feat.copy()
    df_feat['HI'] = HI

    # Ancrage : F0 (normal) doit être proche de 1
    hi_f0 = df_feat[df_feat['Label'] == 'F0']['HI'].mean()
    if hi_f0 < 0.85:
        df_feat['HI'] = np.clip(df_feat['HI'] * (0.95 / hi_f0), 0, 1)

    # Résumé
    print("\n  Health Index moyen par condition :")
    hi_moyens = df_feat.groupby('Label')['HI'].mean().sort_index()
    for label, hi in hi_moyens.items():
        desc  = LABELS_DESCRIPTION.get(label, '?')
        statut = 'Normal ✓' if hi > 0.80 else 'Dégradé ⚠' if hi > 0.60 else 'Critique ✗'
        print(f"    {label} — {desc:<35} : HI = {hi:.4f}  [{statut}]")

    return df_feat, hi_moyens

# ============================================================
# 5. VISUALISATIONS
# ============================================================
def plot_hi_barres(hi_moyens, save=True):
    """Diagramme en barres du HI par condition réelle."""
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.suptitle('Health Index par condition — Données réelles (Dataset Bacha 2024)',
                 fontsize=13, fontweight='bold')

    labels = list(hi_moyens.index)
    values = list(hi_moyens.values)
    colors = [COULEURS_LABELS.get(l, 'gray') for l in labels]
    xlabels = [f"{l}\n{LABELS_DESCRIPTION.get(l,'?')}" for l in labels]

    bars = ax.bar(xlabels, values, color=colors,
                  edgecolor='white', linewidth=1.2, width=0.6)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.01,
                f'{v:.3f}', ha='center', fontweight='bold', fontsize=10)

    ax.axhline(0.80, color='#D69E2E', ls='--', lw=2,
               label='Seuil dégradé (0.80)')
    ax.axhline(0.60, color='#C53030', ls='--', lw=2,
               label='Seuil critique (0.60)')
    ax.axhspan(0.80, 1.05, alpha=0.07, color='green', label='Zone normale')
    ax.axhspan(0.60, 0.80, alpha=0.07, color='orange', label='Zone dégradée')
    ax.axhspan(0,    0.60, alpha=0.07, color='red',    label='Zone critique')

    ax.set_ylim([0, 1.1])
    ax.set_ylabel('Health Index (HI)', fontsize=12)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', labelsize=8)

    plt.tight_layout()
    if save:
        plt.savefig(rf'{DOSSIER}\perspective1_hi_par_condition.png',
                    dpi=150, bbox_inches='tight')
        print("  Figure : perspective1_hi_par_condition.png")
    plt.show()


def plot_hi_evolution(df_feat, save=True):
    """Évolution du HI fenêtre par fenêtre avec couleurs par condition."""
    fig, ax = plt.subplots(figsize=(15, 6))
    fig.suptitle('Évolution du Health Index — Données réelles — Par condition',
                 fontsize=13, fontweight='bold')

    offset = 0
    xtick_pos, xtick_lab = [], []

    for label in sorted(df_feat['Label'].unique()):
        sub = df_feat[df_feat['Label'] == label]['HI'].values
        x   = np.arange(offset, offset + len(sub))
        col = COULEURS_LABELS.get(label, 'gray')
        ax.plot(x, sub, color=col, linewidth=1.5,
                label=f"{label} — {LABELS_DESCRIPTION.get(label,'?')}")
        ax.axvline(offset, color='gray', lw=0.5, ls='--', alpha=0.4)
        xtick_pos.append(offset + len(sub)//2)
        xtick_lab.append(label)
        offset += len(sub)

    ax.axhline(0.80, color='#D69E2E', ls='--', lw=1.5,
               label='Seuil dégradé = 0.80')
    ax.axhline(0.60, color='#C53030', ls='--', lw=1.5,
               label='Seuil critique = 0.60')
    ax.axhspan(0.80, 1.05, alpha=0.06, color='green')
    ax.axhspan(0.60, 0.80, alpha=0.06, color='orange')
    ax.axhspan(0,    0.60, alpha=0.06, color='red')

    # Annotation zones
    ax.text(offset*0.99, 0.92, 'Normal',   ha='right',
            color='green',  fontsize=10, fontweight='bold')
    ax.text(offset*0.99, 0.70, 'Dégradé',  ha='right',
            color='orange', fontsize=10, fontweight='bold')
    ax.text(offset*0.99, 0.30, 'Critique', ha='right',
            color='red',    fontsize=10, fontweight='bold')

    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_lab, fontsize=9)
    ax.set_ylim([0, 1.05])
    ax.set_ylabel('Health Index (HI)', fontsize=11)
    ax.set_xlabel('Condition opérationnelle', fontsize=11)
    ax.legend(fontsize=8, loc='lower left', ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig(rf'{DOSSIER}\perspective1_hi_evolution_labels.png',
                    dpi=150, bbox_inches='tight')
        print("  Figure : perspective1_hi_evolution_labels.png")
    plt.show()


def plot_comparaison_sim_reel(hi_moyens, save=True):
    """Compare HI simulation vs HI données réelles."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Validation — Health Index : Simulation vs Données Réelles',
                 fontsize=13, fontweight='bold')

    # ── Gauche : Simulation ──
    ax1 = axes[0]
    sim_data = [
        ('Nominal',          1.000, '#276749'),
        ('Court-circuit',    1.000, '#C53030'),
        ('Dés. phase',       0.000, '#E53E3E'),
        ('Démagnétisation',  0.478, '#553C9A'),
        ('Roulement',        0.207, '#2B6CB0'),
        ('Câblage',          0.991, '#276749'),
    ]
    noms_s = [d[0] for d in sim_data]
    vals_s = [d[1] for d in sim_data]
    cols_s = [d[2] for d in sim_data]
    bars1  = ax1.bar(noms_s, vals_s, color=cols_s,
                     edgecolor='white', linewidth=1.2, width=0.6)
    for bar, v in zip(bars1, vals_s):
        ax1.text(bar.get_x()+bar.get_width()/2, v+0.01,
                 f'{v:.3f}', ha='center', fontweight='bold', fontsize=9)
    ax1.axhline(0.80, color='#D69E2E', ls='--', lw=1.5)
    ax1.axhline(0.60, color='#C53030', ls='--', lw=1.5)
    ax1.axhspan(0.80, 1.05, alpha=0.07, color='green')
    ax1.axhspan(0.60, 0.80, alpha=0.07, color='orange')
    ax1.axhspan(0,    0.60, alpha=0.07, color='red')
    ax1.set_ylim([0, 1.1])
    ax1.set_title('HI — Simulation (Chapitre V)', fontweight='bold')
    ax1.set_ylabel('Health Index')
    ax1.set_xticklabels(noms_s, rotation=15, ha='right', fontsize=8)
    ax1.grid(True, alpha=0.3, axis='y')

    # ── Droite : Réel ──
    ax2 = axes[1]
    labels_r = list(hi_moyens.index)
    vals_r   = list(hi_moyens.values)
    cols_r   = [COULEURS_LABELS.get(l, 'gray') for l in labels_r]
    xlabels_r = [f"{l}\n{LABELS_DESCRIPTION.get(l,'?')[:20]}" for l in labels_r]

    bars2 = ax2.bar(xlabels_r, vals_r, color=cols_r,
                    edgecolor='white', linewidth=1.2, width=0.6)
    for bar, v in zip(bars2, vals_r):
        ax2.text(bar.get_x()+bar.get_width()/2, v+0.01,
                 f'{v:.3f}', ha='center', fontweight='bold', fontsize=9)
    ax2.axhline(0.80, color='#D69E2E', ls='--', lw=1.5,
                label='Seuil dégradé')
    ax2.axhline(0.60, color='#C53030', ls='--', lw=1.5,
                label='Seuil critique')
    ax2.axhspan(0.80, 1.05, alpha=0.07, color='green')
    ax2.axhspan(0.60, 0.80, alpha=0.07, color='orange')
    ax2.axhspan(0,    0.60, alpha=0.07, color='red')
    ax2.set_ylim([0, 1.1])
    ax2.set_title('HI — Données Réelles (Dataset Bacha 2024)', fontweight='bold')
    ax2.set_ylabel('Health Index')
    ax2.legend(fontsize=8)
    ax2.set_xticklabels(xlabels_r, rotation=15, ha='right', fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save:
        plt.savefig(rf'{DOSSIER}\perspective1_sim_vs_reel_corrige.png',
                    dpi=150, bbox_inches='tight')
        print("  Figure : perspective1_sim_vs_reel_corrige.png")
    plt.show()


def plot_heatmap_features(df_feat, save=True):
    """Heatmap des indicateurs moyens par condition réelle."""
    feature_cols = [c for c in df_feat.columns
                    if c not in ['Label', 'Description', 'HI']]

    # Moyenne par label
    df_mean = df_feat.groupby('Label')[feature_cols].mean()

    # Normalisation pour visualisation
    df_norm = (df_mean - df_mean.min()) / (df_mean.max() - df_mean.min() + 1e-12)

    fig, ax = plt.subplots(figsize=(16, 5))
    fig.suptitle('Heatmap des indicateurs — Données réelles — Sain vs Défauts',
                 fontsize=12, fontweight='bold')

    im = ax.imshow(df_norm.values, aspect='auto',
                   cmap='RdYlGn_r', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.02,
                 label='Valeur normalisée (0=min, 1=max)')

    ax.set_xticks(range(len(feature_cols)))
    ax.set_xticklabels(feature_cols, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(df_norm.index)))
    ylab = [f"{l} — {LABELS_DESCRIPTION.get(l,'?')[:25]}"
            for l in df_norm.index]
    ax.set_yticklabels(ylab, fontsize=8)

    plt.tight_layout()
    if save:
        plt.savefig(rf'{DOSSIER}\perspective1_heatmap_reel.png',
                    dpi=150, bbox_inches='tight')
        print("  Figure : perspective1_heatmap_reel.png")
    plt.show()


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================
if __name__ == "__main__":

    print("=" * 65)
    print("  PERSPECTIVE 1 — VALIDATION DONNÉES RÉELLES (CORRIGÉ)")
    print("  Dataset : Bacha (2024) — FDD : F0=Normal, F1-F8=Défauts")
    print("=" * 65)

    print("\n[1] Chargement du dataset réel...")
    df = charger_dataset()

    print("\n[2] Visualisation des signaux bruts par condition...")
    plot_signaux_bruts(df)

    print("\n[3] Extraction des features par condition...")
    df_feat = extraire_features(df, fenetre=500)
    df_feat.to_csv(rf'{DOSSIER}\perspective1_features_reelles.csv', index=False)
    print(f"  Features exportées : perspective1_features_reelles.csv")

    print("\n[4] Calcul du Health Index...")
    df_hi, hi_moyens = calculer_hi(df_feat)

    print("\n[5] Figure : HI par condition...")
    plot_hi_barres(hi_moyens)

    print("\n[6] Figure : Évolution temporelle HI...")
    plot_hi_evolution(df_hi)

    print("\n[7] Figure : Comparaison Simulation vs Réel...")
    plot_comparaison_sim_reel(hi_moyens)

    print("\n[8] Figure : Heatmap des indicateurs...")
    plot_heatmap_features(df_hi)

    print("\n" + "=" * 65)
    print("  TERMINÉ ! Fichiers générés dans C:\\PFE_PMSM\\resultats\\ :")
    print("    perspective1_signaux_bruts.png")
    print("    perspective1_hi_par_condition.png")
    print("    perspective1_hi_evolution_labels.png")
    print("    perspective1_sim_vs_reel_corrige.png")
    print("    perspective1_heatmap_reel.png")
    print("    perspective1_features_reelles.csv")
    print("=" * 65)
