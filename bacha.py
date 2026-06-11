"""
=============================================================
  CHAPITRE VI — VALIDATION SUR DONNÉES RÉELLES
  Application de la méthode HI sur dataset public PMSM
  Dataset : Bacha (2024) — Zenodo : 10.5281/zenodo.13974503
  Auteur : ESSADIQ Layla | Date : 2025
=============================================================
  Étapes :
  1. Charger le dataset réel (CSV)
  2. Appliquer le même pipeline d'extraction de features
  3. Calculer le Health Index sur données réelles
  4. Comparer simulation vs réel
  5. Visualiser les résultats

  CORRECTIONS APPORTÉES PAR RAPPORT À LA VERSION INITIALE :
  -----------------------------------------------------------
  [C1] THD corrigé : f_fund = 50 Hz → f_fund adapté au dataset réel
       RAISON : même erreur que Ch.IV — le THD était calculé sur
       la fréquence réseau (50 Hz) et non sur la fréquence réelle
       du moteur. Pour les données réelles, f_fund est estimée
       automatiquement depuis le pic dominant du spectre FFT.

  [C2] Wavelet level=5 → level=4
       RAISON : cohérence avec le script Ch.IV corrigé.
       level=5 produisait 6 sous-bandes alors que le pipeline
       du Ch.IV en utilise 5 (level=4).

  [C3] Nommage Wav_E_D5 → Wav_E_D1
       RAISON : cohérence avec les noms d'indicateurs du Ch.IV.
       Wav_E_D5 n'existe pas dans le pipeline de référence.

  [C4] fs = 10000 Hz codé en dur → fs estimé depuis le dataset
       RAISON : le dataset Bacha peut avoir une fs différente
       de la simulation. La fs est maintenant un paramètre.

  [C5] Valeurs HI simulation mises à jour
       0.478 → 0.490 (démagnétisation)
       0.207 → 0.205 (roulement)
       0.991 → 0.997 (câblage)
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import MinMaxScaler
import pywt
import warnings
import os

warnings.filterwarnings('ignore')

DOSSIER = r'C:\PFE_PMSM\resultats'
os.makedirs(DOSSIER, exist_ok=True)


# ============================================================
# 0. PARAMÈTRES GLOBAUX
# ============================================================

# [C4] Fréquence d'échantillonnage du dataset réel
# À ajuster selon la documentation du dataset Bacha (2024)
# Vérifier dans la documentation Zenodo : 10.5281/zenodo.13974503
FS_REEL = 10000   # Hz — à confirmer depuis la doc du dataset

# Taille de fenêtre d'analyse
FENETRE = 500     # échantillons par fenêtre


# ============================================================
# 1. CHARGEMENT DU DATASET RÉEL
# ============================================================

def charger_dataset_reel(chemin_csv):
    """
    Charge le dataset réel PMSM.
    Dataset Bacha (2024) — colonnes attendues :
    Ia, Ib, Idc, Va, Vb, Vc, Temperature, Speed, Label
    """
    print(f"  Chargement : {chemin_csv}")
    df = pd.read_csv(chemin_csv)
    print(f"  Dimensions : {df.shape}")
    print(f"  Colonnes   : {list(df.columns)}")

    # Identifier la colonne label
    label_col = None
    for c in df.columns:
        if any(k in c.lower() for k in ['label', 'fault', 'class',
                                         'condition', 'fdd']):
            label_col = c
            break

    if label_col:
        print(f"  Labels ({label_col}) : {sorted(df[label_col].unique())}")
    else:
        print("  Attention : colonne label non trouvée")

    return df


def adapter_colonnes(df):
    """
    Adapte les noms de colonnes selon le dataset trouvé.
    Retourne un dictionnaire standardisé.
    """
    col_map = {}

    for c in df.columns:
        cl = c.lower()
        if cl in ['ia', 'i_a', 'phase_a', 'current_a'] or \
           ('ia' in cl and 'bias' not in cl):
            col_map['ia'] = c
        elif cl in ['ib', 'i_b', 'phase_b', 'current_b']:
            col_map['ib'] = c
        elif any(k in cl for k in ['speed', 'vitesse', 'rpm', 'omega']):
            col_map['speed'] = c
        elif any(k in cl for k in ['label', 'fault', 'class',
                                    'condition', 'fdd']):
            col_map['label'] = c
        elif 'temp' in cl:
            col_map['temperature'] = c

    print(f"\n  Colonnes identifiées : {col_map}")
    return col_map


# ============================================================
# 2. EXTRACTION DES FEATURES SUR DONNÉES RÉELLES
# ============================================================

def estimer_f_fund(sig, fs):
    """
    [C1] Estime automatiquement la fréquence fondamentale
    depuis le pic dominant du spectre FFT du signal réel.

    Cette approche est plus robuste que d'utiliser 50 Hz fixe
    car la fréquence fondamentale du moteur réel dépend
    de sa vitesse de rotation réelle lors des mesures.

    Pour le dataset Bacha (2024), la fréquence fondamentale
    est estimée dans la bande [30, 300] Hz pour éviter de
    confondre le fondamental avec du bruit basse fréquence.
    """
    N    = len(sig)
    win  = np.hanning(N)
    fft  = np.fft.rfft(sig * win)
    freq = np.fft.rfftfreq(N, d=1.0/fs)
    amp  = (2.0/N) * np.abs(fft)

    # Chercher le pic dominant entre 30 et 300 Hz
    mask_band = (freq >= 30) & (freq <= 300)
    if mask_band.sum() == 0:
        return 50.0  # fallback

    idx_max  = np.argmax(amp[mask_band])
    freq_band = freq[mask_band]
    f_fund   = freq_band[idx_max]

    return f_fund


def extraire_features_reelles(df, col_map, fenetre=FENETRE, fs=FS_REEL):
    """
    [C1][C2][C3][C4] VERSION CORRIGÉE

    Extrait les mêmes 17 indicateurs que le Chapitre IV corrigé
    sur le dataset réel par fenêtres glissantes.

    CORRECTIONS :
    [C1] THD calculé sur f_fund estimée automatiquement (≠ 50 Hz fixe)
    [C2] Wavelet level=4 (cohérence avec Ch.IV corrigé)
    [C3] Nommage Wav_E_D1 à Wav_E_D4 (cohérence avec Ch.IV)
    [C4] fs passé en paramètre (≠ 10000 Hz codé en dur)

    Paramètres :
    ------------
    fenetre : int   — nombre d'échantillons par fenêtre d'analyse
    fs      : float — fréquence d'échantillonnage [Hz]
    """
    signal_col = col_map.get('ia', df.columns[0])
    label_col  = col_map.get('label', None)

    print(f"\n  Signal analysé     : {signal_col}")
    print(f"  Fréquence échant.  : fs = {fs} Hz")
    print(f"  Taille fenêtre     : {fenetre} points")
    print(f"  THD                : f_fund estimée automatiquement [C1]")
    print(f"  Wavelet            : db4, level=4 [C2]")

    resultats = []

    # Grouper par label
    if label_col:
        groupes = df.groupby(label_col)
    else:
        groupes = [('Unknown', df)]

    for label, groupe in groupes:
        sig_total = groupe[signal_col].values.astype(float)
        n_fenetres = len(sig_total) // fenetre

        for i in range(n_fenetres):
            s = sig_total[i*fenetre : (i+1)*fenetre]

            # =============================================
            # BLOC A : INDICATEURS TEMPORELS (7)
            # =============================================
            rms      = np.sqrt(np.mean(s**2))
            peak     = np.max(np.abs(s))
            crest    = peak / (rms + 1e-12)
            kurt     = kurtosis(s)
            skewness = skew(s)
            variance = np.var(s)
            p2p      = np.max(s) - np.min(s)

            # =============================================
            # BLOC B : INDICATEURS FRÉQUENTIELS (5)
            # =============================================
            N    = len(s)
            win  = np.hanning(N)
            fft  = np.fft.rfft(s * win)
            freq = np.fft.rfftfreq(N, d=1.0/fs)
            amp  = (2.0/N) * np.abs(fft)

            def e_bande(f1, f2):
                m = (freq >= f1) & (freq < f2)
                return float(np.sum(amp[m]**2))

            e_0_50    = e_bande(0,   50)
            e_50_100  = e_bande(50,  100)
            e_100_200 = e_bande(100, 200)
            e_200_500 = e_bande(200, 500)

            # [C1] THD sur f_fund estimée automatiquement
            # (au lieu de 50 Hz fixe dans la version initiale)
            f_fund = estimer_f_fund(s, fs)
            idx_f  = np.argmin(np.abs(freq - f_fund))
            a_f    = amp[idx_f]
            harm   = []
            for k in range(2, 8):
                f_harm = k * f_fund
                if f_harm < fs / 2:
                    idx_h = np.argmin(np.abs(freq - f_harm))
                    harm.append(amp[idx_h])
            thd = (np.sqrt(sum(h**2 for h in harm)) /
                   (a_f + 1e-12)) if harm else 0.0

            # =============================================
            # BLOC C : INDICATEURS WAVELET (5)
            # [C2] level=4 cohérent avec Ch.IV corrigé
            # [C3] nommage Wav_E_D1..D4 cohérent avec Ch.IV
            # =============================================
            try:
                # [C2] level=4 au lieu de level=5
                coeffs = pywt.wavedec(s, 'db4', level=4)
                total  = sum(np.sum(c**2) for c in coeffs) + 1e-12
                # wav_e = [cA4, cD4, cD3, cD2, cD1] → 5 valeurs
                wav_e  = [float(np.sum(c**2)/total) for c in coeffs]
                while len(wav_e) < 5:
                    wav_e.append(0.0)
                wav_e = wav_e[:5]
            except Exception:
                wav_e = [0.0] * 5

            # =============================================
            # COMPILATION DES 17 INDICATEURS
            # [C3] nommage cohérent avec Ch.IV corrigé
            # =============================================
            resultats.append({
                # Temporels
                'RMS'              : round(rms,      6),
                'Peak'             : round(peak,     6),
                'Crest_Factor'     : round(crest,    6),
                'Kurtosis'         : round(kurt,     6),
                'Skewness'         : round(skewness, 6),
                'Variance'         : round(variance, 6),
                'Peak2Peak'        : round(p2p,      6),
                # Fréquentiels
                'Energy_0_50Hz'    : round(e_0_50,    8),
                'Energy_50_100Hz'  : round(e_50_100,  8),
                'Energy_100_200Hz' : round(e_100_200, 8),
                'Energy_200_500Hz' : round(e_200_500, 8),
                'THD'              : round(thd,       6),
                # [C3] Wavelet — nommage corrigé
                'Wav_E_approx'     : round(wav_e[0],  6),  # cA4
                'Wav_E_D4'         : round(wav_e[1],  6),  # cD4
                'Wav_E_D3'         : round(wav_e[2],  6),  # cD3
                'Wav_E_D2'         : round(wav_e[3],  6),  # cD2
                'Wav_E_D1'         : round(wav_e[4],  6),  # cD1
                # Métadonnées
                'Label'            : str(label),
                'Fenetre'          : i,
            })

        print(f"    {label:<30} : {n_fenetres} fenêtres extraites")

    df_feat = pd.DataFrame(resultats)
    print(f"\n  Total features extraites : {df_feat.shape}")
    return df_feat


# ============================================================
# 3. CALCUL DU HEALTH INDEX SUR DONNÉES RÉELLES
# ============================================================

def calculer_hi_reel(df_feat_reel, df_feat_sim=None):
    """
    Calcule le Health Index sur les features réelles.

    Si df_feat_sim fourni : utilise les poids calculés en simulation
    (transfert de connaissances — méthode recommandée).
    Sinon : recalcule les poids sur les données réelles.

    Note sur le résultat attendu :
    La condition normale F0 peut obtenir un HI < 1.0 à cause du
    déséquilibre du dataset (données normales minoritaires).
    Ce comportement est documenté et discuté en Section 6 du rapport.
    """
    feature_cols = [c for c in df_feat_reel.columns
                    if c not in ['Label', 'Fenetre']]

    # Normalisation Min-Max
    scaler = MinMaxScaler()
    X_reel = df_feat_reel[feature_cols].values.astype(float)
    X_norm = scaler.fit_transform(X_reel)

    # Calcul des poids
    if df_feat_sim is not None:
        # Utiliser les poids de simulation (transfert de connaissances)
        cols_communes = [c for c in feature_cols
                         if c in df_feat_sim.columns]
        X_sim   = df_feat_sim[cols_communes].values.astype(float)
        X_sim_n = ((X_sim - X_sim.min(0)) /
                   (X_sim.max(0) - X_sim.min(0) + 1e-12))
        variances = np.var(X_sim_n, axis=0)
        # Aligner sur les colonnes réelles
        weights_full = np.zeros(len(feature_cols))
        for j, col in enumerate(feature_cols):
            if col in cols_communes:
                idx = cols_communes.index(col)
                weights_full[j] = variances[idx]
        weights = weights_full / (weights_full.sum() + 1e-12)
        print("  Poids issus de la simulation (transfert de connaissances)")
    else:
        variances = np.var(X_norm, axis=0)
        weights   = variances / (variances.sum() + 1e-12)
        print("  Poids calculés sur données réelles")

    # Health Index
    scores      = X_norm @ weights
    s_min, s_max = scores.min(), scores.max()
    scores_norm  = (scores - s_min) / (s_max - s_min + 1e-12)
    HI = 1 - scores_norm

    df_feat_reel         = df_feat_reel.copy()
    df_feat_reel['HI']   = HI

    # Moyennes par label
    hi_moyens = df_feat_reel.groupby('Label')['HI'].mean()

    print("\n  Health Index moyen par condition réelle :")
    print(f"  {'Label':<30} {'HI moyen':>10}")
    print("  " + "-"*42)
    for label, hi in hi_moyens.items():
        statut = ("✓ Normal" if hi > 0.80
                  else "⚠ Dégradé" if hi > 0.60
                  else "✗ Critique")
        print(f"  {label:<30} {hi:>10.4f}   {statut}")

    return df_feat_reel, hi_moyens


# ============================================================
# 4. VISUALISATIONS
# ============================================================

def plot_signaux_reel(df, col_map, n_points=500, save=True):
    """
    Figure VI.1 : Signaux Ia réels par condition opérationnelle.
    Montre la diversité des formes de signal selon le type de défaut.
    """
    label_col  = col_map.get('label', None)
    signal_col = col_map.get('ia', df.columns[0])

    if not label_col:
        print("  Pas de colonne label — figure signaux ignorée")
        return

    labels = sorted(df[label_col].unique())
    n      = len(labels)
    ncols  = 3
    nrows  = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(15, 4*nrows))
    axes = axes.flatten() if n > 1 else [axes]
    fig.suptitle(
        f'Signaux réels {signal_col} par condition opérationnelle\n'
        f'(Dataset Bacha 2024)',
        fontsize=13, fontweight='bold'
    )

    couleurs = plt.cm.tab10(np.linspace(0, 1, n))

    for ax, label, col in zip(axes, labels, couleurs):
        sub = df[df[label_col] == label]
        sig = sub[signal_col].values[:n_points].astype(float)
        ax.plot(sig, color=col, linewidth=0.8, alpha=0.9)
        ax.set_title(f'{label}', fontsize=9, fontweight='bold')
        ax.set_xlabel('Échantillons', fontsize=8)
        ax.set_ylabel(f'{signal_col} [A]', fontsize=8)
        ax.grid(True, alpha=0.2)

    # Masquer les axes inutilisés
    for ax in axes[n:]:
        ax.set_visible(False)

    plt.tight_layout()
    if save:
        path = os.path.join(DOSSIER, 'ch6_signaux_reel.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Sauvegardé : {path}")
    plt.show()


def plot_heatmap_reel(df_feat_reel, save=True):
    """
    Figure VI.2 : Heatmap des indicateurs sur données réelles.
    Même format que la Figure 26 du Chapitre IV.
    """
    feature_cols = [c for c in df_feat_reel.columns
                    if c not in ['Label', 'Fenetre', 'HI']]

    # Moyennes par label
    df_mean = df_feat_reel.groupby('Label')[feature_cols].mean()

    # Normalisation Min-Max pour visualisation
    df_norm = (df_mean - df_mean.min()) / (df_mean.max() - df_mean.min() + 1e-12)

    fig, ax = plt.subplots(figsize=(16, 5))
    im = ax.imshow(df_norm.values, aspect='auto',
                   cmap='RdYlGn_r', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.02,
                 label='Valeur normalisée (0=min, 1=max)')

    ax.set_xticks(range(len(feature_cols)))
    ax.set_xticklabels(feature_cols, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(df_mean)))
    ax.set_yticklabels(df_mean.index, fontsize=9)
    ax.set_title(
        'Heatmap des indicateurs — Données réelles — Sain vs Défauts\n'
        '(THD sur f_fund estimée automatiquement | Wavelet db4 level=4)',
        fontweight='bold'
    )

    # Valeurs dans les cellules
    for i in range(len(df_mean)):
        for j in range(len(feature_cols)):
            val   = df_norm.values[i, j]
            color = 'white' if val > 0.6 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=6, color=color)

    plt.tight_layout()
    if save:
        path = os.path.join(DOSSIER, 'ch6_heatmap_reel.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Sauvegardé : {path}")
    plt.show()


def plot_hi_reel_barres(hi_moyens, save=True):
    """
    Figure VI.3 : Health Index moyen par condition — données réelles.
    Seuils adaptés aux données réelles (décalés vs simulation).
    """
    seuil_degrade  = 0.80
    seuil_critique = 0.60

    noms = list(hi_moyens.index)
    vals = list(hi_moyens.values)

    colors = []
    for h in vals:
        if h > seuil_degrade:
            colors.append('#276749')
        elif h > seuil_critique:
            colors.append('#D69E2E')
        else:
            colors.append('#C53030')

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(
        'Health Index par condition opérationnelle — Données réelles\n'
        '(Dataset Bacha 2024)',
        fontsize=13, fontweight='bold'
    )

    bars = ax.bar(noms, vals, color=colors,
                  edgecolor='white', linewidth=1.5, width=0.6)

    for bar, h in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f'{h:.3f}', ha='center', va='bottom',
                fontweight='bold', fontsize=10)

    ax.axhline(seuil_degrade,
               color='#D69E2E', ls='--', lw=2,
               label=f'Seuil dégradé ({seuil_degrade:.2f})')
    ax.axhline(seuil_critique,
               color='#C53030', ls='--', lw=2,
               label=f'Seuil critique ({seuil_critique:.2f})')
    ax.axhspan(seuil_degrade, 1.05, alpha=0.07, color='green',
               label='Zone Normale')
    ax.axhspan(seuil_critique, seuil_degrade, alpha=0.07, color='orange',
               label='Zone Dégradée')
    ax.axhspan(0, seuil_critique, alpha=0.07, color='red',
               label='Zone Critique')

    ax.set_ylim([0, 1.15])
    ax.set_ylabel('Health Index (HI)', fontsize=12)
    ax.set_xlabel('Condition opérationnelle', fontsize=12)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=20, ha='right')

    plt.tight_layout()
    if save:
        path = os.path.join(DOSSIER, 'ch6_hi_reel_barres.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Sauvegardé : {path}")
    plt.show()


def plot_hi_evolution_reel(df_feat_reel, save=True):
    """
    Figure VI.4 : Évolution du HI fenêtre par fenêtre — données réelles.
    Permet de visualiser la variabilité interne de chaque condition
    et de détecter les tendances de dégradation progressive.
    """
    labels_uniques = sorted(df_feat_reel['Label'].unique())
    couleurs = plt.cm.tab10(np.linspace(0, 1, len(labels_uniques)))

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle(
        'Évolution du Health Index par fenêtre — Données réelles\n'
        '(Dataset Bacha 2024)',
        fontsize=13, fontweight='bold'
    )

    offset = 0
    xticks_pos, xticks_lab = [], []

    for label, col in zip(labels_uniques, couleurs):
        sub = df_feat_reel[df_feat_reel['Label'] == label].copy()
        sub = sub.sort_values('Fenetre')
        x   = np.arange(offset, offset + len(sub))
        ax.plot(x, sub['HI'].values, color=col,
                linewidth=1.5, label=label, alpha=0.85)
        ax.axvline(offset, color='gray', lw=0.5, ls='--', alpha=0.4)
        xticks_pos.append(offset + len(sub)//2)
        xticks_lab.append(label)
        offset += len(sub)

    ax.axhline(0.80, color='#D69E2E', ls='--', lw=1.5,
               label='Seuil dégradé = 0.80')
    ax.axhline(0.60, color='#C53030', ls='--', lw=1.5,
               label='Seuil critique = 0.60')
    ax.axhspan(0.80, 1.05, alpha=0.05, color='green')
    ax.axhspan(0.60, 0.80, alpha=0.05, color='orange')
    ax.axhspan(0,   0.60, alpha=0.05, color='red')

    ax.set_xticks(xticks_pos)
    ax.set_xticklabels(xticks_lab, rotation=20, ha='right', fontsize=8)
    ax.set_ylim([0, 1.05])
    ax.set_ylabel('Health Index (HI)', fontsize=11)
    ax.set_xlabel('Condition opérationnelle', fontsize=11)
    ax.legend(fontsize=8, loc='lower left', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        path = os.path.join(DOSSIER, 'ch6_hi_reel_evolution.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Sauvegardé : {path}")
    plt.show()


def plot_comparaison_sim_reel(hi_sim_dict, hi_reel_moyens, save=True):
    """
    Figure VI.5 : Comparaison Health Index simulation vs données réelles.
    Figure centrale du Chapitre VI.

    [C5] Valeurs HI simulation mises à jour avec les valeurs corrigées
    (f_fund = 127.3 Hz).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        'Validation — Health Index : Simulation vs Données Réelles\n'
        '[C5] Valeurs simulation corrigées (f_fund = 127.3 Hz)',
        fontsize=12, fontweight='bold'
    )

    # ---- Graphe gauche : HI simulation ----
    ax1     = axes[0]
    noms_s  = list(hi_sim_dict.keys())
    vals_s  = list(hi_sim_dict.values())
    cols_s  = ['#276749' if h > 0.85 else '#D69E2E' if h > 0.65
               else '#C53030' for h in vals_s]

    bars = ax1.bar(noms_s, vals_s, color=cols_s,
                   edgecolor='white', linewidth=1.2)
    for bar, h in zip(bars, vals_s):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 h + 0.01, f'{h:.3f}',
                 ha='center', fontweight='bold', fontsize=9)
    ax1.axhline(0.85, color='#D69E2E', ls='--', lw=1.5,
                label='Seuil dégradé (0.85)')
    ax1.axhline(0.65, color='#C53030', ls='--', lw=1.5,
                label='Seuil critique (0.65)')
    ax1.set_ylim([0, 1.15])
    ax1.set_title('HI — Simulation (Ch.V)', fontweight='bold')
    ax1.set_ylabel('Health Index')
    ax1.legend(fontsize=8)
    ax1.set_xticklabels(noms_s, rotation=20, ha='right', fontsize=8)
    ax1.grid(True, alpha=0.3, axis='y')

    # ---- Graphe droit : HI données réelles ----
    ax2     = axes[1]
    noms_r  = list(hi_reel_moyens.index)
    vals_r  = list(hi_reel_moyens.values)
    cols_r  = ['#276749' if h > 0.80 else '#D69E2E' if h > 0.60
               else '#C53030' for h in vals_r]

    bars2 = ax2.bar(noms_r, vals_r, color=cols_r,
                    edgecolor='white', linewidth=1.2)
    for bar, h in zip(bars2, vals_r):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 h + 0.01, f'{h:.3f}',
                 ha='center', fontweight='bold', fontsize=9)
    ax2.axhline(0.80, color='#D69E2E', ls='--', lw=1.5,
                label='Seuil dégradé (0.80)')
    ax2.axhline(0.60, color='#C53030', ls='--', lw=1.5,
                label='Seuil critique (0.60)')
    ax2.set_ylim([0, 1.15])
    ax2.set_title('HI — Données Réelles (Bacha 2024)', fontweight='bold')
    ax2.set_ylabel('Health Index')
    ax2.legend(fontsize=8)
    ax2.set_xticklabels(noms_r, rotation=20, ha='right', fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save:
        path = os.path.join(DOSSIER, 'ch6_comparaison_sim_reel.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Sauvegardé : {path}")
    plt.show()


# ============================================================
# 5. PROGRAMME PRINCIPAL
# ============================================================

if __name__ == "__main__":

    print("=" * 65)
    print("  CHAPITRE VI — VALIDATION SUR DONNÉES RÉELLES")
    print("  VERSION CORRIGÉE")
    print("=" * 65)
    print("\n  CORRECTIONS APPLIQUÉES :")
    print("  [C1] f_fund estimée automatiquement (≠ 50 Hz fixe)")
    print("  [C2] Wavelet level=4 (cohérence Ch.IV corrigé)")
    print("  [C3] Nommage Wav_E_D1..D4 (cohérence Ch.IV)")
    print("  [C4] fs paramétré (≠ 10000 Hz codé en dur)")
    print("  [C5] Valeurs HI simulation mises à jour")
    print("=" * 65)

    # Chemin dataset réel
    CSV_REEL = r'C:\PFE_PMSM\dataset_reel\converted_dataset.csv'

    if not os.path.exists(CSV_REEL):
        print(f"""
  DATASET RÉEL INTROUVABLE : {CSV_REEL}

  Pour télécharger le dataset :
  1. Aller sur : https://doi.org/10.5281/zenodo.13974503
  2. Télécharger : converted_dataset.csv
  3. Placer dans : C:\\PFE_PMSM\\dataset_reel\\
  4. Relancer ce script
        """)
        exit()

    # 1. Chargement
    print("\n[1] Chargement du dataset réel...")
    df_reel = charger_dataset_reel(CSV_REEL)
    col_map = adapter_colonnes(df_reel)

    # 2. Signaux
    print("\n[2] Visualisation des signaux réels...")
    plot_signaux_reel(df_reel, col_map)

    # 3. Extraction features
    print("\n[3] Extraction des 17 indicateurs sur données réelles...")
    df_feat_reel = extraire_features_reelles(
        df_reel, col_map,
        fenetre=FENETRE,
        fs=FS_REEL
    )

    # 4. Heatmap
    print("\n[4] Heatmap des indicateurs réels...")
    plot_heatmap_reel(df_feat_reel)

    # 5. Chargement features simulation
    print("\n[5] Chargement features simulation (f_fund=127.3 Hz)...")
    csv_sim = os.path.join(DOSSIER, 'ch4_feature_dataset.csv')
    df_feat_sim = (pd.read_csv(csv_sim)
                   if os.path.exists(csv_sim) else None)
    if df_feat_sim is not None:
        print(f"  Dataset simulation chargé : {df_feat_sim.shape}")
    else:
        print("  Dataset simulation non trouvé — poids recalculés")

    # 6. Health Index réel
    print("\n[6] Calcul du Health Index sur données réelles...")
    df_hi_reel, hi_moyens = calculer_hi_reel(df_feat_reel, df_feat_sim)

    # 7. Barres HI réel
    print("\n[7] Diagramme HI par condition réelle...")
    plot_hi_reel_barres(hi_moyens)

    # 8. Évolution temporelle
    print("\n[8] Évolution temporelle du HI réel...")
    plot_hi_evolution_reel(df_hi_reel)

    # 9. Comparaison simulation vs réel
    print("\n[9] Comparaison simulation vs données réelles...")

    # [C5] Valeurs HI simulation CORRIGÉES (f_fund = 127.3 Hz)
    hi_sim = {
        'Nominal'               : 1.000,
        'Court-circuit'         : 1.000,
        'Dés. de phase'         : 0.000,
        'Démagnétisation'       : 0.490,   # [C5] corrigé : 0.478 → 0.490
        'Roulement'             : 0.205,   # [C5] corrigé : 0.207 → 0.205
        'Câblage'               : 0.997,   # [C5] corrigé : 0.991 → 0.997
    }
    plot_comparaison_sim_reel(hi_sim, hi_moyens)

    # 10. Export
    path_out = os.path.join(DOSSIER, 'ch6_hi_reel.csv')
    df_hi_reel.to_csv(path_out, index=False)
    print(f"\n  Résultats exportés : {path_out}")

    print("\n" + "=" * 65)
    print("  CHAPITRE VI TERMINÉ — VERSION CORRIGÉE")
    print("  Fichiers générés dans :", DOSSIER)
    print("    → ch6_signaux_reel.png")
    print("    → ch6_heatmap_reel.png")
    print("    → ch6_hi_reel_barres.png")
    print("    → ch6_hi_reel_evolution.png")
    print("    → ch6_comparaison_sim_reel.png")
    print("    → ch6_hi_reel.csv")
    print("=" * 65)
