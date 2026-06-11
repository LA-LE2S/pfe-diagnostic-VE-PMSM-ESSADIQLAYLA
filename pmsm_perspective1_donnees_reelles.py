"""
=============================================================
  PERSPECTIVE 1 — VALIDATION SUR DONNÉES RÉELLES
  Application de la méthode HI sur dataset public PMSM
  Dataset : Bacha (2024) — Zenodo : 10.5281/zenodo.13974503
  Auteur : [Votre Nom] | Date : 2025
=============================================================
  Étapes :
  1. Charger le dataset réel (CSV)
  2. Appliquer le même pipeline d'extraction de features
  3. Calculer le Health Index sur données réelles
  4. Comparer simulation vs réel
  5. Visualiser les résultats
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import MinMaxScaler
import warnings, os
warnings.filterwarnings('ignore')

DOSSIER = r'C:\PFE_PMSM\resultats'
os.makedirs(DOSSIER, exist_ok=True)

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
    print(f"  Labels     : {df['label'].unique() if 'label' in df.columns else 'N/A'}")
    return df


def adapter_colonnes(df):
    """
    Adapte les noms de colonnes selon le dataset trouvé.
    Retourne un dictionnaire standardisé.
    """
    # Renommage flexible selon le dataset
    col_map = {}
    cols = [c.lower() for c in df.columns]

    for c in df.columns:
        cl = c.lower()
        if 'ia' in cl or 'i_a' in cl or 'phase_a' in cl:
            col_map['ia'] = c
        elif 'ib' in cl or 'i_b' in cl or 'phase_b' in cl:
            col_map['ib'] = c
        elif 'speed' in cl or 'vitesse' in cl or 'rpm' in cl:
            col_map['speed'] = c
        elif 'label' in cl or 'fault' in cl or 'class' in cl or 'condition' in cl:
            col_map['label'] = c
        elif 'temp' in cl:
            col_map['temperature'] = c

    print(f"\n  Colonnes identifiées : {col_map}")
    return col_map


# ============================================================
# 2. EXTRACTION DES FEATURES SUR DONNÉES RÉELLES
# ============================================================

def extraire_features_reelles(df, col_map, fenetre=1000):
    """
    Extrait les mêmes 17 indicateurs que le Chapitre IV
    sur le dataset réel par fenêtres glissantes.

    fenetre : nombre d'échantillons par fenêtre d'analyse
    """
    signal_col = col_map.get('ia', df.columns[0])
    label_col  = col_map.get('label', None)

    resultats = []

    # Grouper par label si disponible
    if label_col:
        groupes = df.groupby(label_col)
    else:
        groupes = [('Unknown', df)]

    for label, groupe in groupes:
        sig = groupe[signal_col].values.astype(float)

        # Découper en fenêtres
        n_fenetres = len(sig) // fenetre
        for i in range(n_fenetres):
            s = sig[i*fenetre : (i+1)*fenetre]

            # ── Indicateurs temporels ──
            rms       = np.sqrt(np.mean(s**2))
            peak      = np.max(np.abs(s))
            crest     = peak / (rms + 1e-12)
            kurt      = kurtosis(s)
            skewness  = skew(s)
            variance  = np.var(s)
            p2p       = np.max(s) - np.min(s)

            # ── Indicateurs fréquentiels (FFT) ──
            N    = len(s)
            win  = np.hanning(N)
            fft  = np.fft.rfft(s * win)
            freq = np.fft.rfftfreq(N, d=1.0/10000)
            amp  = (2.0/N) * np.abs(fft)

            def e_bande(f1, f2):
                m = (freq >= f1) & (freq < f2)
                return np.sum(amp[m]**2)

            e_0_50    = e_bande(0,   50)
            e_50_100  = e_bande(50,  100)
            e_100_200 = e_bande(100, 200)
            e_200_500 = e_bande(200, 500)

            # THD
            idx_f = np.argmin(np.abs(freq - 50))
            a_f   = amp[idx_f]
            harm  = [amp[np.argmin(np.abs(freq - k*50))] for k in range(2, 8)]
            thd   = np.sqrt(sum(h**2 for h in harm)) / (a_f + 1e-12)

            # ── Indicateurs Wavelet ──
            try:
                import pywt
                coeffs = pywt.wavedec(s, 'db4', level=5)
                total  = sum(np.sum(c**2) for c in coeffs) + 1e-12
                wav_e  = [np.sum(c**2)/total for c in coeffs]
                while len(wav_e) < 5:
                    wav_e.append(0.0)
            except:
                wav_e = [0.0]*5

            resultats.append({
                'RMS'              : round(rms, 4),
                'Peak'             : round(peak, 4),
                'Crest_Factor'     : round(crest, 4),
                'Kurtosis'         : round(kurt, 4),
                'Skewness'         : round(skewness, 4),
                'Variance'         : round(variance, 4),
                'Peak2Peak'        : round(p2p, 4),
                'Energy_0_50Hz'    : round(e_0_50, 6),
                'Energy_50_100Hz'  : round(e_50_100, 6),
                'Energy_100_200Hz' : round(e_100_200, 6),
                'Energy_200_500Hz' : round(e_200_500, 6),
                'THD'              : round(thd, 4),
                'Wav_E_approx'     : round(wav_e[0], 4),
                'Wav_E_D5'         : round(wav_e[1], 4),
                'Wav_E_D4'         : round(wav_e[2], 4),
                'Wav_E_D3'         : round(wav_e[3], 4),
                'Wav_E_D2'         : round(wav_e[4], 4),
                'Label'            : str(label),
                'Fenetre'          : i,
            })

        print(f"    {label} : {n_fenetres} fenêtres extraites")

    df_feat = pd.DataFrame(resultats)
    print(f"\n  Total features extraites : {df_feat.shape}")
    return df_feat


# ============================================================
# 3. CALCUL DU HEALTH INDEX SUR DONNÉES RÉELLES
# ============================================================

def calculer_hi_reel(df_feat_reel, df_feat_sim=None):
    """
    Calcule le Health Index sur les features réelles.
    Si df_feat_sim fourni : utilise les poids calculés en simulation.
    Sinon : recalcule les poids sur les données réelles.
    """
    feature_cols = [c for c in df_feat_reel.columns
                    if c not in ['Label', 'Fenetre']]

    # Normalisation
    scaler = MinMaxScaler()
    X_reel = df_feat_reel[feature_cols].values.astype(float)
    X_norm = scaler.fit_transform(X_reel)

    # Poids
    if df_feat_sim is not None:
        # Utiliser les poids de simulation (transfert de connaissances)
        X_sim = df_feat_sim[[c for c in feature_cols
                              if c in df_feat_sim.columns]].values.astype(float)
        X_sim_n = (X_sim - X_sim.min(0)) / (X_sim.max(0) - X_sim.min(0) + 1e-12)
        variances = np.var(X_sim_n, axis=0)
        print("  Poids issus de la simulation (transfert)")
    else:
        variances = np.var(X_norm, axis=0)
        print("  Poids calculés sur données réelles")

    weights = variances / (variances.sum() + 1e-12)

    # Health Index
    scores = X_norm @ weights
    s_min, s_max = scores.min(), scores.max()
    scores_norm  = (scores - s_min) / (s_max - s_min + 1e-12)
    HI = 1 - scores_norm

    df_feat_reel = df_feat_reel.copy()
    df_feat_reel['HI'] = HI

    # Moyennes par label
    hi_moyens = df_feat_reel.groupby('Label')['HI'].mean()
    print("\n  Health Index moyen par condition réelle :")
    for label, hi in hi_moyens.items():
        print(f"    {label:<30} : HI = {hi:.4f}")

    return df_feat_reel, hi_moyens


# ============================================================
# 4. VISUALISATION — COMPARAISON SIMULATION vs RÉEL
# ============================================================

def plot_comparaison_sim_reel(hi_sim_dict, hi_reel_moyens, save=True):
    """
    Compare les HI obtenus par simulation et par données réelles.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Validation : Health Index Simulation vs Données Réelles',
                 fontsize=13, fontweight='bold')

    # Graphe 1 : HI simulation (rappel)
    ax1 = axes[0]
    noms_sim = list(hi_sim_dict.keys())
    vals_sim = list(hi_sim_dict.values())
    cols_sim = ['#276749' if h > 0.8 else '#D69E2E' if h > 0.6
                else '#C53030' for h in vals_sim]
    bars = ax1.bar(noms_sim, vals_sim, color=cols_sim,
                   edgecolor='white', linewidth=1.2)
    for bar, h in zip(bars, vals_sim):
        ax1.text(bar.get_x()+bar.get_width()/2, h+0.01,
                 f'{h:.3f}', ha='center', fontweight='bold', fontsize=9)
    ax1.axhline(0.85, color='#D69E2E', ls='--', lw=1.5, label='Seuil dégradé')
    ax1.axhline(0.65, color='#C53030', ls='--', lw=1.5, label='Seuil critique')
    ax1.set_ylim([0, 1.1]); ax1.set_title('HI — Simulation (Ch.V)')
    ax1.set_ylabel('Health Index'); ax1.legend(fontsize=8)
    ax1.set_xticklabels(noms_sim, rotation=20, ha='right', fontsize=8)
    ax1.grid(True, alpha=0.3, axis='y')

    # Graphe 2 : HI données réelles
    ax2 = axes[1]
    noms_reel = list(hi_reel_moyens.index)
    vals_reel = list(hi_reel_moyens.values)
    cols_reel = ['#276749' if h > 0.8 else '#D69E2E' if h > 0.6
                 else '#C53030' for h in vals_reel]
    bars2 = ax2.bar(noms_reel, vals_reel, color=cols_reel,
                    edgecolor='white', linewidth=1.2)
    for bar, h in zip(bars2, vals_reel):
        ax2.text(bar.get_x()+bar.get_width()/2, h+0.01,
                 f'{h:.3f}', ha='center', fontweight='bold', fontsize=9)
    ax2.axhline(0.85, color='#D69E2E', ls='--', lw=1.5, label='Seuil dégradé')
    ax2.axhline(0.65, color='#C53030', ls='--', lw=1.5, label='Seuil critique')
    ax2.set_ylim([0, 1.1]); ax2.set_title('HI — Données Réelles (Dataset Bacha 2024)')
    ax2.set_ylabel('Health Index'); ax2.legend(fontsize=8)
    ax2.set_xticklabels(noms_reel, rotation=20, ha='right', fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save:
        plt.savefig(rf'{DOSSIER}\perspective1_sim_vs_reel.png',
                    dpi=150, bbox_inches='tight')
        print(f"\n  Figure sauvegardée : perspective1_sim_vs_reel.png")
    plt.show()


def plot_hi_evolution_reel(df_feat_reel, save=True):
    """
    Trace l'évolution du HI fenêtre par fenêtre pour chaque condition.
    """
    labels_uniques = df_feat_reel['Label'].unique()
    couleurs = plt.cm.tab10(np.linspace(0, 1, len(labels_uniques)))

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle('Évolution du Health Index — Données réelles (par fenêtre)',
                 fontsize=13, fontweight='bold')

    offset = 0
    xticks_pos, xticks_lab = [], []

    for label, col in zip(labels_uniques, couleurs):
        sub = df_feat_reel[df_feat_reel['Label'] == label]
        x   = np.arange(offset, offset + len(sub))
        ax.plot(x, sub['HI'].values, color=col, linewidth=1.5,
                label=label, alpha=0.85)
        ax.axvline(offset, color='gray', lw=0.5, ls='--', alpha=0.5)
        xticks_pos.append(offset + len(sub)//2)
        xticks_lab.append(label)
        offset += len(sub)

    ax.axhline(0.85, color='#D69E2E', ls='--', lw=1.5,
               label='Seuil dégradé = 0.85')
    ax.axhline(0.65, color='#C53030', ls='--', lw=1.5,
               label='Seuil critique = 0.65')
    ax.axhspan(0.85, 1.05, alpha=0.05, color='green')
    ax.axhspan(0.65, 0.85, alpha=0.05, color='orange')
    ax.axhspan(0,   0.65, alpha=0.05, color='red')

    ax.set_xticks(xticks_pos)
    ax.set_xticklabels(xticks_lab, rotation=20, ha='right', fontsize=8)
    ax.set_ylim([0, 1.05])
    ax.set_ylabel('Health Index (HI)', fontsize=11)
    ax.set_xlabel('Condition opérationnelle', fontsize=11)
    ax.legend(fontsize=8, loc='lower left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig(rf'{DOSSIER}\perspective1_hi_reel_evolution.png',
                    dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 5. PROGRAMME PRINCIPAL
# ============================================================

if __name__ == "__main__":

    print("=" * 65)
    print("  PERSPECTIVE 1 — VALIDATION SUR DONNÉES RÉELLES")
    print("=" * 65)

    # ── Chemin vers le dataset réel ──
    # Téléchargez depuis : https://doi.org/10.5281/zenodo.13974503
    # Puis modifiez ce chemin :
    CSV_REEL = r'C:\PFE_PMSM\dataset_reel\converted_dataset.csv'

    if not os.path.exists(CSV_REEL):
        print(f"""
  DATASET RÉEL INTROUVABLE : {CSV_REEL}

  Pour télécharger le dataset :
  1. Allez sur : https://doi.org/10.5281/zenodo.13974503
  2. Téléchargez : converted_dataset.csv
  3. Placez-le dans : C:\\PFE_PMSM\\dataset_reel\\
  4. Relancez ce script

  Source : Bacha A. (2024) — PMSM Inverter Fault Diagnosis
  GitHub : https://github.com/bachaabdelkabir/PMSM-inverter-fault-diagnosis
        """)
        exit()

    # ── 1. Chargement ──
    print("\n[1] Chargement du dataset réel...")
    df_reel  = charger_dataset_reel(CSV_REEL)
    col_map  = adapter_colonnes(df_reel)

    # ── 2. Extraction features ──
    print("\n[2] Extraction des features sur données réelles...")
    df_feat_reel = extraire_features_reelles(df_reel, col_map, fenetre=500)

    # ── 3. Chargement features simulation (pour comparaison) ──
    print("\n[3] Chargement features simulation...")
    csv_sim = os.path.join(DOSSIER, 'ch4_feature_dataset.csv')
    df_feat_sim = pd.read_csv(csv_sim) if os.path.exists(csv_sim) else None

    # ── 4. Health Index réel ──
    print("\n[4] Calcul du Health Index sur données réelles...")
    df_hi_reel, hi_moyens = calculer_hi_reel(df_feat_reel, df_feat_sim)

    # ── 5. Comparaison sim vs réel ──
    print("\n[5] Comparaison Simulation vs Réel...")
    hi_sim = {
        'Nominal'         : 1.000,
        'short_circuit'   : 1.000,
        'phase_imbalance' : 0.000,
        'magnet_demag'    : 0.478,
        'bearing_fault'   : 0.207,
        'wiring_fault'    : 0.991,
    }
    plot_comparaison_sim_reel(hi_sim, hi_moyens)

    # ── 6. Évolution temporelle réelle ──
    print("\n[6] Évolution du HI sur données réelles...")
    plot_hi_evolution_reel(df_hi_reel)

    # ── 7. Export ──
    path_out = os.path.join(DOSSIER, 'perspective1_hi_reel.csv')
    df_hi_reel.to_csv(path_out, index=False)
    print(f"\n  Résultats exportés : {path_out}")

    print("\n" + "=" * 65)
    print("  PERSPECTIVE 1 TERMINÉE !")
    print("  Fichiers générés :")
    print("    perspective1_sim_vs_reel.png")
    print("    perspective1_hi_reel_evolution.png")
    print("    perspective1_hi_reel.csv")
    print("=" * 65)
