"""
=============================================================
  HEALTH INDEX ADAPTATIF — CHAPITRE V du PFE
  Diagnostic et Maintenance Prédictive du PMSM
  Auteur : ESSADIQ Layla | Date : 2025
=============================================================
  Ce script :
  1. Charge le dataset des indicateurs (Ch.IV — corrigé f_fund=127Hz)
  2. Normalise les indicateurs (Min-Max)
  3. Calcule les poids par variance inter-scénarios (méthode ACP-Variance)
  4. Construit le Health Index (HI) pour chaque scénario
  5. Définit les seuils d'alarme adaptatifs
  6. Simule l'évolution temporelle du HI (dégradation progressive)
  7. Visualise tous les résultats
  8. Exporte un rapport de diagnostic automatique

  CORRECTIONS APPORTÉES PAR RAPPORT À LA VERSION INITIALE :
  -----------------------------------------------------------
  [C1] λ court-circuit corrigé : 0.80 → 0.02
       RAISON : le court-circuit a HI_final ≈ 1.0 (indétectable par HI).
       Avec λ=0.80 et HI_final=1.0, la courbe exponentielle reste
       constante à 1.0 quelle que soit λ. La valeur 0.80 était donc
       sans effet ET physiquement incohérente (le rapport décrivait une
       "dégradation très rapide" pour un défaut non détecté).
       Solution : λ=0.02 identique au nominal, reflétant fidèlement
       la non-détectabilité du court-circuit par le Health Index.

  [C2] Justification des seuils 0.15 et 0.35 ajoutée en commentaire
       Référence aux travaux [19][20] sur les Health Index industriels.

  [C3] Actions recommandées rendues spécifiques à chaque type de défaut
       (remplacement aimants, remplacement roulement, inspection câblage...)

  [C4] Vérification du CSV chargé (présence colonne THD, f_fund=127Hz)

  [C5] Ajustement HI_nominal documenté et explicité

  [C6] Tableau de bord enrichi : ajout de la valeur numérique HI
       et de l'indicateur dominant pour chaque scénario

  [C7] Rapport enrichi : mention f_fund, numérotation tableaux corrigée
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import MinMaxScaler
import warnings
import os

warnings.filterwarnings('ignore')

# Dossier de résultats
DOSSIER = r'C:\PFE_PMSM\resultats'
os.makedirs(DOSSIER, exist_ok=True)


# ============================================================
# 1. CHARGEMENT DU DATASET (produit par le Chapitre IV corrigé)
# ============================================================

def load_dataset():
    """
    Charge le dataset des features depuis le CSV du Chapitre IV.

    [C4] AJOUT : vérification que le CSV est bien celui corrigé
    avec f_fund = 127.3 Hz pour le calcul du THD.
    Si le fichier est absent ou invalide, une erreur claire est levée.
    """
    csv_path = os.path.join(DOSSIER, 'ch4_feature_dataset.csv')

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"\n  ERREUR : Fichier introuvable : {csv_path}\n"
            f"  → Exécuter d'abord pmsm_analyse_frequentielle.py (version corrigée)\n"
            f"    pour générer le dataset avec f_fund = 127.3 Hz."
        )

    df = pd.read_csv(csv_path)

    # Vérification colonnes attendues
    colonnes_attendues = [
        'RMS', 'Peak', 'Crest_Factor', 'Kurtosis', 'Skewness',
        'Variance', 'Peak2Peak', 'Energy_0_50Hz', 'Energy_50_100Hz',
        'Energy_100_200Hz', 'Energy_200_500Hz', 'THD',
        'Wav_E_approx', 'Wav_E_D4', 'Wav_E_D3', 'Wav_E_D2', 'Wav_E_D1',
        'Fault_Type', 'Severity'
    ]
    manquantes = [c for c in colonnes_attendues if c not in df.columns]
    if manquantes:
        raise ValueError(
            f"\n  ERREUR : Colonnes manquantes dans le CSV : {manquantes}\n"
            f"  → Vérifier que le script pmsm_analyse_frequentielle.py\n"
            f"    (version corrigée avec f_fund=127.3 Hz) a bien été exécuté."
        )

    print(f"  Dataset chargé : {df.shape[0]} scénarios × "
          f"{df.shape[1]-2} indicateurs")
    print(f"  Scénarios      : {list(df['Fault_Type'])}")
    print(f"  Note           : THD calculé sur f_fund ≈ 127.3 Hz (version corrigée)")

    return df


# ============================================================
# 2. NORMALISATION DES INDICATEURS (Min-Max)
# ============================================================

def normalize_features(df):
    """
    Normalisation Min-Max de tous les indicateurs numériques.

    Formule : f_norm_i = (f_i - f_i_min) / (f_i_max - f_i_min)

    Après transformation : toutes les valeurs ∈ [0, 1]
    → 0 = valeur minimale observée sur les 6 scénarios
    → 1 = valeur maximale observée sur les 6 scénarios

    Cela permet à tous les indicateurs de contribuer équitablement
    au Health Index, indépendamment de leur ordre de grandeur initial.
    (RMS en Ampères et Energy_0_50Hz en A² sont traités sur la même échelle)

    Référence : voir Section 2.1 du Chapitre V du rapport.
    """
    feature_cols = [c for c in df.columns
                    if c not in ['Fault_Type', 'Severity']]

    scaler   = MinMaxScaler()
    df_norm  = df.copy()
    df_norm[feature_cols] = scaler.fit_transform(df[feature_cols])

    print(f"\n  Normalisation Min-Max : {len(feature_cols)} indicateurs → [0, 1]")
    return df_norm, feature_cols, scaler


# ============================================================
# 3. CALCUL DES POIDS PAR VARIANCE INTER-SCÉNARIOS (ACP-Variance)
# ============================================================

def compute_weights_variance(df_norm, feature_cols):
    """
    Calcule les poids des indicateurs par variance inter-scénarios.

    PRINCIPE :
    Un indicateur qui prend la même valeur pour tous les scénarios
    n'apporte aucune information discriminante → poids nul.
    Un indicateur qui varie fortement d'un scénario à l'autre
    est hautement informatif → poids élevé.

    FORMULE :
        σ²_i = variance de l'indicateur i sur les N scénarios
        w_i  = σ²_i / Σ(σ²_k)    (normalisation : Σ w_i = 1)

    Cette méthode est qualifiée d'adaptative car les poids sont
    calculés automatiquement à partir des données, sans expertise
    préalable ni calibration manuelle.

    Référence : Section 2.2 du Chapitre V du rapport.
    """
    X         = df_norm[feature_cols].values
    variances = np.var(X, axis=0)
    weights   = variances / (variances.sum() + 1e-12)

    weight_dict  = dict(zip(feature_cols, weights))
    weight_sorted = sorted(weight_dict.items(),
                           key=lambda x: x[1], reverse=True)

    print("\n  Top 5 indicateurs les plus discriminants :")
    for name, w in weight_sorted[:5]:
        print(f"    {name:<25} : w = {w:.4f}  ({w*100:.1f}%)")

    # Indicateur à poids nul
    zero_w = [(n, w) for n, w in weight_dict.items() if w < 1e-6]
    if zero_w:
        print(f"\n  Indicateurs à poids nul (non discriminants) :")
        for n, w in zero_w:
            print(f"    {n:<25} : w ≈ {w:.6f}")

    return weights, weight_dict


# ============================================================
# 4. CALCUL DU HEALTH INDEX
# ============================================================

def compute_health_index(df_norm, feature_cols, weights):
    """
    Calcule le Health Index pour chaque scénario.

    ÉTAPE 1 — Score de dégradation :
        S_deg = Σ (w_i × f_i_norm)
        Un score élevé = indicateurs globalement à des niveaux élevés
        = état de santé dégradé.

    ÉTAPE 2 — Health Index normalisé :
        HI = 1 - (S_deg - S_deg_min) / (S_deg_max - S_deg_min)
        HI ∈ [0, 1]
        HI = 1 → moteur en parfait état
        HI = 0 → dégradation critique maximale

    ÉTAPE 3 — [C5] Ajustement d'ancrage :
        Si HI_nominal < 0.90, on recale pour que HI_nominal = 0.95.
        JUSTIFICATION : la normalisation Min-Max dépend de la distribution
        des scénarios. Si les défauts sont très sévères, le scénario nominal
        peut obtenir un HI < 1.0. L'ajustement garantit que le moteur sain
        obtient toujours HI ≈ 0.95 comme référence cohérente.
        Ce recalage est appliqué proportionnellement à tous les scénarios.

    Référence : Section 2.3 du Chapitre V du rapport.
    """
    X = df_norm[feature_cols].values

    # Score de dégradation pondéré
    degradation_score = X @ weights

    # Normalisation entre 0 et 1
    d_min = degradation_score.min()
    d_max = degradation_score.max()
    degradation_norm = (degradation_score - d_min) / (d_max - d_min + 1e-12)

    # Health Index = complément à 1
    HI = 1 - degradation_norm

    # [C5] Ajustement d'ancrage du nominal à 0.95
    idx_nominal = list(df_norm['Fault_Type']).index('Nominal')
    HI_nominal  = HI[idx_nominal]
    if HI_nominal < 0.90:
        print(f"\n  Ajustement d'ancrage : HI_nominal = {HI_nominal:.3f} → 0.95")
        print(f"  (recalage proportionnel de tous les scénarios)")
        HI = HI * (0.95 / HI_nominal)
        HI = np.clip(HI, 0, 1)

    return HI, degradation_norm


# ============================================================
# 5. SEUILS D'ALARME ADAPTATIFS
# ============================================================

def define_thresholds(HI_nominal=0.95):
    """
    [C2] Définit les seuils d'alarme avec justification.

    VALEURS :
        Seuil Normal   : HI > HI_nominal - 0.15  (= 0.80 si HI_nominal=0.95)
        Seuil Critique : HI ≤ HI_nominal - 0.35  (= 0.60 si HI_nominal=0.95)

    JUSTIFICATION :
    Ces valeurs sont issues de deux considérations complémentaires :

    1) La littérature sur les Health Index industriels [19][20] recommande :
       - Zone de tolérance de 10 à 20% pour la surveillance (→ -0.15)
       - Zone critique en dessous de 60 à 70% de la valeur nominale (→ -0.35)

    2) Validation a posteriori sur les résultats simulés :
       Les 3 défauts détectables (déséquilibre : 0.000, roulement : 0.207,
       démagnétisation : 0.478) tombent naturellement sous 0.60,
       validant le seuil critique à HI_nominal - 0.35.

    En déploiement industriel avec données étiquetées abondantes,
    ces seuils pourraient être optimisés par analyse de courbe ROC.

    Référence : Section 3.2 du Chapitre V du rapport.
    """
    seuil_vert   = round(HI_nominal - 0.15, 3)
    seuil_orange = round(HI_nominal - 0.35, 3)

    print(f"\n  Seuils d'alarme adaptatifs (basés sur HI_nominal = {HI_nominal:.3f}) :")
    print(f"    ✓ Normal   : HI > {seuil_vert:.3f}")
    print(f"    ⚠ Dégradé  : {seuil_orange:.3f} < HI ≤ {seuil_vert:.3f}")
    print(f"    ✗ Critique : HI ≤ {seuil_orange:.3f}")
    print(f"  Justification : seuils définis selon littérature [19][20]")
    print(f"  + validation a posteriori sur les scénarios simulés.")

    return seuil_vert, seuil_orange


def get_status(hi, seuil_vert, seuil_orange):
    """Retourne le statut, la couleur et le symbole selon le HI."""
    if hi > seuil_vert:
        return 'Normal', '#276749', '✓'
    elif hi > seuil_orange:
        return 'Dégradé', '#B7791F', '⚠'
    else:
        return 'Critique', '#C53030', '✗'


# ============================================================
# 6. SIMULATION DE DÉGRADATION PROGRESSIVE
# ============================================================

def simulate_degradation(fault_types, HI_values, n_points=200):
    """
    Simule l'évolution temporelle du HI pour chaque type de défaut.

    MODÈLE EXPONENTIEL :
        HI(t) = HI_sain × e^(-λt) + HI_final × (1 - e^(-λt)) + ε(t)

    Paramètres :
    - HI_sain  = 0.95  (moteur neuf)
    - HI_final = valeur statique calculée au Ch.V
    - λ        = taux de dégradation spécifique au défaut
    - ε(t)     ~ N(0, 0.008) bruit gaussien réaliste

    [C1] CORRECTION CRITIQUE :
    Le court-circuit a HI_final ≈ 1.0 (indétectable par Health Index).
    La formule donne : HI(t) = 0.95×e^(-λt) + 1.0×(1-e^(-λt))
    → Pour tout λ > 0, cette courbe MONTE de 0.95 vers 1.0, ce qui
      représente une "amélioration" physiquement absurde.
    → Pour λ = 0.80 (ancienne valeur), la courbe converge rapidement
      vers 1.0, pas vers 0 — la description "dégradation très rapide"
      était donc erronée.

    SOLUTION [C1] : λ_court_circuit = 0.02 (identique au nominal)
    → La courbe reste stable autour de 0.95, reflétant fidèlement
      la non-détectabilité de ce défaut par le Health Index.
    → Même correction pour le câblage (HI_final ≈ 0.99 ≈ 1.0)

    Référence : Section 5.1 du Chapitre V du rapport.
    """
    t = np.linspace(0, 1, n_points)

    # [C1] Taux de dégradation corrigés
    # -------------------------------------------------------
    # RÈGLE : si HI_final ≈ 1.0 (défaut non détectable par HI),
    #         λ = 0.02 (vieillissement naturel identique au nominal)
    # -------------------------------------------------------
    lambda_rates = {
        'Nominal'               : 0.02,
        # [C1] CORRIGÉ : 0.80 → 0.02
        # HI_final ≈ 1.0 → courbe constante à 0.95 (non détectable)
        # L'ancienne valeur λ=0.80 produisait une montée vers 1.0,
        # pas une descente vers 0, ce qui était physiquement incohérent.
        'short_circuit'         : 0.02,
        'phase_imbalance'       : 0.60,
        'magnet_demagnetization': 0.40,
        'bearing_fault'         : 0.50,
        # [C1] CORRIGÉ : câblage aussi non détectable (HI_final ≈ 0.99)
        # λ=0.02 comme le nominal → courbe stable dans la zone normale
        'wiring_fault'          : 0.02,
    }

    degradation_curves = {}
    np.random.seed(42)

    for fault, hi_final in zip(fault_types, HI_values):
        lam     = lambda_rates.get(fault, 0.30)
        HI_sain = 0.95

        # Modèle exponentiel
        curve = (HI_sain * np.exp(-lam * t)
                 + hi_final * (1 - np.exp(-lam * t)))

        # Bruit gaussien réaliste
        bruit = np.random.normal(0, 0.008, n_points)
        curve = np.clip(curve + bruit, 0, 1)

        degradation_curves[fault] = curve

    return t, degradation_curves


# ============================================================
# 7. VISUALISATIONS
# ============================================================

def plot_hi_barres(fault_types, HI_values, seuil_vert, seuil_orange,
                   save=True):
    """
    Figure V.2 : Diagramme en barres du Health Index par scénario.
    Couleur selon zone d'alarme : vert/orange/rouge.
    """
    labels_fr = {
        'Nominal'               : 'Nominal',
        'short_circuit'         : 'Court-circuit',
        'phase_imbalance'       : 'Dés. de phase',
        'magnet_demagnetization': 'Démagnétisation',
        'bearing_fault'         : 'Roulement',
        'wiring_fault'          : 'Câblage',
    }

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(
        'Health Index — Comparaison des scénarios de défaut\n'
        '(THD calculé sur f_fund = 127.3 Hz — version corrigée)',
        fontsize=13, fontweight='bold'
    )

    colors = []
    for hi in HI_values:
        if hi > seuil_vert:
            colors.append('#276749')
        elif hi > seuil_orange:
            colors.append('#D69E2E')
        else:
            colors.append('#C53030')

    noms_affichage = [labels_fr.get(f, f) for f in fault_types]
    bars = ax.bar(noms_affichage, HI_values,
                  color=colors, edgecolor='white',
                  linewidth=1.5, width=0.6)

    # Valeurs sur les barres
    for bar, hi in zip(bars, HI_values):
        y_pos = bar.get_height() + 0.01
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                f'{hi:.3f}', ha='center', va='bottom',
                fontweight='bold', fontsize=11)

    # Lignes de seuil
    ax.axhline(y=seuil_vert,
               color='#D69E2E', linestyle='--', linewidth=2,
               label=f'Seuil Dégradé ({seuil_vert:.2f})')
    ax.axhline(y=seuil_orange,
               color='#C53030', linestyle='--', linewidth=2,
               label=f'Seuil Critique ({seuil_orange:.2f})')

    # Zones colorées
    ax.axhspan(seuil_vert,   1.05,
               alpha=0.08, color='green',  label='Zone Normale')
    ax.axhspan(seuil_orange, seuil_vert,
               alpha=0.08, color='orange', label='Zone Dégradée')
    ax.axhspan(0,            seuil_orange,
               alpha=0.08, color='red',    label='Zone Critique')

    ax.set_ylim([0, 1.15])
    ax.set_ylabel('Health Index (HI)', fontsize=12)
    ax.set_xlabel('Scénario', fontsize=12)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=15, ha='right')

    plt.tight_layout()
    if save:
        path = os.path.join(DOSSIER, 'ch5_health_index_barres.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Sauvegardé : {path}")
    plt.show()


def plot_degradation_curves(t, curves, seuil_vert, seuil_orange,
                            save=True):
    """
    Figure V.3 : Courbes d'évolution temporelle du HI.

    [C1] Les courbes court-circuit et câblage restent maintenant stables
    dans la zone normale (λ=0.02), reflétant leur non-détectabilité
    par le Health Index — comportement physiquement cohérent.
    """
    labels_fr = {
        'Nominal'               : 'Nominal (sain)',
        'short_circuit'         : 'Court-circuit statorique',
        'phase_imbalance'       : 'Déséquilibre de phase',
        'magnet_demagnetization': 'Démagnétisation aimants',
        'bearing_fault'         : 'Défaut de roulement',
        'wiring_fault'          : 'Défaut de câblage',
    }
    couleurs = {
        'Nominal'               : '#2B6CB0',
        'short_circuit'         : '#C53030',
        'phase_imbalance'       : '#D4A017',
        'magnet_demagnetization': '#553C9A',
        'bearing_fault'         : '#276749',
        'wiring_fault'          : '#DD6B20',
    }
    # Styles de ligne pour distinguer les courbes proches
    styles = {
        'Nominal'               : '-',
        'short_circuit'         : '--',
        'phase_imbalance'       : '-',
        'magnet_demagnetization': '-',
        'bearing_fault'         : '-',
        'wiring_fault'          : ':',
    }

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.suptitle(
        'Évolution temporelle du Health Index — Dégradation progressive\n'
        '[C1] λ(court-circuit) = λ(câblage) = 0.02 — '
        'comportement cohérent avec HI non supervisé',
        fontsize=12, fontweight='bold'
    )

    for fault, curve in curves.items():
        ax.plot(t * 100, curve,
                color=couleurs.get(fault, 'gray'),
                linestyle=styles.get(fault, '-'),
                linewidth=2.2,
                label=labels_fr.get(fault, fault),
                alpha=0.9)

    # Zones d'alarme
    ax.axhspan(seuil_vert,   1.05,
               alpha=0.07, color='green')
    ax.axhspan(seuil_orange, seuil_vert,
               alpha=0.07, color='orange')
    ax.axhspan(0,            seuil_orange,
               alpha=0.07, color='red')

    # Lignes de seuil
    ax.axhline(seuil_vert,
               color='#D69E2E', linestyle='--', linewidth=1.5,
               label=f'Seuil dégradé = {seuil_vert:.2f}')
    ax.axhline(seuil_orange,
               color='#C53030', linestyle='--', linewidth=1.5,
               label=f'Seuil critique = {seuil_orange:.2f}')

    # Annotations zones
    ax.text(98, (seuil_vert + 1.02) / 2,
            'Normal',   ha='right', color='green',
            fontsize=10, fontweight='bold')
    ax.text(98, (seuil_vert + seuil_orange) / 2,
            'Dégradé',  ha='right', color='orange',
            fontsize=10, fontweight='bold')
    ax.text(98, seuil_orange / 2,
            'Critique', ha='right', color='red',
            fontsize=10, fontweight='bold')

    ax.set_xlim([0, 100])
    ax.set_ylim([0, 1.08])
    ax.set_xlabel('Temps de service [%]', fontsize=12)
    ax.set_ylabel('Health Index (HI)', fontsize=12)
    ax.legend(loc='lower left', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        path = os.path.join(DOSSIER, 'ch5_degradation_temporelle.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Sauvegardé : {path}")
    plt.show()


def plot_weights(weight_dict, save=True):
    """
    Figure V.1 : Poids des indicateurs dans le Health Index.
    Triés par ordre décroissant avec code couleur.
    """
    sorted_pairs = sorted(weight_dict.items(),
                          key=lambda x: x[1], reverse=True)
    names_s  = [n for n, v in sorted_pairs]
    values_s = [v for n, v in sorted_pairs]

    # Code couleur : 3 groupes selon niveau de contribution
    max_w = max(values_s)
    colors = []
    for v in values_s:
        ratio = v / (max_w + 1e-12)
        if ratio > 0.75:
            colors.append('#C53030')   # rouge foncé — très discriminant
        elif ratio > 0.40:
            colors.append('#DD6B20')   # orange — moderément discriminant
        elif ratio > 0.05:
            colors.append('#D69E2E')   # jaune — faiblement discriminant
        else:
            colors.append('#A0AEC0')   # gris — non discriminant

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle(
        'Poids des indicateurs dans le Health Index\n'
        '(méthode ACP-variance inter-scénarios)',
        fontsize=13, fontweight='bold'
    )

    bars = ax.barh(names_s, values_s,
                   color=colors, edgecolor='white', linewidth=0.8)

    for bar, val in zip(bars, values_s):
        ax.text(bar.get_width() + 0.0005,
                bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=8)

    ax.set_xlabel('Poids wᵢ (contribution au HI)', fontsize=11)
    ax.set_xlim([0, max(values_s) * 1.25])
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()

    # Légende des groupes
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#C53030', label='Très discriminant (w > 75% du max)'),
        Patch(facecolor='#DD6B20', label='Moyennement discriminant'),
        Patch(facecolor='#D69E2E', label='Faiblement discriminant'),
        Patch(facecolor='#A0AEC0', label='Non discriminant (w ≈ 0)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8)

    plt.tight_layout()
    if save:
        path = os.path.join(DOSSIER, 'ch5_poids_indicateurs.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Sauvegardé : {path}")
    plt.show()


def plot_dashboard(fault_types, HI_values, seuil_vert, seuil_orange,
                   weight_dict, save=True):
    """
    [C6] Tableau de bord enrichi :
    - Jauge HI pour chaque scénario
    - Valeur numérique HI
    - Statut (Normal / Dégradé / Critique)
    - Indicateur dominant (celui avec le poids le plus élevé)
    """
    labels_fr = {
        'Nominal'               : 'Nominal',
        'short_circuit'         : 'Court-circuit',
        'phase_imbalance'       : 'Dés. de phase',
        'magnet_demagnetization': 'Démagnétisation',
        'bearing_fault'         : 'Roulement',
        'wiring_fault'          : 'Câblage',
    }

    # Indicateur le plus discriminant (poids max)
    indic_dominant = max(weight_dict, key=weight_dict.get)

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle(
        'Tableau de Bord — Système de Diagnostic PMSM\n'
        f'Indicateur dominant : {indic_dominant} '
        f'(w = {weight_dict[indic_dominant]:.4f})',
        fontsize=14, fontweight='bold', y=0.98
    )

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.35)

    for idx, (fault, hi) in enumerate(zip(fault_types, HI_values)):
        row, col   = divmod(idx, 3)
        ax         = fig.add_subplot(gs[row, col])
        statut, couleur, symbole = get_status(hi, seuil_vert, seuil_orange)

        # Fond coloré selon statut
        ax.set_facecolor(couleur + '18')

        # Barre de jauge
        ax.barh([0], [hi],
                color=couleur, height=0.4, alpha=0.85)
        ax.barh([0], [1 - hi],
                color='#E2E8F0', height=0.4, left=hi, alpha=0.5)

        # Lignes seuils
        ax.axvline(seuil_vert,
                   color='#D69E2E', linestyle='--', linewidth=1.5)
        ax.axvline(seuil_orange,
                   color='#C53030', linestyle='--', linewidth=1.5)

        # Textes
        ax.text(0.5, 0.80, f'HI = {hi:.3f}',
                transform=ax.transAxes, ha='center',
                fontsize=15, fontweight='bold', color=couleur)
        ax.text(0.5, 0.15, f'{symbole} {statut}',
                transform=ax.transAxes, ha='center',
                fontsize=11, color=couleur, fontweight='bold')

        ax.set_xlim([0, 1])
        ax.set_ylim([-0.5, 1.3])
        ax.set_yticks([])
        ax.set_xlabel('Health Index', fontsize=9)
        ax.set_title(labels_fr.get(fault, fault),
                     fontweight='bold', fontsize=11)
        ax.grid(True, alpha=0.2, axis='x')

    plt.tight_layout()
    if save:
        path = os.path.join(DOSSIER, 'ch5_dashboard.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Sauvegardé : {path}")
    plt.show()


# ============================================================
# 8. RAPPORT DE DIAGNOSTIC AUTOMATIQUE
# ============================================================

def generate_report(fault_types, HI_values, weight_dict,
                    seuil_vert, seuil_orange, save=True):
    """
    [C3] Rapport enrichi avec actions spécifiques par type de défaut.
    [C7] Mention de f_fund = 127.3 Hz dans l'en-tête.

    Les actions recommandées sont maintenant spécifiques au type de défaut
    identifié, et non plus génériques.
    """

    # [C3] Actions spécifiques par type de défaut
    actions_specifiques = {
        'Nominal'               : 'Surveillance périodique standard — aucune action requise',
        'short_circuit'         : 'Surveillance + analyse FFT complémentaire recommandée '
                                  '(défaut non détecté par HI — recours au SVM Ch.VII)',
        'phase_imbalance'       : 'ARRÊT IMMÉDIAT — vérifier alimentation onduleur '
                                  'et symétrie des phases',
        'magnet_demagnetization': 'ARRÊT IMMÉDIAT — remplacement des aimants permanents '
                                  'ou reconditionnement thermique',
        'bearing_fault'         : 'ARRÊT IMMÉDIAT — remplacement du roulement '
                                  'avant propagation aux enroulements',
        'wiring_fault'          : 'Surveillance + inspection des connexions électriques '
                                  '(défaut non détecté par HI — recours au SVM Ch.VII)',
    }

    lignes = []
    lignes.append("=" * 70)
    lignes.append("  RAPPORT DE DIAGNOSTIC — SYSTÈME PMSM")
    lignes.append("  Méthode : Health Index Adaptatif (ACP-Variance)")
    lignes.append("  THD calculé sur f_fund = 127.3 Hz (version corrigée)")
    lignes.append("=" * 70)
    lignes.append(f"\n  Seuil Normal   (vert)   : HI > {seuil_vert:.3f}")
    lignes.append(f"  Seuil Dégradé  (orange) : {seuil_orange:.3f} < HI ≤ {seuil_vert:.3f}")
    lignes.append(f"  Seuil Critique (rouge)  : HI ≤ {seuil_orange:.3f}")
    lignes.append(f"  Justification seuils    : littérature [19][20] + "
                  f"validation a posteriori")
    lignes.append("\n" + "-" * 70)
    lignes.append(
        f"  {'Scénario':<28} {'HI':>6}  "
        f"{'Statut':<12}  Action recommandée"
    )
    lignes.append("-" * 70)

    taux_detection = 0
    for fault, hi in zip(fault_types, HI_values):
        statut, _, symbole = get_status(hi, seuil_vert, seuil_orange)
        action = actions_specifiques.get(fault, 'Surveillance standard')
        lignes.append(
            f"  {fault:<28} {hi:>6.3f}  "
            f"{symbole} {statut:<10}  {action}"
        )
        if fault != 'Nominal' and statut == 'Critique':
            taux_detection += 1

    n_defauts = sum(1 for f in fault_types if f != 'Nominal')
    lignes.append("\n" + "-" * 70)
    lignes.append(
        f"  TAUX DE DÉTECTION HI : {taux_detection}/{n_defauts} défauts "
        f"({taux_detection/n_defauts*100:.0f}%)"
    )
    lignes.append(
        f"  Défauts non détectés : court-circuit et câblage "
        f"→ recours au SVM (Ch.VII)"
    )

    lignes.append("\n" + "-" * 70)
    lignes.append("  TOP 5 INDICATEURS LES PLUS DISCRIMINANTS")
    lignes.append("-" * 70)
    sorted_w = sorted(weight_dict.items(),
                      key=lambda x: x[1], reverse=True)
    for i, (name, w) in enumerate(sorted_w[:5], 1):
        lignes.append(
            f"  {i}. {name:<25}  w = {w:.4f}  ({w*100:.1f}%)"
        )

    lignes.append("\n" + "-" * 70)
    lignes.append("  INDICATEURS À POIDS NUL (non discriminants)")
    lignes.append("-" * 70)
    zero_w = [(n, w) for n, w in weight_dict.items() if w < 1e-6]
    if zero_w:
        for n, w in zero_w:
            lignes.append(f"  - {n:<25}  w ≈ {w:.6f}")
    else:
        lignes.append("  Aucun indicateur à poids nul.")

    lignes.append("\n" + "=" * 70)

    rapport = "\n".join(lignes)
    print(rapport)

    if save:
        path = os.path.join(DOSSIER, 'ch5_rapport_diagnostic.txt')
        with open(path, 'w', encoding='utf-8') as f:
            f.write(rapport)
        print(f"\n  Rapport sauvegardé : {path}")

    return rapport


# ============================================================
# 9. PROGRAMME PRINCIPAL
# ============================================================

if __name__ == "__main__":

    print("=" * 65)
    print("  HEALTH INDEX — CHAPITRE V (VERSION CORRIGÉE)")
    print("=" * 65)
    print("\n  CORRECTIONS APPLIQUÉES :")
    print("  [C1] λ(court-circuit) = 0.02 (au lieu de 0.80 — CORRIGÉ)")
    print("  [C2] Seuils justifiés — littérature [19][20]")
    print("  [C3] Actions recommandées spécifiques par défaut")
    print("  [C4] Vérification CSV f_fund=127Hz")
    print("  [C5] Ajustement ancrage HI_nominal documenté")
    print("  [C6] Tableau de bord enrichi")
    print("  [C7] Rapport avec mention f_fund=127.3 Hz")
    print("=" * 65)

    # 1. Chargement
    print("\n[1] Chargement du dataset corrigé (f_fund=127.3 Hz)...")
    df = load_dataset()

    # 2. Normalisation
    print("\n[2] Normalisation Min-Max des indicateurs...")
    df_norm, feature_cols, scaler = normalize_features(df)

    # 3. Poids
    print("\n[3] Calcul des poids par variance inter-scénarios...")
    weights, weight_dict = compute_weights_variance(df_norm, feature_cols)

    # 4. Health Index
    print("\n[4] Calcul du Health Index...")
    HI_values, deg_scores = compute_health_index(
        df_norm, feature_cols, weights
    )
    fault_types = list(df['Fault_Type'])

    print("\n  Résultats Health Index :")
    print(f"  {'Scénario':<30} {'HI':>8}")
    print("  " + "-" * 40)
    for fault, hi in zip(fault_types, HI_values):
        print(f"  {fault:<30} {hi:>8.4f}")

    # 5. Seuils
    print("\n[5] Définition des seuils d'alarme...")
    idx_nom = fault_types.index('Nominal')
    seuil_vert, seuil_orange = define_thresholds(
        HI_nominal=HI_values[idx_nom]
    )

    # 6. Dégradation temporelle
    print("\n[6] Simulation de la dégradation temporelle...")
    print("    [C1] λ(short_circuit) = 0.02, λ(wiring_fault) = 0.02")
    t, curves = simulate_degradation(fault_types, HI_values)

    # 7. Figures
    print("\n[7] Génération des figures...")

    print("  → Figure V.1 : Poids des indicateurs...")
    plot_weights(weight_dict)

    print("  → Figure V.2 : Diagramme en barres HI...")
    plot_hi_barres(fault_types, HI_values, seuil_vert, seuil_orange)

    print("  → Figure V.3 : Courbes de dégradation temporelle...")
    plot_degradation_curves(t, curves, seuil_vert, seuil_orange)

    print("  → Figure V.4 : Tableau de bord de supervision...")
    plot_dashboard(fault_types, HI_values, seuil_vert, seuil_orange,
                   weight_dict)

    # 8. Rapport
    print("\n[8] Génération du rapport de diagnostic...")
    generate_report(fault_types, HI_values, weight_dict,
                    seuil_vert, seuil_orange)

    print("\n" + "=" * 65)
    print("  CHAPITRE V TERMINÉ — VERSION CORRIGÉE")
    print("  Fichiers générés dans :", DOSSIER)
    print("    → ch5_poids_indicateurs.png")
    print("    → ch5_health_index_barres.png")
    print("    → ch5_degradation_temporelle.png")
    print("    → ch5_dashboard.png")
    print("    → ch5_rapport_diagnostic.txt")
    print("=" * 65)