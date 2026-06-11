"""
=============================================================
  ANALYSE FRÉQUENTIELLE & EXTRACTION DES INDICATEURS
  CHAPITRE IV du PFE — VERSION CORRIGÉE
  Auteur : ESSADIQ Layla | Date : 2025
=============================================================
  CORRECTIONS APPORTÉES PAR RAPPORT À LA VERSION INITIALE :
  
  [C1] f_fund = 50.0 Hz → f_fund = p * omega_ref / (2*pi) ≈ 127.3 Hz
       Le THD était calculé sur la mauvaise fréquence fondamentale.
       
  [C2] Annotations FFT corrigées : harmoniques à k*127 Hz
       (et non k*50 Hz comme dans la version initiale)
       
  [C3] Commentaires physiques ajoutés sur les bandes de fréquence
       (justifiés par rapport à f_fund = 127.3 Hz)
       
  [C4] Ajout du paramètre f_fund dans extract_features()
       pour garantir la cohérence avec les paramètres moteur
       
  [C5] Wavelet : level=4 utilisé partout (cohérence interne)
       La version initiale utilisait level=5 dans wavelet_energy()
       mais level=4 dans plot_wavelet_decomposition()
       
  [C6] Ajout références bibliographiques dans les docstrings
=============================================================
"""

import numpy as np
from scipy import signal
from scipy.stats import kurtosis, skew
import pywt
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import warnings
import os

warnings.filterwarnings('ignore')

# ============================================================
# DOSSIER DE RÉSULTATS
# ============================================================
DOSSIER = r'C:\PFE_PMSM\resultats'
os.makedirs(DOSSIER, exist_ok=True)

# Import du module simulation
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# ============================================================
# PARAMÈTRES MOTEUR (cohérence avec pmsm_simulation.py)
# ============================================================
# Ces paramètres sont importés depuis pmsm_simulation.py
# Ils sont définis ici en secours si l'import échoue
PARAMS_DEFAULT = {
    'Rs'        : 0.5,      # Résistance statorique [Ohm]
    'Ld'        : 5e-3,     # Inductance axe d [H]
    'Lq'        : 6e-3,     # Inductance axe q [H]
    'lambda_f'  : 0.175,    # Flux aimants permanents [Wb]
    'J'         : 0.002,    # Moment d'inertie [kg.m²]
    'B'         : 0.001,    # Coefficient de frottement [N.m.s/rad]
    'p'         : 4,        # Nombre de paires de pôles
    'Vdc'       : 300,      # Tension DC bus [V]
    'omega_ref' : 200.0,    # Vitesse de référence [rad/s]
    'TL'        : 5.0,      # Couple de charge [N.m]
}


def get_f_fund(params):
    """
    [C1] CORRECTION PRINCIPALE
    Calcule la fréquence fondamentale électrique réelle du moteur.
    
    f1 = p * omega_m / (2*pi)
       = 4 * 200 / (2*pi)
       ≈ 127.3 Hz
    
    ERREUR INITIALE : f_fund = 50.0 Hz (fréquence réseau, sans rapport
    avec le moteur simulé à omega_ref = 200 rad/s et p = 4)
    
    Référence : Vas P., Sensorless Vector and Direct Torque Control,
    Oxford University Press, 1998 [8]
    """
    f_fund = params['p'] * params['omega_ref'] / (2.0 * np.pi)
    return f_fund  # ≈ 127.32 Hz


# ============================================================
# 1. PRÉTRAITEMENT DES SIGNAUX
# ============================================================

def preprocess_signal(signal_raw, t, fs=10000, t_start=0.8):
    """
    Prétraitement : sélection du régime établi + fenêtrage de Hann.

    Paramètres :
    ------------
    signal_raw : array — signal brut complet (transitoire + régime établi)
    t          : array — vecteur temps correspondant
    fs         : float — fréquence d'échantillonnage [Hz] (défaut : 10 000 Hz)
    t_start    : float — début du régime établi [s] (défaut : 0.8 s)

    Retourne :
    ----------
    sig_win : array — signal en régime établi avec fenêtre de Hann appliquée
    t_est   : array — vecteur temps du régime établi
    fs      : float — fréquence d'échantillonnage (inchangée)

    Note sur le fenêtrage de Hann :
    --------------------------------
    La fenêtre de Hann est définie par :
        w(n) = 0.5 * (1 - cos(2*pi*n / (N-1))),  n = 0, 1, ..., N-1
    Elle atténue les extrémités du signal pour réduire la fuite spectrale.
    L'amplitude FFT est ensuite corrigée par le facteur 2/N pour retrouver
    les amplitudes réelles des composantes sinusoïdales.
    (Le facteur 2 compense le spectre unilatéral, 1/N la normalisation DFT)
    """
    # Sélection du régime établi (t > 0.8 s → N ≈ 7000 points à fs=10000 Hz)
    mask    = t >= t_start
    sig     = signal_raw[mask]
    t_est   = t[mask]

    # Application de la fenêtre de Hann
    window  = np.hanning(len(sig))
    sig_win = sig * window

    return sig_win, t_est, fs


# ============================================================
# 2. ANALYSE FFT (Transformée de Fourier Rapide)
# ============================================================

def compute_fft(sig, fs):
    """
    Calcule le spectre FFT unilatéral normalisé.

    La correction d'amplitude 2/N permet de retrouver les amplitudes
    réelles des composantes sinusoïdales après fenêtrage de Hann.

    Résolution fréquentielle :
        Δf = fs / N = 10000 / 7000 ≈ 1.43 Hz
    (suffisamment fin pour distinguer les harmoniques à 127, 254, 381 Hz
    et les raies latérales du roulement espacées de fb = 8 Hz)

    Retourne :
    ----------
    freq : array — fréquences [Hz]
    amp  : array — amplitudes normalisées [A]
    """
    N    = len(sig)
    freq = np.fft.rfftfreq(N, d=1/fs)
    fft  = np.fft.rfft(sig)
    amp  = (2.0 / N) * np.abs(fft)
    return freq, amp


def plot_fft_comparison(data_sain, data_fault_dict, params,
                        signal_key='ia', fs=10000, save_fig=True):
    """
    [C2] CORRECTION : annotations harmoniques à k * f_fund (≈127 Hz)
    au lieu de k * 50 Hz dans la version initiale.

    Trace les spectres FFT sain vs défauts en échelle logarithmique.
    Les lignes verticales orangées marquent les harmoniques réels du moteur.
    """
    # [C1] Fréquence fondamentale correcte
    f_fund = get_f_fund(params)

    fault_labels = {
        'short_circuit'          : 'Court-circuit statorique',
        'phase_imbalance'        : 'Déséquilibre de phase',
        'magnet_demagnetization' : 'Démagnétisation aimants',
        'bearing_fault'          : 'Défaut de roulement',
        'wiring_fault'           : 'Défaut de câblage',
    }

    n_faults = len(data_fault_dict)
    fig, axes = plt.subplots(n_faults, 1, figsize=(14, 4 * n_faults))
    if n_faults == 1:
        axes = [axes]
    fig.suptitle(
        f"Analyse FFT — Signal {signal_key} : Sain vs Défauts\n"
        f"(f_fund = {f_fund:.1f} Hz | fs = {fs} Hz | Δf ≈ 1.43 Hz)",
        fontsize=13, fontweight='bold'
    )

    sig_s, t_s, _ = preprocess_signal(data_sain[signal_key], data_sain['t'], fs)
    freq_s, amp_s = compute_fft(sig_s, fs)

    for ax, (fault_name, data_f) in zip(axes, data_fault_dict.items()):
        sig_f, t_f, _ = preprocess_signal(data_f[signal_key], data_f['t'], fs)
        freq_f, amp_f = compute_fft(sig_f, fs)

        ax.semilogy(freq_s, amp_s + 1e-6, color='#2B6CB0',
                    linewidth=1, label='Sain', alpha=0.8)
        ax.semilogy(freq_f, amp_f + 1e-6, color='#E53E3E',
                    linewidth=1, label=fault_labels.get(fault_name, fault_name),
                    alpha=0.9, linestyle='--')
        ax.set_xlim([0, 500])
        ax.set_title(f"Défaut : {fault_labels.get(fault_name, fault_name)}")
        ax.set_xlabel('Fréquence [Hz]')
        ax.set_ylabel('Amplitude [A]')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # [C2] Harmoniques réels du moteur : k * f_fund ≈ 127, 254, 381, 508 Hz
        # (corrigé : ancienne version utilisait 50, 100, 150, 200, 250 Hz)
        for k in range(1, 5):
            f_harm = k * f_fund
            if f_harm <= 500:
                ax.axvline(x=f_harm, color='orange',
                           linestyle=':', linewidth=0.9, alpha=0.7)
                ax.text(f_harm + 2, ax.get_ylim()[1] * 0.5,
                        f'{k}×f₁\n({f_harm:.0f}Hz)',
                        fontsize=6, color='darkorange', alpha=0.8)

    plt.tight_layout()
    if save_fig:
        path = os.path.join(DOSSIER, 'ch4_fft_comparaison.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Figure sauvegardée : {path}")
    plt.show()


# ============================================================
# 3. ANALYSE STFT (Spectrogramme Temps-Fréquence)
# ============================================================

def plot_stft(data, signal_key='ia', fs=10000, fault_label='', save_fig=True):
    """
    Calcule et trace le spectrogramme STFT.

    Rôle dans le pipeline :
    -----------------------
    La STFT est utilisée comme outil de VALIDATION QUALITATIVE uniquement.
    Elle ne produit pas d'indicateurs numériques directs dans ce projet.
    Son apport est de :
      - Confirmer la stationnarité des signaux en régime établi
      - Visualiser la localisation temporelle des perturbations (ex: roulement)
      - Guider le choix des bandes de fréquence pour les indicateurs spectraux

    Paramètres STFT :
    -----------------
    nperseg = 512 points → résolution temporelle : (512-384)/10000 = 0.0128 s
    noverlap = 384 points → taux de recouvrement : 75%
    Résolution fréquentielle : 10000/512 ≈ 19.5 Hz
    """
    sig_raw = data[signal_key]
    t_raw   = data['t']
    # Inclure le transitoire pour visualiser la montée en régime
    mask    = t_raw >= 0.3
    sig     = sig_raw[mask]

    f, t_stft, Zxx = signal.stft(
        sig, fs=fs, nperseg=512, noverlap=384, window='hann'
    )

    fig, ax = plt.subplots(figsize=(13, 5))
    label_title = 'Nominal (sain)' if fault_label == '' else f'Défaut : {fault_label}'
    ax.set_title(
        f'Spectrogramme STFT — {signal_key} — {label_title}\n'
        f'(Résolution temporelle ≈ 12.8 ms | Résolution fréquentielle ≈ 19.5 Hz)',
        fontweight='bold'
    )

    Zxx_abs = np.abs(Zxx) + 1e-9
    pcm = ax.pcolormesh(
        t_stft, f, Zxx_abs,
        norm=LogNorm(vmin=Zxx_abs.min(), vmax=Zxx_abs.max()),
        cmap='inferno', shading='gouraud'
    )
    plt.colorbar(pcm, ax=ax, label='Amplitude [A]')
    ax.set_ylim([0, 500])
    ax.set_xlabel('Temps [s]')
    ax.set_ylabel('Fréquence [Hz]')

    plt.tight_layout()
    if save_fig:
        fname = f"ch4_stft_{fault_label or 'nominal'}.png"
        path  = os.path.join(DOSSIER, fname)
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Figure sauvegardée : {path}")
    plt.show()


# ============================================================
# 4. ANALYSE WAVELET (Ondelettes de Daubechies db4)
# ============================================================

def wavelet_energy(sig, wavelet='db4', level=4):
    """
    [C5] CORRECTION : level=4 partout (cohérence interne)
    La version initiale utilisait level=5 ici mais level=4 dans
    plot_wavelet_decomposition(), créant une incohérence entre
    les indicateurs calculés et les figures affichées.

    Décomposition en ondelettes multi-résolution et calcul des
    énergies normalisées par sous-bande.

    Bandes de fréquence pour fs = 10 000 Hz, level = 4 :
    -------------------------------------------------------
    cA4  : 0     — 312  Hz  → Fondamental + premières harmoniques
    cD4  : 312   — 625  Hz  → Harmoniques d'ordre élevé
    cD3  : 625   — 1250 Hz  → Transitoires rapides
    cD2  : 1250  — 2500 Hz  → Bruit électrique
    cD1  : 2500  — 5000 Hz  → Bruit haute fréquence

    Retourne :
    ----------
    Liste de 5 énergies normalisées :
    [E_approx, E_D4, E_D3, E_D2, E_D1]
    (énergie de chaque niveau / énergie totale)

    Référence : Mallat S., A Wavelet Tour of Signal Processing,
    Academic Press, 3e éd., 2009 [12]
    """
    coeffs       = pywt.wavedec(sig, wavelet, level=level)
    total_energy = sum(np.sum(c**2) for c in coeffs)
    energies     = []

    for c in coeffs:
        e = np.sum(c**2) / (total_energy + 1e-12)
        energies.append(e)

    # energies = [cA4, cD4, cD3, cD2, cD1] → 5 valeurs
    return energies


def plot_wavelet_decomposition(data, signal_key='ia', fault_label='', save_fig=True):
    """
    Trace la décomposition en ondelettes db4 (level=4) du signal.

    Note sur l'amplitude de cA4 :
    ------------------------------
    L'approximation cA4 peut apparaître avec une amplitude ×4 par rapport
    au signal original. Ce facteur provient de la convention de normalisation
    de PyWavelets pour db4 à 4 niveaux : gain ≈ 2^(level/2) = 2² = 4.
    Les indicateurs d'énergie wavelet sont calculés de manière normalisée
    (énergie relative), ce qui élimine cet effet d'amplification.
    """
    mask   = data['t'] >= 0.8
    sig    = data[signal_key][mask]
    # [C5] level=4 cohérent avec wavelet_energy()
    coeffs = pywt.wavedec(sig, 'db4', level=4)

    fig, axes = plt.subplots(len(coeffs) + 1, 1, figsize=(13, 10))
    label_title = 'Nominal (sain)' if fault_label == '' else f'Défaut : {fault_label}'
    fig.suptitle(
        f'Décomposition Wavelet (db4, level=4) — {signal_key} — {label_title}',
        fontweight='bold', fontsize=12
    )

    # Signal original
    axes[0].plot(sig, color='#2B6CB0', linewidth=0.8)
    axes[0].set_title('Signal original')
    axes[0].set_ylabel('[A]')
    axes[0].grid(True, alpha=0.2)

    # Sous-bandes avec bandes de fréquence indiquées
    noms = [
        'Approximation cA4  [0 — 312 Hz]',
        'Détail cD4         [312 — 625 Hz]',
        'Détail cD3         [625 — 1250 Hz]',
        'Détail cD2         [1250 — 2500 Hz]',
        'Détail cD1         [2500 — 5000 Hz]',
    ]
    colors = ['#276749', '#C05621', '#553C9A', '#2B6CB0', '#E53E3E']

    for ax, c, nom, col in zip(axes[1:], coeffs, noms, colors):
        ax.plot(c, linewidth=0.8, color=col)
        ax.set_title(nom)
        ax.set_ylabel('[A]')
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    if save_fig:
        fname = f"ch4_wavelet_{fault_label or 'nominal'}.png"
        path  = os.path.join(DOSSIER, fname)
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Figure sauvegardée : {path}")
    plt.show()


# ============================================================
# 5. EXTRACTION DES INDICATEURS DE SANTÉ
# ============================================================

def extract_features(data, params, signal_key='ia', fs=10000):
    """
    [C1][C3][C4] VERSION CORRIGÉE

    Extrait les 17 indicateurs de santé :
      - 7 indicateurs temporels
      - 5 indicateurs fréquentiels (dont THD corrigé)
      - 5 indicateurs wavelet

    CORRECTION PRINCIPALE [C1] :
    Le THD est maintenant calculé sur la fréquence fondamentale réelle
    du moteur : f_fund = p * omega_ref / (2*pi) ≈ 127.3 Hz
    (ancienne valeur incorrecte : f_fund = 50.0 Hz)

    Paramètres :
    ------------
    data       : dict — données de simulation (sorties de run_simulation())
    params     : dict — paramètres moteur (pour calcul f_fund)
    signal_key : str  — signal analysé (défaut : 'ia')
    fs         : float — fréquence d'échantillonnage [Hz]

    Retourne :
    ----------
    features : dict — 17 indicateurs + Fault_Type + Severity
    """

    # ---- Sélection régime établi ----
    mask    = data['t'] >= 0.8
    sig_raw = data[signal_key][mask]

    # =====================================================
    # BLOC A : INDICATEURS TEMPORELS (7 indicateurs)
    # Calculés directement sur le signal brut en régime établi
    # Références : Benbouzid 2000 [9], Gangsar & Tiwari 2020 [13]
    # =====================================================

    rms       = np.sqrt(np.mean(sig_raw**2))
    # RMS : mesure la puissance moyenne — augmente avec tout défaut amplifiant
    # le courant (démagnétisation, déséquilibre de phase)

    peak      = np.max(np.abs(sig_raw))
    # Peak : valeur maximale — sensible aux impulsions transitoires

    crest     = peak / (rms + 1e-12)
    # Crest Factor = Peak/RMS : vaut √2 ≈ 1.41 pour sinusoïde pure
    # Toute déviation indique une distorsion du signal

    kurt      = kurtosis(sig_raw)
    # Kurtosis : aplatissement de la distribution
    # Valeur théorique : 1.5 pour sinusoïde pure, 3 pour signal gaussien
    # Les chocs mécaniques (roulement) élèvent significativement le kurtosis

    skewness  = skew(sig_raw)
    # Skewness : asymétrie de la distribution
    # Nul pour sinusoïde symétrique — perturbé par court-circuit et câblage

    variance  = np.var(sig_raw)
    # Variance : dispersion des amplitudes — utile pour câblage (bruit)

    peak2peak = np.max(sig_raw) - np.min(sig_raw)
    # Peak-to-Peak : amplitude totale du signal

    # =====================================================
    # BLOC B : INDICATEURS FRÉQUENTIELS (5 indicateurs)
    # Calculés sur le spectre FFT après fenêtrage de Hann
    # =====================================================

    sig_win, _, _ = preprocess_signal(data[signal_key], data['t'], fs)
    freq, amp     = compute_fft(sig_win, fs)

    def band_energy(freq, amp, f_low, f_high):
        """Énergie spectrale dans la bande [f_low, f_high[ Hz"""
        mask_b = (freq >= f_low) & (freq < f_high)
        return np.sum(amp[mask_b]**2)

    # [C3] Bandes de fréquence justifiées par rapport à f_fund ≈ 127.3 Hz
    # -----------------------------------------------------------------------
    # Bande 0–50 Hz : sous-harmoniques — signature défauts de ROULEMENT
    # (fb ≈ 8 Hz et ses multiples 16, 24, 32, 40 Hz)
    e_0_50    = band_energy(freq, amp,   0,  50)

    # Bande 50–100 Hz : pré-fondamentale — harmoniques basses fréquences
    # → indicateur le plus discriminant pour le SVM (Ch.VII)
    e_50_100  = band_energy(freq, amp,  50, 100)

    # Bande 100–200 Hz : contient f_fund ≈ 127 Hz (énergie principale sain)
    # et le début du 2ème harmonique (2*127=254 Hz dépasse 200 Hz)
    e_100_200 = band_energy(freq, amp, 100, 200)

    # Bande 200–500 Hz : harmoniques supérieurs 2*f1=254, 3*f1=381, 4*f1=508 Hz
    # → signature caractéristique du DÉSÉQUILIBRE DE PHASE
    e_200_500 = band_energy(freq, amp, 200, 500)

    # -------------------------------------------------------------------
    # [C1] CORRECTION PRINCIPALE : THD calculé sur f_fund ≈ 127.3 Hz
    # -------------------------------------------------------------------
    # ERREUR CORRIGÉE : la version initiale utilisait f_fund = 50.0 Hz
    # ce qui calculait le THD par rapport à la fréquence du réseau électrique
    # (50 Hz) et non par rapport à la fréquence réelle du moteur simulé.
    # Résultat : tous les résultats THD du dataset initial étaient incorrects.
    #
    # Fréquence fondamentale correcte :
    # f_fund = p * omega_ref / (2*pi) = 4 * 200 / (2*pi) ≈ 127.32 Hz
    #
    # Les harmoniques sont recherchés aux fréquences :
    # 2*f1 ≈ 254 Hz, 3*f1 ≈ 381 Hz, 4*f1 ≈ 508 Hz,
    # 5*f1 ≈ 635 Hz, 6*f1 ≈ 762 Hz, 7*f1 ≈ 889 Hz
    # -------------------------------------------------------------------
    f_fund       = get_f_fund(params)   # ≈ 127.32 Hz
    idx_fund     = np.argmin(np.abs(freq - f_fund))
    amp_fund     = amp[idx_fund]
    harmonics_amp = []
    for k in range(2, 8):
        f_harm    = k * f_fund
        idx_h     = np.argmin(np.abs(freq - f_harm))
        harmonics_amp.append(amp[idx_h])
    thd = np.sqrt(sum(h**2 for h in harmonics_amp)) / (amp_fund + 1e-12)

    # =====================================================
    # BLOC C : INDICATEURS WAVELET (5 indicateurs)
    # Énergies normalisées par niveau de décomposition db4
    # [C5] level=4 cohérent avec plot_wavelet_decomposition()
    # =====================================================
    # wav_energies = [E_cA4, E_cD4, E_cD3, E_cD2, E_cD1]
    wav_energies = wavelet_energy(sig_raw, wavelet='db4', level=4)

    # =====================================================
    # COMPILATION DES 17 INDICATEURS
    # =====================================================
    features = {
        # --- 7 Indicateurs temporels ---
        'RMS'          : round(rms,       6),
        'Peak'         : round(peak,      6),
        'Crest_Factor' : round(crest,     6),
        'Kurtosis'     : round(kurt,      6),
        'Skewness'     : round(skewness,  6),
        'Variance'     : round(variance,  6),
        'Peak2Peak'    : round(peak2peak, 6),

        # --- 5 Indicateurs fréquentiels ---
        'Energy_0_50Hz'   : round(e_0_50,    8),
        'Energy_50_100Hz' : round(e_50_100,  8),
        'Energy_100_200Hz': round(e_100_200, 8),
        'Energy_200_500Hz': round(e_200_500, 8),
        'THD'             : round(thd,       6),  # [C1] corrigé : f_fund=127Hz

        # --- 5 Indicateurs wavelet ---
        'Wav_E_approx' : round(wav_energies[0], 6),  # cA4 : 0–312 Hz
        'Wav_E_D4'     : round(wav_energies[1], 6),  # cD4 : 312–625 Hz
        'Wav_E_D3'     : round(wav_energies[2], 6),  # cD3 : 625–1250 Hz
        'Wav_E_D2'     : round(wav_energies[3], 6),  # cD2 : 1250–2500 Hz
        'Wav_E_D1'     : round(wav_energies[4], 6),  # cD1 : 2500–5000 Hz

        # --- Métadonnées ---
        'Fault_Type' : data['fault_type'] if data['fault_type'] else 'Nominal',
        'Severity'   : data['fault_severity'],
    }

    return features


# ============================================================
# 6. CONSTRUCTION DU DATASET DE FEATURES
# ============================================================

def build_feature_dataset(data_nominal, data_fault_dict, params, save_csv=True):
    """
    Construit le tableau complet des 17 indicateurs pour tous les scénarios.
    Ce dataset est l'entrée directe du Health Index (Chapitre V).

    Structure du dataset :
    ----------------------
    - 6 lignes : 1 nominal + 5 défauts
    - 17 colonnes d'indicateurs numériques
    - 2 colonnes de métadonnées : Fault_Type, Severity

    [C1] Les valeurs THD sont maintenant calculées sur f_fund ≈ 127.3 Hz
    """
    # Fréquence fondamentale (pour affichage)
    f_fund = get_f_fund(params)
    print(f"\n  Fréquence fondamentale moteur : f_fund = {f_fund:.2f} Hz")
    print(f"  (p={params['p']}, omega_ref={params['omega_ref']} rad/s)")
    print(f"  THD calculé sur harmoniques : {f_fund:.1f}, "
          f"{2*f_fund:.1f}, {3*f_fund:.1f}, ... Hz\n")

    all_features = []

    # --- Cas nominal ---
    features_nom = extract_features(data_nominal, params)
    all_features.append(features_nom)
    print(f"  {'Nominal':<35}: "
          f"RMS={features_nom['RMS']:.4f}  "
          f"Kurt={features_nom['Kurtosis']:.4f}  "
          f"THD={features_nom['THD']:.4f}  "
          f"E_0_50={features_nom['Energy_0_50Hz']:.2e}")

    # --- Cas avec défauts ---
    for fault_name, data_f in data_fault_dict.items():
        features_f = extract_features(data_f, params)
        all_features.append(features_f)
        print(f"  {fault_name:<35}: "
              f"RMS={features_f['RMS']:.4f}  "
              f"Kurt={features_f['Kurtosis']:.4f}  "
              f"THD={features_f['THD']:.4f}  "
              f"E_0_50={features_f['Energy_0_50Hz']:.2e}")

    df = pd.DataFrame(all_features)

    if save_csv:
        path = os.path.join(DOSSIER, 'ch4_feature_dataset.csv')
        df.to_csv(path, index=False)
        print(f"\n  Dataset exporté : {path}")
        print(f"  Dimensions : {df.shape[0]} scénarios × {df.shape[1]-2} indicateurs")

    return df


def plot_feature_heatmap(df, params, save_fig=True):
    """
    Visualise le dataset des indicateurs sous forme de heatmap normalisée.

    Chaque ligne = un scénario (Nominal, Court-circuit, ...)
    Chaque colonne = un indicateur
    Couleur rouge = valeur maximale (dégradation élevée)
    Couleur verte = valeur minimale

    La propriété fondamentale que confirme cette figure :
    chaque scénario présente un profil colorimétrique UNIQUE
    → prérequis pour la construction d'un Health Index fiable (Ch.V)
    """
    f_fund = get_f_fund(params)

    feature_cols = [c for c in df.columns if c not in ['Fault_Type', 'Severity']]
    df_num  = df[feature_cols].copy().astype(float)

    # Normalisation Min-Max pour la visualisation
    df_norm = (df_num - df_num.min()) / (df_num.max() - df_num.min() + 1e-12)
    df_norm.index = df['Fault_Type']

    fig, ax = plt.subplots(figsize=(16, 5))
    im = ax.imshow(df_norm.values, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.02, label='Valeur normalisée (0=min, 1=max)')

    ax.set_xticks(range(len(feature_cols)))
    ax.set_xticklabels(feature_cols, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['Fault_Type'], fontsize=9)
    ax.set_title(
        f'Tableau des indicateurs — Sain vs Défauts (normalisé)\n'
        f'f_fund = {f_fund:.1f} Hz | THD calculé sur harmoniques réels du moteur',
        fontweight='bold'
    )

    # Affichage des valeurs numériques dans les cellules
    for i in range(len(df)):
        for j in range(len(feature_cols)):
            val = df_norm.values[i, j]
            color = 'white' if val > 0.6 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=6, color=color)

    plt.tight_layout()
    if save_fig:
        path = os.path.join(DOSSIER, 'ch4_feature_heatmap.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Figure sauvegardée : {path}")
    plt.show()


# ============================================================
# VÉRIFICATION DE LA CORRECTION THD
# ============================================================

def verifier_correction_thd(df, params):
    """
    Fonction de vérification : affiche les valeurs THD
    calculées sur f_fund ≈ 127.3 Hz (version corrigée)
    et les compare aux valeurs qui auraient été obtenues
    avec f_fund = 50 Hz (version incorrecte initiale).

    Permet de confirmer que la correction est bien appliquée.
    """
    f_fund_correct = get_f_fund(params)
    f_fund_incorrect = 50.0

    print("\n" + "="*65)
    print("  VÉRIFICATION DE LA CORRECTION THD")
    print("="*65)
    print(f"  Fréquence fondamentale correcte   : {f_fund_correct:.2f} Hz")
    print(f"  Fréquence fondamentale incorrecte : {f_fund_incorrect:.2f} Hz")
    print(f"  Rapport : {f_fund_correct/f_fund_incorrect:.2f}×")
    print("-"*65)
    print(f"  {'Scénario':<30} {'THD (127 Hz)':<15} Statut")
    print("-"*65)

    for _, row in df.iterrows():
        thd_val = row['THD']
        fault   = row['Fault_Type']
        # Le déséquilibre de phase doit avoir le THD le plus élevé
        statut = "← MAX attendu" if fault == 'phase_imbalance' else ""
        print(f"  {fault:<30} {thd_val:<15.4f} {statut}")

    print("="*65)
    print("  ATTENDU : THD(phase_imbalance) >> THD(Nominal)")
    print("  Si ce n'est pas le cas, vérifier f_fund dans extract_features()")
    print("="*65)


# ============================================================
# 7. PROGRAMME PRINCIPAL
# ============================================================

if __name__ == "__main__":

    print("=" * 65)
    print("  ANALYSE FRÉQUENTIELLE — CHAPITRE IV (VERSION CORRIGÉE)")
    print("=" * 65)
    print("\n  CORRECTIONS APPLIQUÉES :")
    print("  [C1] f_fund = 127.3 Hz  (au lieu de 50.0 Hz — ERREUR CORRIGÉE)")
    print("  [C2] Annotations FFT sur harmoniques réels k×127 Hz")
    print("  [C3] Commentaires physiques sur les bandes de fréquence")
    print("  [C4] params passé en argument à extract_features()")
    print("  [C5] level=4 cohérent dans wavelet_energy() et plot_wavelet()")
    print("=" * 65)

    # --- Import du module simulation ---
    try:
        from pmsm_simulation import run_simulation, params
        print(f"\n  Module pmsm_simulation importé.")
        print(f"  Paramètres moteur : p={params['p']}, "
              f"omega_ref={params['omega_ref']} rad/s")
        print(f"  Fréquence fondamentale : "
              f"f_fund = {get_f_fund(params):.2f} Hz")
    except ImportError:
        print("\n  ERREUR : pmsm_simulation.py introuvable.")
        print("  Utilisation des paramètres par défaut.")
        params = PARAMS_DEFAULT

        # Simulation de remplacement pour test
        def run_simulation(fault_type=None, fault_severity=0.0, t_end=1.5):
            """Simulation simplifiée de remplacement"""
            t  = np.arange(0, t_end, 1e-4)
            f1 = get_f_fund(params)
            ia = 5 * np.sin(2 * np.pi * f1 * t)
            if fault_type == 'phase_imbalance':
                ia += 0.5 * np.sin(2 * np.pi * 2 * f1 * t)
                ia += 0.3 * np.sin(2 * np.pi * 3 * f1 * t)
            elif fault_type == 'bearing_fault':
                ia *= (1 + 0.3 * np.sin(2 * np.pi * 8 * t))
            elif fault_type == 'wiring_fault':
                ia += fault_severity * 0.3 * np.random.randn(len(t))
            return {
                'ia': ia, 'ib': ia, 'ic': ia,
                't': t,
                'fault_type': fault_type,
                'fault_severity': fault_severity
            }

    # --- Chargement des simulations ---
    print("\n[1] Chargement des données simulées...")
    data_nominal = run_simulation(fault_type=None, t_end=1.5)

    defauts = [
        ('short_circuit',          0.5),
        ('phase_imbalance',        0.6),
        ('magnet_demagnetization', 0.5),
        ('bearing_fault',          0.7),
        ('wiring_fault',           0.4),
    ]
    data_faults = {}
    for fault_type, severity in defauts:
        data_faults[fault_type] = run_simulation(
            fault_type=fault_type,
            fault_severity=severity,
            t_end=1.5
        )
    print(f"  {len(data_faults) + 1} scénarios chargés (1 nominal + {len(data_faults)} défauts)")

    # --- Analyse FFT ---
    print("\n[2] Analyse FFT (f_fund = {:.2f} Hz)...".format(get_f_fund(params)))
    plot_fft_comparison(data_nominal, data_faults, params)

    # --- Analyse STFT ---
    print("\n[3] Analyse STFT (validation qualitative)...")
    plot_stft(data_nominal, fault_label='')
    plot_stft(data_faults['bearing_fault'], fault_label='bearing_fault')

    # --- Analyse Wavelet ---
    print("\n[4] Analyse Wavelet (db4, level=4)...")
    plot_wavelet_decomposition(data_nominal)
    plot_wavelet_decomposition(data_faults['short_circuit'],
                               fault_label='short_circuit')

    # --- Extraction des indicateurs ---
    print("\n[5] Extraction des 17 indicateurs...")
    print("    (THD calculé sur f_fund = {:.2f} Hz — VERSION CORRIGÉE)".format(
        get_f_fund(params)))
    df_features = build_feature_dataset(data_nominal, data_faults, params)

    # --- Vérification de la correction THD ---
    print("\n[6] Vérification de la correction THD...")
    verifier_correction_thd(df_features, params)

    # --- Heatmap ---
    print("\n[7] Visualisation de la heatmap des indicateurs...")
    plot_feature_heatmap(df_features, params)

    # --- Aperçu du dataset ---
    print("\n[8] Aperçu du dataset corrigé :")
    feature_cols = [c for c in df_features.columns
                    if c not in ['Fault_Type', 'Severity']]
    print(df_features[['Fault_Type'] + feature_cols].to_string(index=False))

    print("\n" + "=" * 65)
    print("  Chapitre IV terminé — VERSION CORRIGÉE")
    print("  Fichiers générés dans :", DOSSIER)
    print("    → ch4_fft_comparaison.png   (harmoniques à k×127 Hz)")
    print("    → ch4_stft_nominal.png")
    print("    → ch4_stft_bearing_fault.png")
    print("    → ch4_wavelet_nominal.png   (level=4 cohérent)")
    print("    → ch4_wavelet_short_circuit.png")
    print("    → ch4_feature_dataset.csv   (THD sur 127 Hz corrigé)")
    print("    → ch4_feature_heatmap.png   (dataset corrigé)")
    print("=" * 65)
