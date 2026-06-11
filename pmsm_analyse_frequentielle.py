"""
=============================================================
  ANALYSE FRÉQUENTIELLE & EXTRACTION DES INDICATEURS
  CHAPITRE IV du PFE
  Auteur : [Votre Nom] | Date : 2025
=============================================================
  Ce script :
  1. Importe les signaux simulés (depuis pmsm_simulation.py)
  2. Applique FFT, STFT, Wavelet
  3. Identifie les signatures fréquentielles des défauts
  4. Extrait les indicateurs de santé (features)
  5. Exporte un tableau prêt pour le Health Index (Ch.V)
=============================================================
"""

import numpy as np
from scipy import signal
from scipy.stats import kurtosis, skew
import pywt
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
import warnings
import os
warnings.filterwarnings('ignore')

# Création automatique du dossier de résultats
DOSSIER = r'C:\PFE_PMSM\resultats'
os.makedirs(DOSSIER, exist_ok=True)

# On importe les fonctions du module simulation
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# ============================================================
# 1. PRÉTRAITEMENT DES SIGNAUX
# ============================================================

def preprocess_signal(signal_raw, t, fs=10000, t_start=0.8):
    """
    Prétraitement : sélection du régime établi + fenêtrage de Hann.

    fs       : fréquence d'échantillonnage [Hz]
    t_start  : début du régime établi [s] (on ignore le transitoire)
    """
    # Sélection du régime établi
    mask    = t >= t_start
    sig     = signal_raw[mask]
    t_est   = t[mask]

    # Fenêtre de Hann (réduit les fuites spectrales)
    window  = np.hanning(len(sig))
    sig_win = sig * window

    return sig_win, t_est, fs


# ============================================================
# 2. ANALYSE FFT (Transformée de Fourier Rapide)
# ============================================================

def compute_fft(sig, fs):
    """
    Calcule le spectre FFT.
    Retourne les fréquences et l'amplitude normalisée.
    """
    N    = len(sig)
    freq = np.fft.rfftfreq(N, d=1/fs)
    fft  = np.fft.rfft(sig)
    amp  = (2.0 / N) * np.abs(fft)
    return freq, amp


def plot_fft_comparison(data_sain, data_fault_dict, signal_key='ia', fs=10000, save_fig=True):
    """
    Trace les spectres FFT : sain vs tous les défauts
    """
    fault_labels = {
        'short_circuit'          : 'Court-circuit statorique',
        'phase_imbalance'        : 'Déséquilibre de phase',
        'magnet_demagnetization' : 'Démagnétisation aimants',
        'bearing_fault'          : 'Défaut de roulement',
        'wiring_fault'           : 'Défaut de câblage',
    }

    n_faults = len(data_fault_dict)
    fig, axes = plt.subplots(n_faults, 1, figsize=(14, 4*n_faults))
    fig.suptitle(f"Analyse FFT — Signal {signal_key} : Sain vs Défauts", fontsize=13, fontweight='bold')

    sig_s, t_s, _ = preprocess_signal(data_sain[signal_key], data_sain['t'], fs)
    freq_s, amp_s = compute_fft(sig_s, fs)

    for ax, (fault_name, data_f) in zip(axes, data_fault_dict.items()):
        sig_f, t_f, _ = preprocess_signal(data_f[signal_key], data_f['t'], fs)
        freq_f, amp_f = compute_fft(sig_f, fs)

        ax.semilogy(freq_s, amp_s + 1e-6, color='#2B6CB0', linewidth=1, label='Sain', alpha=0.8)
        ax.semilogy(freq_f, amp_f + 1e-6, color='#E53E3E', linewidth=1,
                    label=fault_labels.get(fault_name, fault_name), alpha=0.9, linestyle='--')
        ax.set_xlim([0, 500])
        ax.set_title(f"Défaut : {fault_labels.get(fault_name, fault_name)}")
        ax.set_xlabel('Fréquence [Hz]')
        ax.set_ylabel('Amplitude [A]')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        # Annotation des harmoniques principales
        for k, f_harm in enumerate([50, 100, 150, 200, 250]):
            ax.axvline(x=f_harm, color='orange', linestyle=':', linewidth=0.8, alpha=0.6)

    plt.tight_layout()
    if save_fig:
        plt.savefig('C:/PFE_PMSM/resultats/ch4_fft_comparaison.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 3. ANALYSE STFT (Spectrogramme Temps-Fréquence)
# ============================================================

def plot_stft(data, signal_key='ia', fs=10000, fault_label='', save_fig=True):
    """
    Calcule et trace le spectrogramme STFT d'un signal.
    """
    sig_raw = data[signal_key]
    t_raw   = data['t']
    mask    = t_raw >= 0.3
    sig     = sig_raw[mask]

    f, t_stft, Zxx = signal.stft(sig, fs=fs, nperseg=512, noverlap=384, window='hann')

    fig, ax = plt.subplots(figsize=(13, 5))
    label_title = 'Nominal (sain)' if fault_label == '' else f'Défaut : {fault_label}'
    ax.set_title(f'Spectrogramme STFT — {signal_key} — {label_title}', fontweight='bold')

    Zxx_abs = np.abs(Zxx) + 1e-9
    pcm = ax.pcolormesh(t_stft, f, Zxx_abs, norm=LogNorm(vmin=Zxx_abs.min(), vmax=Zxx_abs.max()),
                        cmap='inferno', shading='gouraud')
    plt.colorbar(pcm, ax=ax, label='Amplitude [A]')
    ax.set_ylim([0, 500])
    ax.set_xlabel('Temps [s]')
    ax.set_ylabel('Fréquence [Hz]')

    plt.tight_layout()
    if save_fig:
        fname = f"C:/PFE_PMSM/resultats/ch4_stft_{fault_label or 'nominal'}.png"
        plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 4. ANALYSE WAVELET (Ondelettes de Daubechies)
# ============================================================

def wavelet_energy(sig, wavelet='db4', level=5):
    """
    Décomposition en ondelettes et calcul de l'énergie par niveau.
    Retourne les énergies normalisées de chaque sous-bande.
    """
    coeffs = pywt.wavedec(sig, wavelet, level=level)
    energies = []
    total_energy = sum(np.sum(c**2) for c in coeffs)

    for c in coeffs:
        e = np.sum(c**2) / (total_energy + 1e-12)
        energies.append(e)

    return energies  # [approximation, detail5, detail4, detail3, detail2, detail1]


def plot_wavelet_decomposition(data, signal_key='ia', fault_label='', save_fig=True):
    """
    Trace la décomposition en ondelettes du signal.
    """
    mask = data['t'] >= 0.8
    sig  = data[signal_key][mask]
    coeffs = pywt.wavedec(sig, 'db4', level=4)

    fig, axes = plt.subplots(len(coeffs)+1, 1, figsize=(13, 10))
    label_title = 'Nominal (sain)' if fault_label == '' else f'Défaut : {fault_label}'
    fig.suptitle(f'Décomposition Wavelet (db4) — {signal_key} — {label_title}',
                 fontweight='bold', fontsize=12)

    axes[0].plot(sig, color='#2B6CB0', linewidth=0.8)
    axes[0].set_title('Signal original')
    axes[0].set_ylabel('[A]')

    noms = ['Approximation (cA4)', 'Détail niveau 4 (cD4)',
            'Détail niveau 3 (cD3)', 'Détail niveau 2 (cD2)', 'Détail niveau 1 (cD1)']
    colors = ['#276749', '#C05621', '#553C9A', '#2B6CB0', '#E53E3E']

    for ax, c, nom, col in zip(axes[1:], coeffs, noms, colors):
        ax.plot(c, linewidth=0.8, color=col)
        ax.set_title(nom)
        ax.set_ylabel('[A]')
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    if save_fig:
        fname = f"C:/PFE_PMSM/resultats/ch4_wavelet_{fault_label or 'nominal'}.png"
        plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 5. EXTRACTION DES INDICATEURS DE SANTÉ
# ============================================================

def extract_features(data, signal_key='ia', fs=10000):
    """
    Extrait tous les indicateurs temporels, fréquentiels et wavelet.

    Retourne un dictionnaire d'indicateurs (features).
    """
    mask    = data['t'] >= 0.8
    sig_raw = data[signal_key][mask]

    # ---- Indicateurs temporels ----
    rms     = np.sqrt(np.mean(sig_raw**2))
    peak    = np.max(np.abs(sig_raw))
    crest   = peak / (rms + 1e-12)
    kurt    = kurtosis(sig_raw)
    skewness = skew(sig_raw)
    variance = np.var(sig_raw)
    peak2peak = np.max(sig_raw) - np.min(sig_raw)

    # ---- Indicateurs fréquentiels (FFT) ----
    sig_win, _, _ = preprocess_signal(data[signal_key], data['t'], fs)
    freq, amp     = compute_fft(sig_win, fs)

    # Énergie dans les bandes de fréquence caractéristiques
    def band_energy(freq, amp, f_low, f_high):
        mask_b = (freq >= f_low) & (freq < f_high)
        return np.sum(amp[mask_b]**2)

    e_0_50   = band_energy(freq, amp, 0,   50)    # fondamental
    e_50_100 = band_energy(freq, amp, 50,  100)   # 2ème harmonique
    e_100_200= band_energy(freq, amp, 100, 200)   # 3ème + 4ème harmonique
    e_200_500= band_energy(freq, amp, 200, 500)   # hautes fréquences
    e_total  = band_energy(freq, amp, 0,   500)

    # THD - Taux de Distorsion Harmonique
    f_fund = 50.0
    idx_fund = np.argmin(np.abs(freq - f_fund))
    amp_fund = amp[idx_fund]
    harmonics_amp = []
    for k in range(2, 8):
        idx_h = np.argmin(np.abs(freq - k*f_fund))
        harmonics_amp.append(amp[idx_h])
    thd = np.sqrt(sum(h**2 for h in harmonics_amp)) / (amp_fund + 1e-12)

    # ---- Indicateurs Wavelet ----
    wav_energies = wavelet_energy(sig_raw, wavelet='db4', level=5)

    # ---- Compilation des features ----
    features = {
        # Temporels
        'RMS'         : round(rms, 4),
        'Peak'        : round(peak, 4),
        'Crest_Factor': round(crest, 4),
        'Kurtosis'    : round(kurt, 4),
        'Skewness'    : round(skewness, 4),
        'Variance'    : round(variance, 4),
        'Peak2Peak'   : round(peak2peak, 4),
        # Fréquentiels
        'Energy_0_50Hz'   : round(e_0_50, 6),
        'Energy_50_100Hz' : round(e_50_100, 6),
        'Energy_100_200Hz': round(e_100_200, 6),
        'Energy_200_500Hz': round(e_200_500, 6),
        'THD'             : round(thd, 4),
        # Wavelet
        'Wav_E_approx' : round(wav_energies[0], 4),
        'Wav_E_D5'     : round(wav_energies[1], 4),
        'Wav_E_D4'     : round(wav_energies[2], 4),
        'Wav_E_D3'     : round(wav_energies[3], 4),
        'Wav_E_D2'     : round(wav_energies[4], 4),
        # Label
        'Fault_Type'   : data['fault_type'] if data['fault_type'] else 'Nominal',
        'Severity'     : data['fault_severity'],
    }

    return features


# ============================================================
# 6. CONSTRUCTION DU TABLEAU DE FEATURES (DATASET)
# ============================================================

def build_feature_dataset(data_nominal, data_fault_dict, save_csv=True):
    """
    Construit le tableau complet des indicateurs pour tous les scénarios.
    Ce dataset sera utilisé pour construire le Health Index (Ch.V).
    """
    all_features = []

    # Cas nominal
    features_nom = extract_features(data_nominal)
    all_features.append(features_nom)
    print(f"  Sain    : RMS={features_nom['RMS']:.3f}, Kurt={features_nom['Kurtosis']:.3f}, THD={features_nom['THD']:.3f}")

    # Cas avec défauts
    for fault_name, data_f in data_fault_dict.items():
        features_f = extract_features(data_f)
        all_features.append(features_f)
        print(f"  {fault_name:<35}: RMS={features_f['RMS']:.3f}, Kurt={features_f['Kurtosis']:.3f}, THD={features_f['THD']:.3f}")

    df = pd.DataFrame(all_features)

    if save_csv:
        df.to_csv('C:/PFE_PMSM/resultats/ch4_feature_dataset.csv', index=False)
        print("\n  Dataset exporté : ch4_feature_dataset.csv")

    return df


def plot_feature_heatmap(df, save_fig=True):
    """
    Visualise le tableau des indicateurs sous forme de heatmap normalisée.
    """
    numeric_cols = [c for c in df.columns if c not in ['Fault_Type', 'Severity']]
    df_num = df[numeric_cols].copy().astype(float)

    # Normalisation min-max pour la visualisation
    df_norm = (df_num - df_num.min()) / (df_num.max() - df_num.min() + 1e-12)
    df_norm.index = df['Fault_Type']

    fig, ax = plt.subplots(figsize=(16, 5))
    im = ax.imshow(df_norm.values, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.02, label='Valeur normalisée (0=min, 1=max)')

    ax.set_xticks(range(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['Fault_Type'], fontsize=9)
    ax.set_title('Tableau des indicateurs — Sain vs Défauts (normalisé)', fontweight='bold')

    plt.tight_layout()
    if save_fig:
        plt.savefig('C:/PFE_PMSM/resultats/ch4_feature_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 7. PROGRAMME PRINCIPAL
# ============================================================
if __name__ == "__main__":

    # --- Import des simulations (exécuter pmsm_simulation.py en premier) ---
    # On relance ici rapidement les simulations pour avoir les données
    print("="*60)
    print("  ANALYSE FRÉQUENTIELLE — CHAPITRE IV")
    print("="*60)

    # Import du module simulation
    try:
        from pmsm_simulation import run_simulation, params
    except ImportError:
        print("  Erreur : pmsm_simulation.py introuvable dans le même dossier.")
        exit()

    print("\n[1] Chargement des données simulées...")
    data_nominal = run_simulation(fault_type=None, t_end=1.5)

    defauts = [
        ('short_circuit',           0.5),
        ('phase_imbalance',         0.6),
        ('magnet_demagnetization',  0.5),
        ('bearing_fault',           0.7),
        ('wiring_fault',            0.4),
    ]
    data_faults = {}
    for fault_type, severity in defauts:
        data_faults[fault_type] = run_simulation(fault_type=fault_type, fault_severity=severity, t_end=1.5)

    print("\n[2] Analyse FFT...")
    plot_fft_comparison(data_nominal, data_faults)

    print("\n[3] Analyse STFT (spectrogramme)...")
    plot_stft(data_nominal, fault_label='')
    plot_stft(data_faults['bearing_fault'], fault_label='bearing_fault')

    print("\n[4] Analyse Wavelet...")
    plot_wavelet_decomposition(data_nominal)
    plot_wavelet_decomposition(data_faults['short_circuit'], fault_label='short_circuit')

    print("\n[5] Extraction des indicateurs...")
    df_features = build_feature_dataset(data_nominal, data_faults)

    print("\n[6] Visualisation du tableau des features...")
    plot_feature_heatmap(df_features)

    print("\n[7] Aperçu du dataset :")
    print(df_features.to_string())

    print("\n  Chapitre IV terminé.")
    print("  Fichiers générés : ch4_fft_comparaison.png, ch4_feature_heatmap.png,")
    print("                     ch4_feature_dataset.csv")
    print("="*60)
