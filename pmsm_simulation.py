"""
=============================================================
  SIMULATION PMSM - CHAPITRE III du PFE
  Modélisation dans le repère d-q + Analyse des signaux
  Auteur : [Votre Nom] | Date : 2025
=============================================================
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
import os
warnings.filterwarnings('ignore')

# Création automatique du dossier de résultats
DOSSIER = r'C:\PFE_PMSM\resultats'
os.makedirs(DOSSIER, exist_ok=True)

# ============================================================
# 1. PARAMÈTRES DU MOTEUR PMSM (moteur de traction typique VE)
# ============================================================
params = {
    'Rs'     : 0.5,       # Résistance statorique [Ohm]
    'Ld'     : 5e-3,      # Inductance axe d [H]
    'Lq'     : 6e-3,      # Inductance axe q [H]
    'lambda_f': 0.175,    # Flux des aimants permanents [Wb]
    'J'      : 0.002,     # Moment d'inertie [kg.m²]
    'B'      : 0.001,     # Coefficient de frottement [N.m.s/rad]
    'p'      : 4,         # Nombre de paires de pôles
    'Vdc'    : 300,       # Tension DC bus [V]
    'omega_ref': 200.0,   # Vitesse de référence [rad/s]
    'TL'     : 5.0,       # Couple de charge [N.m]
}

# ============================================================
# 2. CONTRÔLEUR PI (Régulation de courant FOC)
# ============================================================
# Gains PI ajustés pour réponse rapide
Kp_d, Ki_d = 10.0, 500.0
Kp_q, Ki_q = 10.0, 500.0
Kp_w, Ki_w = 0.5,  20.0

# ============================================================
# 3. MODÈLE MATHÉMATIQUE DU PMSM (équations d-q)
# ============================================================
def pmsm_model(t, state, params, fault_type=None, fault_severity=0.0):
    """
    Modèle d'état du PMSM dans le repère d-q.

    State : [id, iq, omega_m, theta_e, int_ed, int_eq, int_ew]
    Params : dictionnaire des paramètres moteur
    fault_type : type de défaut à injecter ('short_circuit', 'phase_imbalance',
                 'magnet_demagnetization', 'bearing_fault', 'wiring_fault')
    fault_severity : niveau du défaut entre 0 (sain) et 1 (critique)
    """
    Rs      = params['Rs']
    Ld      = params['Ld']
    Lq      = params['Lq']
    lf      = params['lambda_f']
    J       = params['J']
    B       = params['B']
    p       = params['p']
    TL      = params['TL']
    w_ref   = params['omega_ref']

    id_, iq, omega_m, theta_e, int_ed, int_eq, int_ew = state

    # ---- Application du défaut ----
    Rs_eff = Rs
    lf_eff = lf
    TL_eff = TL

    if fault_type == 'short_circuit':
        # Court-circuit statorique : augmentation de Rs
        Rs_eff = Rs * (1 + fault_severity * 3.0)

    elif fault_type == 'phase_imbalance':
        # Déséquilibre de phase : perturbation du courant q
        noise_amp = fault_severity * 2.0
        iq = iq + noise_amp * np.sin(5 * omega_m * t + 1.0)

    elif fault_type == 'magnet_demagnetization':
        # Dégradation des aimants : réduction du flux
        lf_eff = lf * (1 - fault_severity * 0.4)

    elif fault_type == 'bearing_fault':
        # Défaut de roulement : oscillation de couple à fréquence caractéristique
        f_bearing = 8.0  # fréquence caractéristique roulement [Hz]
        TL_eff = TL * (1 + fault_severity * 0.5 * np.abs(np.sin(2*np.pi*f_bearing*t)))

    elif fault_type == 'wiring_fault':
        # Défaut de câblage : augmentation de Rs + bruit
        Rs_eff = Rs * (1 + fault_severity * 1.5)
        iq = iq + fault_severity * 0.3 * np.random.randn()

    # ---- Vitesse électrique ----
    omega_e = p * omega_m

    # ---- Contrôleur PI (FOC - Field Oriented Control) ----
    # Référence courant d = 0 (maximum torque per ampere)
    id_ref = 0.0
    iq_ref = Kp_w * (w_ref - omega_m) + Ki_w * int_ew
    iq_ref = np.clip(iq_ref, -20.0, 20.0)

    # Erreurs courant
    error_d = id_ref - id_
    error_q = iq_ref - iq

    # Tensions de référence
    Vd = Kp_d * error_d + Ki_d * int_ed - omega_e * Lq * iq
    Vq = Kp_q * error_q + Ki_q * int_eq + omega_e * (Ld * id_ + lf_eff)

    # Saturation tension (limitation MLI)
    V_max = params['Vdc'] / np.sqrt(3)
    Vd = np.clip(Vd, -V_max, V_max)
    Vq = np.clip(Vq, -V_max, V_max)

    # ---- Équations différentielles ----
    # Courant axe d
    did_dt = (Vd - Rs_eff*id_ + omega_e*Lq*iq) / Ld

    # Courant axe q
    diq_dt = (Vq - Rs_eff*iq - omega_e*(Ld*id_ + lf_eff)) / Lq

    # Couple électromagnétique
    Te = 1.5 * p * (lf_eff * iq + (Ld - Lq) * id_ * iq)

    # Équation mécanique
    domega_dt = (Te - TL_eff - B*omega_m) / J

    # Angle électrique
    dtheta_dt = omega_e

    # Intégrateurs PI
    dint_ed = error_d
    dint_eq = error_q
    dint_ew = w_ref - omega_m

    return [did_dt, diq_dt, domega_dt, dtheta_dt, dint_ed, dint_eq, dint_ew]


# ============================================================
# 4. RECONSTRUCTION DES SIGNAUX TRIPHASÉS (d-q → abc)
# ============================================================
def dq_to_abc(id_, iq, theta_e):
    """Transformation de Park inverse : repère d-q → triphasé abc"""
    i_alpha = id_ * np.cos(theta_e) - iq * np.sin(theta_e)
    i_beta  = id_ * np.sin(theta_e) + iq * np.cos(theta_e)
    ia = i_alpha
    ib = -0.5*i_alpha + (np.sqrt(3)/2)*i_beta
    ic = -0.5*i_alpha - (np.sqrt(3)/2)*i_beta
    return ia, ib, ic


# ============================================================
# 5. SIMULATION
# ============================================================
def run_simulation(fault_type=None, fault_severity=0.0, t_end=1.5, dt=1e-4):
    """
    Lance la simulation du PMSM.

    fault_type      : None (sain) ou type de défaut
    fault_severity  : 0.0 = sain, 1.0 = défaut maximal
    t_end           : durée de simulation [s]
    dt              : pas de temps [s]
    """
    t_span = (0, t_end)
    t_eval = np.arange(0, t_end, dt)

    # Conditions initiales [id, iq, omega_m, theta_e, int_ed, int_eq, int_ew]
    y0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    print(f"  Simulation : {'Nominale (saine)' if fault_type is None else fault_type} "
          f"(sévérité={fault_severity:.1f})")

    sol = solve_ivp(
        lambda t, y: pmsm_model(t, y, params, fault_type, fault_severity),
        t_span, y0, t_eval=t_eval,
        method='RK45', rtol=1e-6, atol=1e-8, max_step=dt
    )

    t = sol.t
    id_  = sol.y[0]
    iq   = sol.y[1]
    omega_m = sol.y[2]
    theta_e = sol.y[3]

    # Reconstruction des courants triphasés
    ia, ib, ic = dq_to_abc(id_, iq, theta_e)

    # Couple électromagnétique
    lf_eff = params['lambda_f'] * (1 - fault_severity*0.4 if fault_type == 'magnet_demagnetization' else 1)
    Te = 1.5 * params['p'] * (lf_eff * iq + (params['Ld'] - params['Lq']) * id_ * iq)

    return {
        't': t, 'id': id_, 'iq': iq,
        'omega_m': omega_m, 'theta_e': theta_e,
        'ia': ia, 'ib': ib, 'ic': ic, 'Te': Te,
        'fault_type': fault_type, 'fault_severity': fault_severity
    }


# ============================================================
# 6. VISUALISATION DES RÉSULTATS — RÉGIME NOMINAL
# ============================================================
def plot_nominal(data, save_fig=True):
    """Trace les résultats de la simulation nominale"""
    t = data['t']
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle('Simulation PMSM — Régime Nominal (Sans Défaut)', fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # -- Vitesse --
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, data['omega_m'], color='#2B6CB0', linewidth=1.5)
    ax1.axhline(params['omega_ref'], color='red', linestyle='--', linewidth=1, label='Référence')
    ax1.set_title('Vitesse angulaire ω_m'); ax1.set_xlabel('Temps [s]'); ax1.set_ylabel('[rad/s]')
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    # -- Couple --
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, data['Te'], color='#276749', linewidth=1.5)
    ax2.set_title('Couple électromagnétique Te'); ax2.set_xlabel('Temps [s]'); ax2.set_ylabel('[N.m]')
    ax2.grid(True, alpha=0.3)

    # -- Courants dq --
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(t, data['id'], label='id', color='#C05621')
    ax3.plot(t, data['iq'], label='iq', color='#744210')
    ax3.set_title('Courants axes d-q'); ax3.set_xlabel('Temps [s]'); ax3.set_ylabel('[A]')
    ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

    # -- Courants triphasés --
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(t, data['ia'], label='ia', linewidth=1)
    ax4.plot(t, data['ib'], label='ib', linewidth=1)
    ax4.plot(t, data['ic'], label='ic', linewidth=1)
    ax4.set_title('Courants triphasés ia, ib, ic'); ax4.set_xlabel('Temps [s]'); ax4.set_ylabel('[A]')
    ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3)
    ax4.set_xlim([0.8, 1.5])  # zoom en régime établi

    # -- Phase ia (zoom) --
    ax5 = fig.add_subplot(gs[2, 0])
    mask = t > 0.8
    ax5.plot(t[mask], data['ia'][mask], color='#553C9A', linewidth=1.5)
    ax5.set_title('Phase ia — Régime établi (zoom)'); ax5.set_xlabel('Temps [s]'); ax5.set_ylabel('[A]')
    ax5.grid(True, alpha=0.3)

    # -- Espace de phase id-iq --
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(data['id'], data['iq'], color='#702459', alpha=0.5, linewidth=0.8)
    ax6.set_title('Plan de phase id–iq'); ax6.set_xlabel('id [A]'); ax6.set_ylabel('iq [A]')
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_fig:
        plt.savefig('C:/PFE_PMSM/resultats/ch3_simulation_nominale.png', dpi=150, bbox_inches='tight')
        print("  Figure sauvegardée : ch3_simulation_nominale.png")
    plt.show()


# ============================================================
# 7. VISUALISATION COMPARAISON SAIN vs DÉFAUT
# ============================================================
def plot_comparison(data_sain, data_fault, save_fig=True):
    """Compare les courants ia en régime sain et avec défaut"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"Comparaison : Sain vs Défaut [{data_fault['fault_type']}]",
                 fontsize=13, fontweight='bold')

    labels = ['Courant ia [A]', 'Courant iq [A]', 'Vitesse [rad/s]', 'Couple [N.m]']
    keys   = ['ia', 'iq', 'omega_m', 'Te']
    colors_s = ['#2B6CB0', '#276749', '#C05621', '#553C9A']
    colors_f = ['#E53E3E', '#D44A23', '#B7791F', '#9F4FB3']

    for ax, key, label, cs, cf in zip(axes.flat, keys, labels, colors_s, colors_f):
        mask = data_sain['t'] > 0.8
        ax.plot(data_sain['t'][mask],  data_sain[key][mask],  color=cs, linewidth=1.2, label='Sain')
        mask_f = data_fault['t'] > 0.8
        ax.plot(data_fault['t'][mask_f], data_fault[key][mask_f], color=cf, linewidth=1.2,
                linestyle='--', label=f"Défaut ({data_fault['fault_type']})")
        ax.set_title(label); ax.set_xlabel('Temps [s]')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_fig:
        fname = f"C:/PFE_PMSM/resultats/ch3_comparaison_{data_fault['fault_type']}.png"
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        print(f"  Figure sauvegardée : {fname}")
    plt.show()


# ============================================================
# 8. PROGRAMME PRINCIPAL
# ============================================================
if __name__ == "__main__":

    print("="*60)
    print("  SIMULATION PMSM — CHAPITRE III")
    print("="*60)

    # --- Simulation nominale ---
    print("\n[1] Simulation du régime nominal...")
    data_nominal = run_simulation(fault_type=None, fault_severity=0.0, t_end=1.5)
    plot_nominal(data_nominal)

    # --- Simulations avec défauts ---
    defauts = [
        ('short_circuit',           0.5),
        ('phase_imbalance',         0.6),
        ('magnet_demagnetization',  0.5),
        ('bearing_fault',           0.7),
        ('wiring_fault',            0.4),
    ]

    print("\n[2] Simulation des défauts...")
    resultats_defauts = {}
    for fault_type, severity in defauts:
        data_f = run_simulation(fault_type=fault_type, fault_severity=severity, t_end=1.5)
        resultats_defauts[fault_type] = data_f
        plot_comparison(data_nominal, data_f)

    print("\n[3] Toutes les simulations terminées.")
    print("    Les figures sont sauvegardées dans /outputs/")
    print("    Utiliser ces données pour le Chapitre IV (analyse fréquentielle).")
    print("="*60)
