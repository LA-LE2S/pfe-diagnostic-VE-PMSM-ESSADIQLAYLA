"""
=============================================================
  INTERFACE DE SUPERVISION — CHAPITRE VI du PFE
  Diagnostic et Maintenance Prédictive du PMSM
  Auteur : [Votre Nom] | Date : 2025
=============================================================
  Interface graphique complète avec 4 onglets :
  Onglet 1 : Visualisation des signaux temporels
  Onglet 2 : Analyse fréquentielle (FFT + Wavelet)
  Onglet 3 : Health Index + Alertes
  Onglet 4 : Rapport de diagnostic + Export
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk, messagebox
import os, sys, warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CHEMINS
# ─────────────────────────────────────────────
DOSSIER = r'C:\PFE_PMSM\resultats'
os.makedirs(DOSSIER, exist_ok=True)

# ─────────────────────────────────────────────
# IMPORT DES MODULES PRÉCÉDENTS
# ─────────────────────────────────────────────
sys.path.insert(0, r'C:\PFE_PMSM')
from pmsm_simulation import run_simulation, params
from pmsm_analyse_frequentielle import (
    preprocess_signal, compute_fft, wavelet_energy
)

# ─────────────────────────────────────────────
# COULEURS ET STYLES
# ─────────────────────────────────────────────
COULEURS = {
    'Nominal'               : '#2B6CB0',
    'short_circuit'         : '#C53030',
    'phase_imbalance'       : '#D69E2E',
    'magnet_demagnetization': '#553C9A',
    'bearing_fault'         : '#276749',
    'wiring_fault'          : '#DD6B20',
}
LABELS_FR = {
    'Nominal'               : 'Nominal (sain)',
    'short_circuit'         : 'Court-circuit statorique',
    'phase_imbalance'       : 'Déséquilibre de phase',
    'magnet_demagnetization': 'Démagnétisation aimants',
    'bearing_fault'         : 'Défaut de roulement',
    'wiring_fault'          : 'Défaut de câblage',
}
SCENARIOS = [
    ('Nominal',                None,                    0.0),
    ('short_circuit',          'short_circuit',         0.5),
    ('phase_imbalance',        'phase_imbalance',       0.6),
    ('magnet_demagnetization', 'magnet_demagnetization',0.5),
    ('bearing_fault',          'bearing_fault',         0.7),
    ('wiring_fault',           'wiring_fault',          0.4),
]

# ─────────────────────────────────────────────
# CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────────
def charger_donnees():
    """Lance toutes les simulations et charge le dataset features."""
    print("  Chargement des simulations...")
    simulations = {}
    for nom, fault, sev in SCENARIOS:
        simulations[nom] = run_simulation(
            fault_type=fault, fault_severity=sev, t_end=1.5
        )
        print(f"    {nom} OK")

    csv_path = os.path.join(DOSSIER, 'ch4_feature_dataset.csv')
    if os.path.exists(csv_path):
        df_features = pd.read_csv(csv_path)
        print("  Dataset features chargé.")
    else:
        df_features = None
        print("  Dataset features introuvable — lancez d'abord pmsm_analyse_frequentielle.py")

    return simulations, df_features

# ─────────────────────────────────────────────
# CALCUL DU HEALTH INDEX
# ─────────────────────────────────────────────
def calculer_hi(df_features):
    """Calcule le Health Index depuis le dataset features."""
    if df_features is None:
        return {}, 0.85, 0.65

    feature_cols = [c for c in df_features.columns
                    if c not in ['Fault_Type', 'Severity']]
    X = df_features[feature_cols].values.astype(float)

    # Normalisation
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_norm = (X - X_min) / (X_max - X_min + 1e-12)

    # Poids par variance
    variances = np.var(X_norm, axis=0)
    weights   = variances / (variances.sum() + 1e-12)

    # Score de dégradation
    scores = X_norm @ weights
    s_min, s_max = scores.min(), scores.max()
    scores_norm  = (scores - s_min) / (s_max - s_min + 1e-12)
    HI = 1 - scores_norm

    # Ancrage nominal
    idx_nom = list(df_features['Fault_Type']).index('Nominal')
    if HI[idx_nom] < 0.9:
        HI = np.clip(HI * (0.95 / HI[idx_nom]), 0, 1)

    hi_dict = dict(zip(df_features['Fault_Type'], HI))
    seuil_vert   = HI[idx_nom] - 0.15
    seuil_orange = HI[idx_nom] - 0.35
    return hi_dict, seuil_vert, seuil_orange

# ─────────────────────────────────────────────────────────────────
# CLASSE PRINCIPALE : APPLICATION TKINTER
# ─────────────────────────────────────────────────────────────────
class AppSupervision:
    def __init__(self, root, simulations, df_features):
        self.root        = root
        self.sims        = simulations
        self.df_features = df_features
        self.hi_dict, self.seuil_vert, self.seuil_orange = calculer_hi(df_features)

        # ── Configuration fenêtre ──
        self.root.title("Système de Supervision PMSM — Diagnostic Prédictif")
        self.root.geometry("1280x800")
        self.root.configure(bg='#1a1a2e')

        self._build_header()
        self._build_notebook()

    # ── EN-TÊTE ──────────────────────────────────────────
    def _build_header(self):
        header = tk.Frame(self.root, bg='#16213e', height=55)
        header.pack(fill='x', side='top')
        header.pack_propagate(False)

        tk.Label(header,
                 text="  SYSTEME DE SUPERVISION ET DIAGNOSTIC — MOTEUR PMSM",
                 font=('Segoe UI', 13, 'bold'),
                 bg='#16213e', fg='#e0e0e0').pack(side='left', pady=12)

        tk.Label(header,
                 text="PFE — AFD Technologies / Stellantis  ",
                 font=('Segoe UI', 10),
                 bg='#16213e', fg='#888').pack(side='right', pady=12)

    # ── NOTEBOOK (onglets) ───────────────────────────────
    def _build_notebook(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TNotebook',        background='#1a1a2e', borderwidth=0)
        style.configure('TNotebook.Tab',    background='#16213e', foreground='#aaa',
                         padding=[16, 8],   font=('Segoe UI', 10))
        style.map('TNotebook.Tab',
                  background=[('selected','#0f3460')],
                  foreground=[('selected','#ffffff')])

        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill='both', expand=True, padx=8, pady=8)

        self._onglet_signaux()
        self._onglet_fft()
        self._onglet_hi()
        self._onglet_rapport()

    # ════════════════════════════════════════════════════
    # ONGLET 1 : SIGNAUX TEMPORELS
    # ════════════════════════════════════════════════════
    def _onglet_signaux(self):
        frame = ttk.Frame(self.nb)
        self.nb.add(frame, text='  Onglet 1 — Signaux temporels  ')

        # Sélecteur de scénario
        ctrl = tk.Frame(frame, bg='#f0f4f8')
        ctrl.pack(fill='x', padx=10, pady=6)

        tk.Label(ctrl, text="Scénario :", bg='#f0f4f8',
                 font=('Segoe UI', 10)).pack(side='left', padx=8)

        self.var_sc = tk.StringVar(value='Nominal')
        noms = [s[0] for s in SCENARIOS]
        cb = ttk.Combobox(ctrl, textvariable=self.var_sc,
                          values=noms, width=30, state='readonly')
        cb.pack(side='left', padx=4)

        tk.Label(ctrl, text="Signal :", bg='#f0f4f8',
                 font=('Segoe UI', 10)).pack(side='left', padx=12)
        self.var_sig = tk.StringVar(value='ia')
        for sig in ['ia', 'ib', 'ic', 'omega_m', 'Te']:
            tk.Radiobutton(ctrl, text=sig, variable=self.var_sig,
                           value=sig, bg='#f0f4f8').pack(side='left', padx=3)

        tk.Button(ctrl, text="Afficher",
                  command=self._plot_signaux,
                  bg='#0f3460', fg='white',
                  font=('Segoe UI', 10), relief='flat',
                  padx=12).pack(side='left', padx=12)

        # Zone figure
        self.fig_sig = Figure(figsize=(13, 6), facecolor='white')
        self.canvas_sig = FigureCanvasTkAgg(self.fig_sig, master=frame)
        self.canvas_sig.get_tk_widget().pack(fill='both', expand=True,
                                              padx=10, pady=4)
        self._plot_signaux()

    def _plot_signaux(self):
        sc  = self.var_sc.get()
        sig = self.var_sig.get()
        data = self.sims[sc]

        self.fig_sig.clear()
        gs = self.fig_sig.add_gridspec(2, 2, hspace=0.45, wspace=0.3)

        # Signal complet
        ax1 = self.fig_sig.add_subplot(gs[0, :])
        ax1.plot(data['t'], data[sig],
                 color=COULEURS.get(sc, '#2B6CB0'), linewidth=0.8)
        ax1.set_title(f"Signal {sig} — {LABELS_FR.get(sc, sc)}",
                      fontweight='bold')
        ax1.set_xlabel('Temps [s]'); ax1.grid(True, alpha=0.3)

        # Zoom régime établi
        ax2 = self.fig_sig.add_subplot(gs[1, 0])
        mask = data['t'] > 0.8
        ax2.plot(data['t'][mask], data[sig][mask],
                 color=COULEURS.get(sc, '#2B6CB0'), linewidth=1)
        ax2.set_title('Zoom — régime établi (t > 0.8s)')
        ax2.set_xlabel('Temps [s]'); ax2.grid(True, alpha=0.3)

        # Comparaison avec nominal
        ax3 = self.fig_sig.add_subplot(gs[1, 1])
        dn = self.sims['Nominal']
        ax3.plot(dn['t'][mask], dn[sig][mask],
                 color='#2B6CB0', linewidth=1, label='Nominal', alpha=0.7)
        if sc != 'Nominal':
            ax3.plot(data['t'][mask], data[sig][mask],
                     color=COULEURS.get(sc,'gray'), linewidth=1,
                     linestyle='--', label=LABELS_FR.get(sc, sc))
        ax3.set_title('Comparaison avec nominal')
        ax3.set_xlabel('Temps [s]')
        ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

        self.canvas_sig.draw()

    # ════════════════════════════════════════════════════
    # ONGLET 2 : ANALYSE FRÉQUENTIELLE
    # ════════════════════════════════════════════════════
    def _onglet_fft(self):
        frame = ttk.Frame(self.nb)
        self.nb.add(frame, text='  Onglet 2 — Analyse fréquentielle  ')

        ctrl = tk.Frame(frame, bg='#f0f4f8')
        ctrl.pack(fill='x', padx=10, pady=6)

        tk.Label(ctrl, text="Méthode :", bg='#f0f4f8',
                 font=('Segoe UI', 10)).pack(side='left', padx=8)
        self.var_meth = tk.StringVar(value='FFT')
        for m in ['FFT', 'Wavelet']:
            tk.Radiobutton(ctrl, text=m, variable=self.var_meth,
                           value=m, bg='#f0f4f8').pack(side='left', padx=4)

        tk.Label(ctrl, text="Scénario :", bg='#f0f4f8',
                 font=('Segoe UI', 10)).pack(side='left', padx=12)
        self.var_sc2 = tk.StringVar(value='bearing_fault')
        noms = [s[0] for s in SCENARIOS]
        ttk.Combobox(ctrl, textvariable=self.var_sc2,
                     values=noms, width=25, state='readonly').pack(side='left')

        tk.Button(ctrl, text="Analyser",
                  command=self._plot_freq,
                  bg='#0f3460', fg='white',
                  font=('Segoe UI', 10), relief='flat',
                  padx=12).pack(side='left', padx=12)

        self.fig_freq = Figure(figsize=(13, 6), facecolor='white')
        self.canvas_freq = FigureCanvasTkAgg(self.fig_freq, master=frame)
        self.canvas_freq.get_tk_widget().pack(fill='both', expand=True,
                                               padx=10, pady=4)
        self._plot_freq()

    def _plot_freq(self):
        sc   = self.var_sc2.get()
        meth = self.var_meth.get()
        self.fig_freq.clear()

        data_n = self.sims['Nominal']
        data_f = self.sims[sc]
        fs = 10000

        if meth == 'FFT':
            ax1 = self.fig_freq.add_subplot(2, 1, 1)
            ax2 = self.fig_freq.add_subplot(2, 1, 2)

            for ax, data, label, color in [
                (ax1, data_n, 'Nominal (sain)', '#2B6CB0'),
                (ax2, data_f, LABELS_FR.get(sc, sc), COULEURS.get(sc,'gray'))
            ]:
                sig_w, _, _ = preprocess_signal(data['ia'], data['t'], fs)
                freq, amp   = compute_fft(sig_w, fs)
                ax.semilogy(freq, amp + 1e-9, color=color, linewidth=1)
                ax.set_xlim([0, 500]); ax.set_ylim([1e-6, 10])
                ax.set_title(f'Spectre FFT — {label}', fontweight='bold')
                ax.set_xlabel('Fréquence [Hz]')
                ax.set_ylabel('Amplitude [A]')
                ax.grid(True, alpha=0.3, which='both')
                for f_h in [50, 100, 150, 200, 250]:
                    ax.axvline(f_h, color='orange', lw=0.7, alpha=0.5)

            self.fig_freq.tight_layout()

        else:  # Wavelet
            import pywt
            gs = self.fig_freq.add_gridspec(5, 2, hspace=0.6, wspace=0.3)
            titres = ['Signal original', 'Approx. (cA4)',
                      'Détail D4', 'Détail D3', 'Détail D2']
            colors = ['#2B6CB0', '#276749', '#C05621', '#553C9A', '#C53030']

            for col, (data, label) in enumerate([(data_n, 'Nominal'),
                                                  (data_f, LABELS_FR.get(sc,sc))]):
                mask = data['t'] > 0.8
                sig  = data['ia'][mask]
                coeffs = pywt.wavedec(sig, 'db4', level=4)
                all_c  = [sig] + list(coeffs)[:4]

                for row, (c, titre, col_c) in enumerate(
                        zip(all_c, titres, colors)):
                    ax = self.fig_freq.add_subplot(gs[row, col])
                    ax.plot(c, color=col_c, linewidth=0.6)
                    ax.set_title(f'{titre} — {label}', fontsize=8)
                    ax.tick_params(labelsize=7)
                    ax.grid(True, alpha=0.2)

        self.canvas_freq.draw()

    # ════════════════════════════════════════════════════
    # ONGLET 3 : HEALTH INDEX + ALERTES
    # ════════════════════════════════════════════════════
    def _onglet_hi(self):
        frame = ttk.Frame(self.nb)
        self.nb.add(frame, text='  Onglet 3 — Health Index  ')

        # Panneau gauche : jauges
        left = tk.Frame(frame, bg='#f8fafc', width=420)
        left.pack(side='left', fill='y', padx=8, pady=8)
        left.pack_propagate(False)

        tk.Label(left, text="État de santé par scénario",
                 font=('Segoe UI', 12, 'bold'),
                 bg='#f8fafc').pack(pady=10)

        self.jauge_labels = {}
        for nom, _, _ in SCENARIOS:
            hi  = self.hi_dict.get(nom, 0.5)
            if hi > self.seuil_vert:
                statut, fg, bg = 'Normal   ✓', '#155724', '#d4edda'
            elif hi > self.seuil_orange:
                statut, fg, bg = 'Dégradé  ⚠', '#856404', '#fff3cd'
            else:
                statut, fg, bg = 'Critique ✗', '#721c24', '#f8d7da'

            row = tk.Frame(left, bg=bg, relief='solid', bd=1)
            row.pack(fill='x', padx=12, pady=3)

            tk.Label(row, text=f"{LABELS_FR.get(nom,nom):<30}",
                     font=('Consolas', 9), bg=bg, fg='#333').pack(side='left', padx=8, pady=6)
            tk.Label(row, text=f"HI = {hi:.3f}",
                     font=('Consolas', 9, 'bold'), bg=bg, fg=fg).pack(side='left')
            tk.Label(row, text=f"  {statut}",
                     font=('Consolas', 9, 'bold'), bg=bg, fg=fg).pack(side='right', padx=8)

        # Légende seuils
        leg = tk.Frame(left, bg='#f8fafc')
        leg.pack(pady=12, padx=12, fill='x')
        tk.Label(leg, text=f"Seuil Normal   : HI > {self.seuil_vert:.2f}",
                 bg='#f8fafc', fg='#155724', font=('Segoe UI', 9)).pack(anchor='w')
        tk.Label(leg, text=f"Seuil Dégradé  : HI > {self.seuil_orange:.2f}",
                 bg='#f8fafc', fg='#856404', font=('Segoe UI', 9)).pack(anchor='w')
        tk.Label(leg, text=f"Seuil Critique : HI ≤ {self.seuil_orange:.2f}",
                 bg='#f8fafc', fg='#721c24', font=('Segoe UI', 9)).pack(anchor='w')

        # Panneau droit : graphique dégradation
        right = tk.Frame(frame, bg='white')
        right.pack(side='left', fill='both', expand=True, padx=4, pady=8)

        self.fig_hi = Figure(figsize=(8, 6), facecolor='white')
        self.canvas_hi = FigureCanvasTkAgg(self.fig_hi, master=right)
        self.canvas_hi.get_tk_widget().pack(fill='both', expand=True)
        self._plot_hi_graph()

    def _plot_hi_graph(self):
        ax = self.fig_hi.add_subplot(111)
        t  = np.linspace(0, 100, 200)

        lambdas = {
            'Nominal': 0.02, 'short_circuit': 0.8,
            'phase_imbalance': 0.6, 'magnet_demagnetization': 0.4,
            'bearing_fault': 0.5, 'wiring_fault': 0.3,
        }
        for nom, _, _ in SCENARIOS:
            hi_fin = self.hi_dict.get(nom, 0.5)
            lam    = lambdas.get(nom, 0.3)
            curve  = 0.95 * np.exp(-lam * t/100) + hi_fin * (1 - np.exp(-lam * t/100))
            np.random.seed(42)
            curve += np.random.normal(0, 0.006, len(t))
            curve  = np.clip(curve, 0, 1)
            ax.plot(t, curve, color=COULEURS.get(nom,'gray'),
                    linewidth=2, label=LABELS_FR.get(nom, nom))

        ax.axhspan(self.seuil_vert, 1.05, alpha=0.07, color='green')
        ax.axhspan(self.seuil_orange, self.seuil_vert, alpha=0.07, color='orange')
        ax.axhspan(0, self.seuil_orange, alpha=0.07, color='red')
        ax.axhline(self.seuil_vert,   color='#D69E2E', ls='--', lw=1.5,
                   label=f'Seuil dégradé = {self.seuil_vert:.2f}')
        ax.axhline(self.seuil_orange, color='#C53030', ls='--', lw=1.5,
                   label=f'Seuil critique = {self.seuil_orange:.2f}')

        ax.set_xlim([0, 100]); ax.set_ylim([0, 1.05])
        ax.set_xlabel('Temps de service [%]', fontsize=11)
        ax.set_ylabel('Health Index (HI)', fontsize=11)
        ax.set_title('Évolution temporelle du Health Index', fontweight='bold')
        ax.legend(fontsize=8, loc='lower left')
        ax.grid(True, alpha=0.3)
        self.canvas_hi.draw()

    # ════════════════════════════════════════════════════
    # ONGLET 4 : RAPPORT + EXPORT
    # ════════════════════════════════════════════════════
    def _onglet_rapport(self):
        frame = ttk.Frame(self.nb)
        self.nb.add(frame, text='  Onglet 4 — Rapport & Export  ')

        # Boutons export
        ctrl = tk.Frame(frame, bg='#f0f4f8')
        ctrl.pack(fill='x', padx=10, pady=8)

        tk.Button(ctrl, text="Exporter rapport TXT",
                  command=self._export_txt,
                  bg='#276749', fg='white',
                  font=('Segoe UI', 10), relief='flat',
                  padx=14, pady=4).pack(side='left', padx=8)

        tk.Button(ctrl, text="Exporter figures PNG",
                  command=self._export_figures,
                  bg='#0f3460', fg='white',
                  font=('Segoe UI', 10), relief='flat',
                  padx=14, pady=4).pack(side='left', padx=8)

        tk.Button(ctrl, text="Ouvrir dossier résultats",
                  command=lambda: os.startfile(DOSSIER),
                  bg='#553C9A', fg='white',
                  font=('Segoe UI', 10), relief='flat',
                  padx=14, pady=4).pack(side='left', padx=8)

        # Zone texte rapport
        txt_frame = tk.Frame(frame)
        txt_frame.pack(fill='both', expand=True, padx=10, pady=4)

        scrollbar = tk.Scrollbar(txt_frame)
        scrollbar.pack(side='right', fill='y')

        self.txt_rapport = tk.Text(txt_frame, font=('Consolas', 10),
                                    bg='#1e1e1e', fg='#d4d4d4',
                                    yscrollcommand=scrollbar.set,
                                    relief='flat', padx=12, pady=8)
        self.txt_rapport.pack(fill='both', expand=True)
        scrollbar.config(command=self.txt_rapport.yview)

        self._generer_rapport()

    def _generer_rapport(self):
        actions = {
            'Normal'  : 'Surveillance périodique standard',
            'Dégradé' : 'Inspection planifiée sous 30 jours',
            'Critique': 'Arrêt immédiat et maintenance corrective',
        }
        lignes = [
            "=" * 65,
            "  RAPPORT DE DIAGNOSTIC — SYSTÈME PMSM",
            "  Méthode : Health Index Adaptatif (ACP-Variance)",
            "=" * 65,
            f"",
            f"  Seuil Normal   : HI > {self.seuil_vert:.2f}",
            f"  Seuil Dégradé  : {self.seuil_orange:.2f} < HI ≤ {self.seuil_vert:.2f}",
            f"  Seuil Critique : HI ≤ {self.seuil_orange:.2f}",
            "",
            "-" * 65,
            f"  {'Scénario':<30} {'HI':>6}  {'Statut':<12}  Action recommandée",
            "-" * 65,
        ]

        for nom, _, _ in SCENARIOS:
            hi = self.hi_dict.get(nom, 0.5)
            if hi > self.seuil_vert:
                statut, sym = 'Normal', '✓'
            elif hi > self.seuil_orange:
                statut, sym = 'Dégradé', '⚠'
            else:
                statut, sym = 'Critique', '✗'
            action = actions[statut]
            lignes.append(
                f"  {LABELS_FR.get(nom,nom):<30} {hi:>6.3f}  {sym} {statut:<10}  {action}"
            )

        lignes += ["", "=" * 65,
                   "  INDICATEURS LES PLUS DISCRIMINANTS",
                   "-" * 65]

        if self.df_features is not None:
            fc = [c for c in self.df_features.columns
                  if c not in ['Fault_Type','Severity']]
            X = self.df_features[fc].values.astype(float)
            Xn = (X - X.min(0)) / (X.max(0) - X.min(0) + 1e-12)
            w  = np.var(Xn, axis=0)
            w /= w.sum() + 1e-12
            top5 = sorted(zip(fc, w), key=lambda x: x[1], reverse=True)[:5]
            for i, (n, wi) in enumerate(top5, 1):
                lignes.append(f"  {i}. {n:<28} poids = {wi:.4f} ({wi*100:.1f}%)")

        lignes.append("=" * 65)

        rapport = "\n".join(lignes)
        self.txt_rapport.delete('1.0', tk.END)
        self.txt_rapport.insert(tk.END, rapport)
        self._rapport_texte = rapport

    def _export_txt(self):
        path = os.path.join(DOSSIER, 'ch6_rapport_interface.txt')
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self._rapport_texte)
        messagebox.showinfo("Export réussi",
                            f"Rapport sauvegardé :\n{path}")

    def _export_figures(self):
        """Exporte les figures principales en PNG."""
        # Figure HI barres
        fig, ax = plt.subplots(figsize=(10, 5))
        noms = [LABELS_FR.get(s[0],s[0]) for s in SCENARIOS]
        his  = [self.hi_dict.get(s[0], 0.5) for s in SCENARIOS]
        cols = ['#276749' if h > self.seuil_vert
                else '#D69E2E' if h > self.seuil_orange
                else '#C53030' for h in his]
        bars = ax.bar(noms, his, color=cols, edgecolor='white', linewidth=1.2)
        for bar, h in zip(bars, his):
            ax.text(bar.get_x()+bar.get_width()/2, h+0.01,
                    f'{h:.3f}', ha='center', fontweight='bold', fontsize=10)
        ax.axhline(self.seuil_vert,   color='#D69E2E', ls='--', lw=2)
        ax.axhline(self.seuil_orange, color='#C53030', ls='--', lw=2)
        ax.set_ylim([0, 1.1])
        ax.set_title('Health Index — Interface de supervision', fontweight='bold')
        ax.set_ylabel('HI'); ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        path_fig = os.path.join(DOSSIER, 'ch6_hi_interface.png')
        plt.savefig(path_fig, dpi=150, bbox_inches='tight')
        plt.close()
        messagebox.showinfo("Export réussi",
                            f"Figure sauvegardée :\n{path_fig}")

# ─────────────────────────────────────────────
# PROGRAMME PRINCIPAL
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  INTERFACE DE SUPERVISION — CHAPITRE VI")
    print("=" * 60)

    print("\n[1] Chargement des données...")
    simulations, df_features = charger_donnees()

    print("\n[2] Lancement de l'interface graphique...")
    root = tk.Tk()
    app  = AppSupervision(root, simulations, df_features)
    print("  Interface ouverte — utilisez les onglets !")
    root.mainloop()
    print("\n  Interface fermée.")
    print("=" * 60)
