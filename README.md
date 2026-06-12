# pfe-diagnostic-VE-PMSM-ESSADIQLAYLA
Diagnostic prédictif moteur PMSM — ENSAM Rabat 202
# Diagnostic Prédictif des Systèmes de Propulsion Électriques (PMSM)

Auteur : Layla Essadiq  

## Résumé

Ce dépôt contient l'ensemble du code, des données et de la documentation du projet de fin d'études portant sur le développement d'une méthodologie de diagnostic et de maintenance prédictive des systèmes de propulsion des véhicules électriques, centrée sur le moteur synchrone à aimants permanents (PMSM).

**Mots-clés :** PMSM, diagnostic prédictif, Health Index, FFT, ondelettes, SVM, maintenance prédictive

---

## 📁 Structure du dépôt

PFE-PMSM-Diagnostic/
├── README.md                          # Ce fichier
├── LICENSE                            # Licence MIT
├── requirements.txt                   # Dépendances Python (versions exactes)
│
├── data/
│   ├── simule/                        # Données simulées (générées)
│   └── reel/
│       └── converted_dataset.csv      # Dataset expérimental Bacha et al. (2024)
│
├── src/                               # Scripts principaux du pipeline
│   ├── pmsm_simulation.py            # Chapitre III : Simulation PMSM
│   ├── pmsm_analyse_frequentielle.py # Chapitre IV : Analyse fréquentielle
│   ├── pmsm_health_index.py          # Chapitre V : Health Index
│   ├── explorer_dataset.py           # Chapitre VI : Exploration dataset réel
│   ├── bacha.py                      # Chapitre VI : Validation sur données réelles
│   ├── pmsm_perspective1.py         # Chapitre VI : Comparaison sim vs réel
│   ├── pmsm_perspective2_ml.py      # Chapitre VII : Machine Learning
│   └── pmsm_interface.py            # Interface de visualisation
│
├── tests/                             # Jeux d'essai et tests
│   ├── test_minimal.py               # Test rapide (< 30s)
│   ├── sample_data.npz               # Données minimales (3 fenêtres)
│   └── test_results/                 # Résultats des tests (généré)
│
├── docs/                              # Documentation complémentaire
│   └── guide_installation.pdf        # Guide détaillé (Annexe B du rapport)
│
└── resultats/                         # Résultats générés automatiquement
├── ch3_.png                     # Figures Chapitre III
├── ch4_.png                     # Figures Chapitre IV
├── ch5_.png                     # Figures Chapitre V
├── ch6_.png                     # Figures Chapitre VI
├── perspective2_*.png            # Figures Chapitre VII
└── *.csv                         # Données intermédiaires
