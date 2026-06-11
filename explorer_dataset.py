"""
=============================================================
  EXPLORATEUR DU DATASET RÉEL — À lancer en premier
  Pour identifier les colonnes et labels du dataset Bacha
=============================================================
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

CSV = r'C:\PFE_PMSM\dataset_reel\converted_dataset.csv'
df  = pd.read_csv(CSV)

print("=" * 60)
print("  EXPLORATION DU DATASET RÉEL")
print("=" * 60)
print(f"\n  Dimensions : {df.shape}")
print(f"\n  Toutes les colonnes :")
for i, c in enumerate(df.columns):
    print(f"    {i:2d}. {c}  →  type={df[c].dtype}, "
          f"exemples={list(df[c].unique()[:5])}")

# Chercher la colonne label
print("\n  Colonnes possibles pour le label :")
for c in df.columns:
    if df[c].dtype == object or df[c].nunique() < 20:
        print(f"    '{c}' — valeurs uniques : {sorted(df[c].unique())}")

# Statistiques de base
print("\n  Statistiques numériques :")
print(df.describe().round(3).to_string())

# Sauvegarder un aperçu
df.head(30).to_csv(r'C:\PFE_PMSM\resultats\apercu_dataset_reel.csv', index=False)
print("\n  Aperçu sauvegardé : apercu_dataset_reel.csv")
print("\n  Envoyez ce résultat à Claude pour corriger le script !")
