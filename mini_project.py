"""

**Étape 1 — Exploration** Calcule les statistiques descriptives et visualise les relations entre chaque variable explicative et le spread. Identifie visuellement quelles variables semblent les plus importantes.

**Étape 2 — Modèle de base** Entraîne une régression linéaire multiple sur l'ensemble des variables. Affiche les coefficients et interprète-les dans un contexte financier (ex : que signifie le coefficient de `rating_score` pour un gérant obligataire ?).

**Étape 3 — Évaluation** Calcule R², RMSE, et MAE. Fais un train/test split (80/20) et compare les performances in-sample vs out-of-sample. Qu'est-ce que ça dit sur la qualité de ton modèle ?

**Étape 4 — Diagnostic des résidus** Produis un QQ-plot et teste la normalité des résidus (Shapiro-Wilk ou Jarque-Bera). Les hypothèses OLS sont-elles respectées ? Commente.

**Étape 5 — Question ouverte** Une variable n'a quasiment aucun pouvoir explicatif dans ce dataset. Laquelle, et pourquoi selon toi ?

"""

import pandas as pd
import numpy as np
import matplotlib as plt

np.random.seed(42)
n = 120

data = {
    "maturite": np.random.uniform(1, 15, n),
    "rating_score": np.random.randint(1, 8, n),  # 1=AAA, 7=BBB-
    "dette_ebitda": np.random.uniform(0.5, 6.0, n),
    "couverture_int": np.random.uniform(1.5, 12.0, n),
    "taille_emission": np.random.uniform(200, 2000, n),  # en millions EUR
}

# Spread = combinaison linéaire des facteurs + bruit
spread = (
    20
    + 8 * data["maturite"]
    + 18 * data["rating_score"]
    + 12 * data["dette_ebitda"]
    - 6 * data["couverture_int"]
    - 0.02 * data["taille_emission"]
    + np.random.normal(0, 15, n)
)

df = pd.DataFrame(data)
df["spread_bps"] = spread


def exploration(df=df):
    """
    Compute descriptives statiscal & visualisations of the relationship between each explicative variables and the spread.
    This function aims to understand which variables is the most important.
    """
    # Provide descriptive stats
    print(df.describe())

    # Compute the mean/std/var for each column
    for col in df.columns:
        # Compute metrics and relation
        metrics = {
            "mean": df[col].mean(),
            "standard deviation": df[col].std(),
            "variance": df[col].var(),
        }

    print(f"Column: {col}")
    print(metrics)


def visualization(df):
    """
    Visualize statsiscal relationship between variables & parameters
    """
