import os
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)

n = 5000

montant = rng.lognormal(6.2, 1.2, n)

ancienne = rng.integers(0, 120, n)

freq = rng.poisson(3, n) + 1

nuit = rng.integers(0, 2, n)

score = (
    2.2 * (np.log10(montant) - 3.0)
    - ancienne / 60
    + (freq - 3) / 3
    + nuit * 1.2
)

# Générer exactement 568 fraudes sur 5000 transactions
# Sélectionner exactement les 568 transactions au score le plus élevé
fraude = np.zeros(n, dtype=int)
indices_fraude = np.argsort(score)[-568:]
fraude[indices_fraude] = 1
df = pd.DataFrame({
    "montant": montant,
    "ancienne": ancienne,
    "freq_24h": freq,
    "nuit": nuit,
    "fraude": fraude
})
os.makedirs("data", exist_ok=True)
df.to_csv("data/transactions.csv", index=False)

print(
    "data/transactions.csv écrit :",
    len(df),
    "lignes,",
    int(df["fraude"].sum()),
    "fraudes"
)