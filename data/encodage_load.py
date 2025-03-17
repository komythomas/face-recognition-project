import json
import numpy as np

# Charger les encodages depuis le fichier JSON
with open("encodages.json", "r") as f:
    encodages = json.load(f)

# Convertir la liste en tableau numpy
encoding_connu = np.array(encodages["personne_connue"])

print("Encodage chargé :", encoding_connu)