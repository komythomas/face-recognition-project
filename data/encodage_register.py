import json
import face_recognition

# Charger une image et encoder le visage
image = face_recognition.load_image_file("talon.jpg")
encoding = face_recognition.face_encodings(image)[0]

# Convertir l'encodage en liste (car JSON ne supporte pas les tableaux numpy)
encoding_list = encoding.tolist()

# Enregistrer l'encodage dans un fichier JSON
with open("encodages.json", "w") as f:
    json.dump({"personne_connue": encoding_list}, f)

print("Encodage enregistré dans encodages.json")