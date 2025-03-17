import face_recognition
import cv2

# Charger les images de référence
image_connue = face_recognition.load_image_file("talon.jpg")
encoding_connu = face_recognition.face_encodings(image_connue)[0]

# Charger l'image inconnue
image_inconnue = face_recognition.load_image_file("gouv.jpg")
encoding_inconnu = face_recognition.face_encodings(image_inconnue)

# Comparer les encodages
for i, encoding in enumerate(encoding_inconnu):
    resultats = face_recognition.compare_faces([encoding_connu], encoding)
    if resultats[0]:
        print(f"Visage {i + 1} : C'est la même personne !")
    else:
        print(f"Visage {i + 1} : Ce n'est pas la même personne.")