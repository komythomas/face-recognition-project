import cv2
import face_recognition

# Charger l'image de référence
image_connue = face_recognition.load_image_file("komy_portrait.jpg")
encoding_connu = face_recognition.face_encodings(image_connue)[0]

# Démarrer la webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    rgb_frame = frame[:, :, ::-1]  # Convertir en RGB

    # Détecter les visages
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Comparer avec l'encodage connu
        resultats = face_recognition.compare_faces([encoding_connu], face_encoding)

        if resultats[0]:
            nom = "Personne connue"
        else:
            nom = "Inconnu"

        # Dessiner un rectangle et afficher le nom
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, nom, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Reconnaissance faciale', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()