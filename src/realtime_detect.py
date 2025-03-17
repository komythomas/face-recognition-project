import cv2
import face_recognition

# Démarrer la webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    rgb_frame = frame[:, :, ::-1]  # Convertir en RGB

    # Détecter les visages
    face_locations = face_recognition.face_locations(rgb_frame)

    # Dessiner des rectangles autour des visages détectés
    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    # Afficher le résultat
    cv2.imshow('Détection en temps réel', frame)

    # Quitter avec la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()