import cv2
import face_recognition

# Charger l'image
image = face_recognition.load_image_file("talon.jpg")

# Détecter les visages
face_locations = face_recognition.face_locations(image, model="cnn")
print(f"Nombre de visages détectés : {len(face_locations)}")

# Convertir l'image en format OpenCV (BGR)
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# Dessiner des rectangles autour des visages détectés
for (top, right, bottom, left) in face_locations:
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

# Afficher l'image
cv2.imshow("Visages détectés", image)
cv2.waitKey(0)
cv2.destroyAllWindows()