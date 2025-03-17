import tkinter as tk
from tkinter import messagebox, filedialog
import customtkinter as ctk
import cv2
import face_recognition
import json
import numpy as np
import logging
import datetime
import threading

# Configurer les logs
logging.basicConfig(filename="reconnaissance.log", level=logging.INFO, format="%(asctime)s - %(message)s")

# Charger ou créer les encodages
try:
    with open("encodages.json", "r") as f:
        encodages = json.load(f)
except FileNotFoundError:
    encodages = {}

# Fonction pour enregistrer un nouvel encodage
def enregistrer_encodage(nom, image_path):
    image = face_recognition.load_image_file(image_path)
    encoding = face_recognition.face_encodings(image)[0]
    encodages[nom] = encoding.tolist()
    with open("encodages.json", "w") as f:
        json.dump(encodages, f)
    logging.info(f"Encodage enregistré pour {nom}")

# Fonction pour enregistrer un visage
def enregistrer_visage():
    image_path = filedialog.askopenfilename(title="Sélectionner une image", filetypes=[("Images", "*.jpg *.png")])
    if image_path:
        nom = ctk.CTkInputDialog(text="Entrez le nom de la personne :", title="Enregistrer un visage").get_input()
        if nom:
            enregistrer_encodage(nom, image_path)
            messagebox.showinfo("Succès", f"Visage de {nom} enregistré avec succès !")

# Fonction pour supprimer un visage
def supprimer_visage():
    nom = ctk.CTkInputDialog(text="Entrez le nom de la personne à supprimer :", title="Supprimer un visage").get_input()
    if nom and nom in encodages:
        del encodages[nom]
        with open("encodages.json", "w") as f:
            json.dump(encodages, f)
        messagebox.showinfo("Succès", f"Visage de {nom} supprimé avec succès !")
    else:
        messagebox.showwarning("Erreur", f"{nom} non trouvé !")

# Fonction pour enregistrer les logs
def log_visage(nom, position):
    heure = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = f"{heure} - Visage détecté : {nom} à la position {position}"
    logging.info(message)

# Fonction pour démarrer la reconnaissance faciale
def demarrer_reconnaissance():
    thread = threading.Thread(target=reconnaissance_faciale)
    thread.daemon = True
    thread.start()

# Fonction de reconnaissance faciale
def reconnaissance_faciale():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = frame[:, :, ::-1]  # Convertir en RGB

        # Détecter les visages avec HOG
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Mettre à jour le nombre de visages détectés
        nombre_visages.set(f"Visages détectés : {len(face_locations)}")

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Comparer avec les encodages connus
            for nom, encoding_connu in encodages.items():
                encoding_connu = np.array(encoding_connu)
                resultats = face_recognition.compare_faces([encoding_connu], face_encoding)
                if resultats[0]:
                    nom_affiche = nom
                    break
            else:
                nom_affiche = "Inconnu"

            # Dessiner un rectangle et afficher le nom
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, nom_affiche, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Enregistrer le log
            log_visage(nom_affiche, (top, right, bottom, left))

        cv2.imshow('Reconnaissance faciale', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Créer l'interface avec CustomTkinter
ctk.set_appearance_mode("System")  # Thème système
ctk.set_default_color_theme("blue")  # Thème bleu

root = ctk.CTk()
root.title("Reconnaissance Faciale")
root.geometry("400x300")

# Titre
titre = ctk.CTkLabel(root, text="Reconnaissance Faciale", font=("Arial", 20))
titre.pack(pady=20)

# Boutons
btn_demarrer = ctk.CTkButton(root, text="Démarrer la reconnaissance", command=demarrer_reconnaissance)
btn_demarrer.pack(pady=10)

btn_enregistrer = ctk.CTkButton(root, text="Enregistrer un visage", command=enregistrer_visage)
btn_enregistrer.pack(pady=10)

btn_supprimer = ctk.CTkButton(root, text="Supprimer un visage", command=supprimer_visage)
btn_supprimer.pack(pady=10)

btn_quitter = ctk.CTkButton(root, text="Quitter", command=root.quit)
btn_quitter.pack(pady=10)

# Label pour afficher le nombre de visages détectés
nombre_visages = tk.StringVar()
nombre_visages.set("Visages détectés : 0")
label_visages = ctk.CTkLabel(root, textvariable=nombre_visages, font=("Arial", 14))
label_visages.pack(pady=10)

root.mainloop()