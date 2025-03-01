import os
import cv2 as cv
import numpy as np

# Carregar classificadores Haar para rosto frontal e perfil
face_classifier_frontal = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
face_classifier_profile = cv.CascadeClassifier("haarcascade_profileface.xml")

if face_classifier_frontal.empty() or face_classifier_profile.empty():
    print("Erro ao carregar arquivos Haar Cascade. Verifique os caminhos.")
    exit()

recognizer = cv.face.LBPHFaceRecognizer_create()

# Pasta com várias fotos suas
pasta_fotos = "fotografias"
imagens_treino = []
labels = []

# Carregar e processar todas as imagens da pasta
for filename in os.listdir(pasta_fotos):
    if filename.endswith((".png", ".jpg", ".jpeg")):
        caminho_imagem = os.path.join(pasta_fotos, filename)
        imagem = cv.imread(caminho_imagem, cv.IMREAD_GRAYSCALE)
        if imagem is None:
            print(f"Erro ao carregar {filename}")
            continue

        # Detectar rostos frontais
        faces_frontal = face_classifier_frontal.detectMultiScale(imagem, 1.3, 3, minSize=(30, 30))
        faces_profile = face_classifier_profile.detectMultiScale(imagem, 1.3, 3, minSize=(30, 30))

        # Combinar resultados (usar o primeiro rosto detectado)
        faces_ref = faces_frontal if len(faces_frontal) > 0 else faces_profile
        if len(faces_ref) == 0:
            print(f"Nenhum rosto detectado em {filename}")
            continue

        x, y, w, h = faces_ref[0]
        rosto = imagem[y:y+h, x:x+w]
        imagens_treino.append(rosto)
        labels.append(1)

if not imagens_treino:
    print("Nenhuma imagem válida para treinamento. Encerrando...")
    exit()

recognizer.train(imagens_treino, np.array(labels))

def detect_recognize(framee, gray_frame):
    # Detectar rostos frontais e de perfil
    faces_frontal = face_classifier_frontal.detectMultiScale(gray_frame, 1.3, 3, minSize=(30, 30))
    faces_profile = face_classifier_profile.detectMultiScale(gray_frame, 1.3, 3, minSize=(30, 30))

    # Combinar os rostos detectados (evitar duplicatas)
    faces = list(faces_frontal)
    for profile in faces_profile:
        # Verificar se o rosto de perfil já foi detectado como frontal
        overlap = False
        for frontal in faces_frontal:
            if abs(frontal[0] - profile[0]) < 20 and abs(frontal[1] - profile[1]) < 20:
                overlap = True
                break
        if not overlap:
            faces.append(profile)

    for (x, y, w, h) in faces:
        rosto_gray = gray_frame[y:y+h, x:x+w]
        label, confidence = recognizer.predict(rosto_gray)

        if label == 1 and confidence < 35:
            nome = "Eu"
            cor = (0, 255, 0)
        else:
            nome = "Desconhecido"
            cor = (0, 0, 255)

        cv.rectangle(framee, (x, y), (x+w, y+h), cor, 2)
        cv.putText(framee, f"{nome} ({int(confidence)})", (x, y - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.9, cor, 2)

    return framee

# Iniciar a captura de vídeo
cap = cv.VideoCapture(2)
if not cap.isOpened():
    print("Não foi possível abrir a câmera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Não foi possível receber o frame. Encerrando...")
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_processado = detect_recognize(frame, gray)
    cv.imshow('frame', frame_processado)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()