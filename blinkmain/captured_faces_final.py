import cv2
import os
import time
from env.gazetracking.gaze_tracking import GazeTracking

# Inicializa o gaze tracking
gaze = GazeTracking()

# Caminho onde as imagens serão salvas
output_folder = r"......."
os.makedirs(output_folder, exist_ok=True)

# Inicializa a câmera
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Erro: Não foi possível abrir a câmera.")
    exit()

last_capture_time = time.time()
img_counter = 0  # Continuar a partir do último número de imagem

print("Iniciando captura de imagens usando GazeTracking... Pressione 'q' para parar.")

while True:
    ret, frame = cam.read()
    if not ret:
        print("Erro ao capturar imagem.")
        break

    # Garante que o frame está no formato correto
    if frame is None or frame.dtype != 'uint8' or len(frame.shape) != 3 or frame.shape[2] != 3:
        print("Erro: Frame inválido para análise.")
        continue

    # Frame BGR original é o ideal para o GazeTracking
    gaze.refresh(frame)

    # Obtém o frame com anotações (desenhos do gaze tracking)
    gaze.frame = frame.copy()
    annotated_frame = gaze.annotated_frame()

    # Exibe o frame na janela
    cv2.imshow("GazeTracking - Captura de Faces", annotated_frame)

    # Salva a cada 1 segundo
    if time.time() - last_capture_time >= 1:
        img_name = f"face_{img_counter}.jpg"
        img_path = os.path.join(output_folder, img_name)
        cv2.imwrite(img_path, annotated_frame)
        print(f"[INFO] Imagem salva: {img_name}")
        img_counter += 1
        last_capture_time = time.time()

    # Tecla 'q' para parar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
print("Captura finalizada.")
