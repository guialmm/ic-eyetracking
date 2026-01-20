import cv2
import pyautogui
import numpy as np
from env.gazetracking.gaze_tracking import GazeTracking

# Inicializar rastreamento ocular
gaze = GazeTracking()
pyautogui.FAILSAFE = False

# Inicializar câmera
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Erro: Não foi possível acessar a câmera.")
    exit()

# Obter dimensões da tela
screen_w, screen_h = pyautogui.size()

while True:
    ret, frame = cam.read()
    if not ret:
        print("Falha ao capturar frame da câmera.")
        break

    # Atualizar rastreamento ocular
    gaze.refresh(frame)
    gaze_x = gaze.horizontal_ratio()
    gaze_y = gaze.vertical_ratio()
    annotated_frame = gaze.annotated_frame()

    if gaze_x is not None and gaze_y is not None:
        # Garantir que os valores estejam dentro do intervalo [0, 1]
        gaze_x = np.clip(gaze_x, 0.0, 1.0)
        gaze_y = np.clip(gaze_y, 0.0, 1.0)

        # Calcular posição do mouse diretamente com base nas coordenadas do olhar
        target_x = int(gaze_x * screen_w)
        target_y = int(gaze_y * screen_h)

        # Mover o mouse para a posição calculada
        pyautogui.moveTo(target_x, target_y)

        # Exibir valores no console
        print(f"Gaze detectado: gaze_x={gaze_x}, gaze_y={gaze_y}")

    # Mostrar o frame anotado
    cv2.imshow('Eye Controlled Mouse - Demonstração', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cam.release()
cv2.destroyAllWindows()
print("Programa encerrado. Câmera liberada.")