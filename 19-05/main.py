import cv2
import pyautogui
import time
from calibragem import calibrar
from env.gazetracking.gaze_tracking import GazeTracking
import collections
import numpy as np

pyautogui.FAILSAFE = False

# Inicializa a câmera e o objeto GazeTracking
gaze = GazeTracking()
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Erro: Não conseguiu abrir a câmera padrão.")
    exit()


# Realiza a calibração e obtém os limites
print("Iniciando calibragem...")
resultado_calibracao = calibrar()
if not resultado_calibracao:
    print("Calibração falhou. Verifique o ambiente e tente novamente.")
    exit()

x_min, x_max, y_min, y_max = resultado_calibracao
print(f"Calibração concluída: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}")
print("Iniciando rastreamento...")

# Obtém as dimensões da tela
screen_w, screen_h = pyautogui.size()

# Inicializa deque para rastreamento suave
window_size = 10
positions = collections.deque(maxlen=window_size)

# Funções para o cálculo da nova métrica
def calculate_distance(C, P):
    return np.sqrt((P[0] - C[0])**2 + (P[1] - C[1])**2)

def calculate_v_red(C, P, D):
    dist = calculate_distance(C, P)
    return 2 * np.arctan(dist / (2 * D))

def calculate_v_deg(C, P, D):
    v_red = calculate_v_red(C, P, D)
    return (v_red / (2 * np.pi)) * 360

def calculate_accuracy(C, P_list, D):
    v_deg_values = [calculate_v_deg(C, P, D) for P in P_list]
    return np.mean(v_deg_values)

# Parâmetro de distância (D) utilizado na métrica
D = 10  # Valor arbitrário para D

while True:
    ret, frame = cam.read()
    if not ret:
        print("Falha ao capturar frame da câmera.")
        break

    gaze.refresh(frame)

    gaze_x = gaze.horizontal_ratio()
    gaze_y = gaze.vertical_ratio()

    annotated_frame = gaze.annotated_frame()

    if gaze_x is not None and gaze_y is not None:
        positions.append((gaze_x, gaze_y))

        if len(positions) > 1:
            avg_x = np.mean([p[0] for p in positions])
            avg_y = np.mean([p[1] for p in positions])
        else:
            avg_x, avg_y = gaze_x, gaze_y

        normalized_x = (avg_x - x_min) / (x_max - x_min)
        normalized_y = (avg_y - y_min) / (y_max - y_min)

        screen_x = normalized_x * screen_w
        screen_y = normalized_y * screen_h

        pyautogui.moveTo(screen_x, screen_y)

        accuracy = calculate_accuracy((0.5, 0.5), positions, D)
        cv2.putText(annotated_frame, f"Acurácia: {accuracy:.2f} graus", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(annotated_frame, f"Gaze X: {gaze_x:.2f}, Gaze Y: {gaze_y:.2f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2)
    else:
        cv2.putText(annotated_frame, "Gaze not detected", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Eye Controlled Mouse', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
