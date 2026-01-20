import cv2
from env.gazetracking.gaze_tracking import GazeTracking

# Inicializa o objeto GazeTracking
gaze = GazeTracking()

# Inicializa a câmera
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Erro: Não foi possível acessar a câmera.")
    exit()

while True:
    ret, frame = cam.read()
    if not ret:
        print("Erro ao capturar frame da câmera.")
        break

    # Atualiza o rastreamento do olhar
    gaze.refresh(frame)

    # Obtém o frame anotado com a máscara do GazeTracking
    annotated_frame = gaze.annotated_frame()

    # Exibe o frame com a máscara
    cv2.imshow('Mascara GazeTracking', annotated_frame)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera os recursos
cam.release()
cv2.destroyAllWindows()