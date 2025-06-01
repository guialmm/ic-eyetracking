import json
import time
import cv2
import numpy as np
from env.gazetracking.gaze_tracking import GazeTracking
import os
import tkinter as tk
import matplotlib.pyplot as plt

gaze = GazeTracking()

METRICAS_PATH = r"C:\Users\guial\Documents\pythonProject\Blink-main\metricas"
RESULTADOS_PATH = r"C:\Users\guial\Documents\pythonProject\Blink-main\resultados"
os.makedirs(METRICAS_PATH, exist_ok=True)
os.makedirs(RESULTADOS_PATH, exist_ok=True)

def obter_resolucao_tela():
    root = tk.Tk()
    root.withdraw()
    return root.winfo_screenwidth(), root.winfo_screenheight()

def draw_marker(x, y, radius=30, color=(255, 255, 255)):
    width, height = obter_resolucao_tela()
    pos_x = int(x * width)
    pos_y = int(y * height)

    window_name = "Calibration"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.circle(frame, (pos_x, pos_y), radius, color, -1)

    cv2.imshow(window_name, frame)
    cv2.waitKey(1)

def capture(duration):
    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        print("Erro: Não foi possível abrir a câmera para captura.")
        return None

    samples = []
    start_time = time.time()

    while time.time() - start_time < duration:
        ret, frame = webcam.read()
        if not ret:
            webcam.release()
            return None

        gaze.refresh(frame)
        annotated_frame = gaze.annotated_frame()
        cv2.imshow("Calibração - Verifique se o rosto está visível", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            webcam.release()
            cv2.destroyAllWindows()
            return None

        gaze_x = gaze.horizontal_ratio()
        gaze_y = gaze.vertical_ratio()
        if gaze_x is not None and gaze_y is not None:
            samples.append((gaze_x, gaze_y))

    webcam.release()
    cv2.destroyAllWindows()
    return samples if samples else None

def calibrar():
    pontos_calibracao = [
        (0.0, 0.0), (0.5, 0.0), (1.0, 0.0),   # linha superior
        (0.0, 0.5), (0.5, 0.5), (1.0, 0.5),   # linha do meio
        (0.0, 1.0), (0.5, 1.0), (1.0, 1.0)    # linha inferior
    ]

    print("Iniciando calibração com 9 pontos...")
    all_x = []
    all_y = []
    clusters = []

    for x, y in pontos_calibracao:
        draw_marker(x, y, 30, (255, 255, 255))
        time.sleep(1)
        samples = capture(3)

        if samples is None:
            print("Erro: Não foi possível capturar dados para o ponto de calibração.")
            continue

        valid_samples_x = [sample[0] for sample in samples]
        valid_samples_y = [sample[1] for sample in samples]

        if valid_samples_x and valid_samples_y:
            avg_x = np.mean(valid_samples_x)
            avg_y = np.mean(valid_samples_y)
            all_x.append(avg_x)
            all_y.append(avg_y)
            clusters.append(samples)
        else:
            print("Erro: Nenhum dado válido foi coletado para o ponto de calibração.")

    if all_x and all_y:
        x_min = min(all_x)
        x_max = max(all_x)
        y_min = min(all_y)
        y_max = max(all_y)

        fator_suavizacao = 0.1
        resultado_final = [
            x_max, x_min,
            y_min - fator_suavizacao * (y_max - y_min),
            y_max + fator_suavizacao * (y_max - y_min)
        ]

        print(f"\nResultado da calibração: x_min={resultado_final[0]}, x_max={resultado_final[1]}, y_min={resultado_final[2]}, y_max={resultado_final[3]}")

        salvar_metricas_e_imagem(clusters)
        return resultado_final
    else:
        print("Erro: A calibração falhou devido à falta de dados.")
        return None

def salvar_metricas_e_imagem(clusters):
    todos_pontos = [p for cluster in clusters for p in cluster]
    xs_all = [p[0] for p in todos_pontos]
    ys_all = [p[1] for p in todos_pontos]
    centroide_geral = [np.mean(xs_all), np.mean(ys_all)]

    cores = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'cyan', 'magenta']

    plt.figure(figsize=(8, 6))
    for i, cluster in enumerate(clusters):
        xs = [p[0] for p in cluster]
        ys = [p[1] for p in cluster]
        plt.scatter(xs, ys, c=cores[i % len(cores)], label=f'Ponto {i+1}')

    plt.scatter(centroide_geral[0], centroide_geral[1], c='black', s=200, marker='x', label='Centróide Geral')

    plt.title("Calibração Final com 9 Pontos")
    plt.xlabel("Horizontal Ratio")
    plt.ylabel("Vertical Ratio")
    plt.legend()
    plt.grid(True)

    caminho_img = os.path.join(RESULTADOS_PATH, "calibracao_final_9pontos.png")
    plt.savefig(caminho_img)
    plt.close()
