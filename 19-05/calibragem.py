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
    # Alteração principal: usar índice 0 para câmera integrada
    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        print("Erro: Não foi possível abrir a câmera para captura.")
        return None

    samples = []
    start_time = time.time()

    while time.time() - start_time < duration:
        ret, frame = webcam.read()
        if not ret:
            print("Erro: Não foi possível acessar a câmera durante a captura.")
            webcam.release()
            return None

        gaze.refresh(frame)
        annotated_frame = gaze.annotated_frame()
        cv2.imshow("Calibração - Verifique se o rosto está visível", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Calibração interrompida pelo usuário.")
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
        (0, 0), (1, 0), (0, 1), (1, 1), (0.5, 0.5)
    ]
    resultados = []

    print("Iniciando calibrações consecutivas...")
    for tentativa in range(3):
        print(f"\nCalibração {tentativa + 1} de 3...")
        all_x = []
        all_y = []
        clusters_por_tentativa = []

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
                clusters_por_tentativa.append(samples)
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
            resultados.append(resultado_final)

            print(f"Resultado da calibração {tentativa + 1}: x_min={resultado_final[0]}, x_max={resultado_final[1]}, y_min={resultado_final[2]}, y_max={resultado_final[3]}")

            if tentativa == 2:
                salvar_metricas_e_imagem(clusters_por_tentativa)

        else:
            print(f"Erro: A calibração {tentativa + 1} falhou devido à falta de dados.")
            return None

    if resultados:
        x_min = min([res[0] for res in resultados])
        x_max = max([res[1] for res in resultados])
        y_min = min([res[2] for res in resultados])
        y_max = max([res[3] for res in resultados])
        resultado_final = [x_min, x_max, y_min, y_max]

        os.makedirs('./cache', exist_ok=True)
        with open('./cache/calibragem.json', 'w') as f:
            json.dump(resultado_final, f, indent=2)

        print(f"\nCalibração final refinada: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}")
        return resultado_final
    else:
        print("Erro: Todas as calibrações falharam.")
        return None

def salvar_metricas_e_imagem(clusters):
    centróides = []
    for cluster in clusters:
        xs = [p[0] for p in cluster]
        ys = [p[1] for p in cluster]
        centroide = [np.mean(xs), np.mean(ys)]
        centróides.append(centroide)

    metricas_salvar = {
        "clusters": clusters,
        "centroids": centróides
    }

    arquivo_metricas = os.path.join(METRICAS_PATH, "metricas_calibragem.json")
    with open(arquivo_metricas, "w") as f:
        json.dump(metricas_salvar, f, indent=2)

    cores = ['blue', 'orange', 'green', 'red', 'purple']

    plt.figure(figsize=(8, 6))
    for i, cluster in enumerate(clusters):
        xs = [p[0] for p in cluster]
        ys = [p[1] for p in cluster]
        plt.scatter(xs, ys, c=cores[i], label=f'cluster {i}')

    for i, c in enumerate(centróides):
        plt.scatter(c[0], c[1], c='red', s=150, label='centroid' if i == 0 else "")

    plt.title("Calibração Final")
    plt.xlabel("Horizontal Ratio")
    plt.ylabel("Vertical Ratio")
    plt.legend()
    plt.grid(True)

    caminho_img = os.path.join(RESULTADOS_PATH, "calibração-final.png")
    plt.savefig(caminho_img)
    plt.close()
