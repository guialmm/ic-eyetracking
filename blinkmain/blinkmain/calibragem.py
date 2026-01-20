import json
import time
import cv2
import numpy as np
from env.gazetracking.gaze_tracking import GazeTracking
import os
import tkinter as tk
import matplotlib.pyplot as plt

gaze = GazeTracking()

# Caminhos relativos ao diretório atual
METRICAS_PATH = "metricas"
RESULTADOS_PATH = "resultados"
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
    # Função para inicializar câmera de forma mais robusta
    def init_camera_calibration():
        for i in range(3):  # Tentar até 3 vezes
            webcam = cv2.VideoCapture(0)
            if webcam.isOpened():
                # Configurar a câmera
                webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                webcam.set(cv2.CAP_PROP_FPS, 30)
                return webcam
            else:
                webcam.release()
                time.sleep(1)  # Aguardar 1 segundo antes de tentar novamente
        return None
    
    webcam = init_camera_calibration()
    if webcam is None:
        print("Erro: Não foi possível abrir a câmera para captura após múltiplas tentativas.")
        return None

    samples = []
    start_time = time.time()

    try:
        while time.time() - start_time < duration:
            ret, frame = webcam.read()
            if not ret:
                print("Erro ao capturar frame durante calibração")
                break

            gaze.refresh(frame)
            annotated_frame = gaze.annotated_frame()
            cv2.imshow("Calibração - Verifique se o rosto está visível", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            gaze_x = gaze.horizontal_ratio()
            gaze_y = gaze.vertical_ratio()
            if gaze_x is not None and gaze_y is not None:
                samples.append((gaze_x, gaze_y))
    
    finally:
        # Limpeza adequada
        try:
            webcam.release()
            cv2.destroyAllWindows()
            time.sleep(0.2)
            cv2.waitKey(1)
        except:
            pass
    return samples if samples else None

def calibrar():
    # 5 pontos de calibração: 4 cantos internos e 1 no centro
    pontos_calibracao = [
        (0.2, 0.2), (0.8, 0.2),
        (0.2, 0.8), (0.8, 0.8),
        (0.5, 0.5)
    ]

    resultados = []
    todas_as_etapas = []

    cores_etapas = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255)
    ]

    print("Iniciando calibrações consecutivas...")
    for tentativa in range(3):
        print(f"\nCalibração {tentativa + 1} de 3...")
        all_x = []
        all_y = []
        clusters_por_tentativa = []

        cor_etapa_atual = cores_etapas[tentativa]

        for idx, (x, y) in enumerate(pontos_calibracao):
            draw_marker(x, y, 30, cor_etapa_atual)
            time.sleep(1)
            samples = capture(3)

            if samples is None:
                print("Erro: Não foi possível capturar dados para o ponto de calibração.")
                continue

            valid_samples_x = [s[0] for s in samples]
            valid_samples_y = [s[1] for s in samples]

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

            print(
                f"Resultado da calibração {tentativa + 1}: x_min={resultado_final[0]}, x_max={resultado_final[1]}, y_min={resultado_final[2]}, y_max={resultado_final[3]}"
            )

            todas_as_etapas.append(clusters_por_tentativa)
        else:
            print(f"Erro: A calibração {tentativa + 1} falhou devido à falta de dados.")
            return None

    salvar_metricas_e_imagem(todas_as_etapas)

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

def salvar_metricas_e_imagem(todas_as_etapas):
    cores = ['blue', 'orange', 'green', 'brown', 'purple']  # 5 cores para 5 pontos
    nomes = ['Ponto 1', 'Ponto 2', 'Ponto 3', 'Ponto 4', 'Ponto 5']

    plt.figure(figsize=(10, 8))
    todos_x = []
    todos_y = []

    for i, etapa in enumerate(todas_as_etapas):  # 3 calibrações
        if etapa:
            for j, cluster in enumerate(etapa):  # 5 pontos
                xs = [p[0] for p in cluster]
                ys = [p[1] for p in cluster]
                plt.scatter(xs, ys, c=cores[j], label=nomes[j] if i == 0 else "", alpha=0.8)
                todos_x.extend(xs)
                todos_y.extend(ys)

    # Centrôide geral
    cx = np.mean(todos_x)
    cy = np.mean(todos_y)
    plt.scatter(cx, cy, c='black', s=200, marker='x', label='Centróide Geral')

    plt.title("Calibração Final com 5 Pontos")
    plt.xlabel("Horizontal Ratio")
    plt.ylabel("Vertical Ratio")
    plt.grid(True)
    plt.legend()

    caminho_img = os.path.join(RESULTADOS_PATH, "metricas_5_pontos.png")
    plt.savefig(caminho_img)
    plt.close()