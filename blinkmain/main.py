import cv2
import pyautogui
import time
from calibragem import calibrar
from env.gazetracking.gaze_tracking import GazeTracking
import collections
import numpy as np
import os

# Tentar importar PyTorch para usar os modelos .pth
try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
    print("PyTorch disponivel - modelos .pth serao carregados")
except ImportError:
    PYTORCH_AVAILABLE = False
    print("PyTorch nao disponivel - usando rastreamento padrao")

# Classe simples para modelo de eye tracking
class SimpleEyeModel(nn.Module):
    def _init_(self):
        super(SimpleEyeModel, self)._init_()
        self.fc1 = nn.Linear(2, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Função para carregar modelo .pth
def load_eye_model():
    if not PYTORCH_AVAILABLE:
        return None
    
    model_paths = [
        "melhor_modelo_eyetracking_CORRIGIDO.pth",
        "modelo_eyetracking_LR_FIXO.pth"
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                model = SimpleEyeModel()
                checkpoint = torch.load(model_path, map_location='cpu')
                
                # Tentar diferentes formatos de checkpoint
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    elif 'state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['state_dict'])
                    else:
                        # Tentar usar o próprio dict como state_dict
                        try:
                            model.load_state_dict(checkpoint)
                        except:
                            print(f"Formato de checkpoint nao reconhecido: {model_path}")
                            continue
                else:
                    model.load_state_dict(checkpoint)
                
                model.eval()
                print(f"Modelo carregado com sucesso: {model_path}")
                return model
            except Exception as e:
                print(f"Erro ao carregar {model_path}: {e}")
                continue
    
    print("Nenhum modelo valido encontrado")
    return None

# Função para melhorar predição usando modelo
def enhance_gaze_with_model(model, gaze_x, gaze_y):
    if model is None or gaze_x is None or gaze_y is None:
        return gaze_x, gaze_y
    
    try:
        input_tensor = torch.tensor([gaze_x, gaze_y], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            enhanced_x = output[0][0].item()
            enhanced_y = output[0][1].item()
        return enhanced_x, enhanced_y
    except Exception as e:
        print(f"Erro na predicao do modelo: {e}")
        return gaze_x, gaze_y

# Carregar modelo se disponível
eye_model = load_eye_model() if PYTORCH_AVAILABLE else None

pyautogui.FAILSAFE = False

# Inicializa a câmera e o objeto GazeTracking
gaze = GazeTracking()

# Função para inicializar câmera de forma mais robusta
def init_camera():
    for i in range(3):  # Tentar até 3 vezes
        cam = cv2.VideoCapture(0)
        if cam.isOpened():
            # Configurar a câmera
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cam.set(cv2.CAP_PROP_FPS, 30)
            return cam
        else:
            cam.release()
            time.sleep(1)  # Aguardar 1 segundo antes de tentar novamente
    return None

cam = init_camera()
if cam is None:
    print("Erro: Nao conseguiu abrir a camera após multiplas tentativas.")
    print("Dicas:")
    print("1. Verifique se a camera nao esta sendo usada por outro programa")
    print("2. Tente fechar outros programas que possam estar usando a camera")
    print("3. Reinicie o computador se necessario")
    exit()


# Realiza a calibração e obtém os limites
print("Iniciando calibragem...")
resultado_calibracao = calibrar()
if not resultado_calibracao:
    print("Calibracao falhou. Verifique o ambiente e tente novamente.")
    exit()

x_min, x_max, y_min, y_max = resultado_calibracao
print(f"Calibracao concluida: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}")
print("Iniciando rastreamento...")

# Obtém as dimensões da tela
screen_w, screen_h = pyautogui.size()

# Inicializa deque para rastreamento suave
window_size = 15  # Aumentado para mais suavização
positions = collections.deque(maxlen=window_size)

# Funções para o cálculo da nova métrica
def calculate_distance(C, P):
    return np.sqrt((P[0] - C[0])*2 + (P[1] - C[1])*2)

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

# Variáveis para suavização simples
prev_gaze_x = None
prev_gaze_y = None
smoothing_factor = 0.8  # Aumentado para mais suavização

# Função de suavização simples
def smooth_gaze(current_x, current_y, prev_x, prev_y, factor=0.7):
    if prev_x is None or prev_y is None:
        return current_x, current_y
    if current_x is None or current_y is None:
        return prev_x, prev_y
    
    smooth_x = factor * current_x + (1 - factor) * prev_x
    smooth_y = factor * current_y + (1 - factor) * prev_y
    return smooth_x, smooth_y

while True:
    ret, frame = cam.read()
    if not ret:
        print("Falha ao capturar frame da camera.")
        break

    gaze.refresh(frame)

    gaze_x = gaze.horizontal_ratio()
    gaze_y = gaze.vertical_ratio()

    annotated_frame = gaze.annotated_frame()

    if gaze_x is not None and gaze_y is not None:
        # Aplicar suavização simples
        smooth_x, smooth_y = smooth_gaze(gaze_x, gaze_y, prev_gaze_x, prev_gaze_y, smoothing_factor)
        prev_gaze_x, prev_gaze_y = smooth_x, smooth_y
        
        # Aplicar modelo PyTorch se disponível
        enhanced_x, enhanced_y = enhance_gaze_with_model(eye_model, smooth_x, smooth_y)
        
        positions.append((enhanced_x, enhanced_y))

        if len(positions) > 1:
            avg_x = np.mean([p[0] for p in positions])
            avg_y = np.mean([p[1] for p in positions])
        else:
            avg_x, avg_y = enhanced_x, enhanced_y

        # Normalização com clipping e zona morta
        normalized_x = (avg_x - x_min) / (x_max - x_min) if (x_max - x_min) != 0 else 0.5
        normalized_y = (avg_y - y_min) / (y_max - y_min) if (y_max - y_min) != 0 else 0.5
        
        # Aplicar clipping para evitar valores extremos
        normalized_x = max(0.0, min(1.0, normalized_x))
        normalized_y = max(0.0, min(1.0, normalized_y))
        
        # Zona morta no centro para reduzir tremulação
        dead_zone = 0.05
        if abs(normalized_x - 0.5) < dead_zone:
            normalized_x = 0.5
        if abs(normalized_y - 0.5) < dead_zone:
            normalized_y = 0.5
        
        # Reduzir sensibilidade nos extremos
        def ease_movement(value):
            if value < 0.1:
                return value * 0.5 + 0.05
            elif value > 0.9:
                return (value - 0.9) * 0.5 + 0.9
            else:
                return value
        
        normalized_x = ease_movement(normalized_x)
        normalized_y = ease_movement(normalized_y)
        
        # Adicionar margem na tela para evitar cantos
        margin_x = screen_w * 0.05  # 5% de margem
        margin_y = screen_h * 0.05  # 5% de margem
        
        screen_x = margin_x + normalized_x * (screen_w - 2 * margin_x)
        screen_y = margin_y + normalized_y * (screen_h - 2 * margin_y)
        
        # Movimento suavizado do mouse
        current_mouse_x, current_mouse_y = pyautogui.position()
        
        # Calcular a diferença
        diff_x = screen_x - current_mouse_x
        diff_y = screen_y - current_mouse_y
        
        # Aplicar threshold de movimento mínimo
        movement_threshold = 10
        if abs(diff_x) > movement_threshold or abs(diff_y) > movement_threshold:
            # Movimento gradual - não vai direto para a posição
            move_factor = 0.3  # Quão rápido o mouse se move (0.1 = muito lento, 1.0 = instantâneo)
            new_x = current_mouse_x + diff_x * move_factor
            new_y = current_mouse_y + diff_y * move_factor
            
            pyautogui.moveTo(new_x, new_y)

        accuracy = calculate_accuracy((0.5, 0.5), positions, D)
        cv2.putText(annotated_frame, f"Acuracia: {accuracy:.2f} graus", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Mostrar status do modelo
        if eye_model is not None:
            cv2.putText(annotated_frame, "Modelo PyTorch: ATIVO", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Enhanced: {enhanced_x:.2f}, {enhanced_y:.2f}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(annotated_frame, "Modelo PyTorch: INATIVO", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(annotated_frame, f"Suavizacao: ATIVA", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(annotated_frame, f"Gaze Raw: {gaze_x:.2f}, {gaze_y:.2f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Normalized: {normalized_x:.2f}, {normalized_y:.2f}", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
        cv2.putText(annotated_frame, f"Screen: {int(screen_x)}, {int(screen_y)}", (20, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
    else:
        cv2.putText(annotated_frame, "Gaze not detected", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Eye Controlled Mouse', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Limpeza adequada
try:
    cam.release()
    cv2.destroyAllWindows()
    # Forçar liberação da câmera
    time.sleep(0.5)
    cv2.waitKey(1)
except:
    pass

print("Programa encerrado. Camera liberada.")