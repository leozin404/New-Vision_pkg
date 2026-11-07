import cv2
import numpy as np

def drawBallBox(img, results, ball_index):
    ball_pos =  np.array([])
    img_cp = img.copy()

    if ball_index == -1:
        return img_cp, ball_pos
    
    ball_box_xywh = results.boxes[ball_index].xywhn.numpy() #get the most conf ball detect box in xyxy format
    array_box_xywh = np.reshape(ball_box_xywh, -1)  #convert matriz to array

    ball_pos = np.array([0, 0], dtype=float)
    ball_pos[0] = float(array_box_xywh[0] * img.shape[1])
    ball_pos[1] = float(array_box_xywh[1] * img.shape[0])
    
    raio_ball   = int((array_box_xywh[2] * img.shape[1] +array_box_xywh[3] * img.shape[0]) / 4)

    cv2.circle(img_cp, (int(ball_pos[0]), int(ball_pos[1])), abs(raio_ball), (255, 0, 0), 2)
    cv2.circle(img_cp, (int(ball_pos[0]), int(ball_pos[1])), 5, (255, 0, 0), -1)

    return img_cp, ball_pos

def drawGoalpostBox(img, results, goalpost_indices):
    img_cp = img.copy()
    goalpost_positions = []

    for idx in goalpost_indices:
        goalpost_box_xywh = results.boxes[idx].xywhn.numpy()
        array_box_xywh = np.reshape(goalpost_box_xywh, -1)

        center_x = int(array_box_xywh[0] * img.shape[1])
        center_y = int(array_box_xywh[1] * img.shape[0])
        width = int(array_box_xywh[2] * img.shape[1])
        height = int(array_box_xywh[3] * img.shape[0])

        x1 = int(center_x - width / 2)
        y1 = int(center_y - height / 2)
        x2 = int(center_x + width / 2)
        y2 = int(center_y + height / 2)

        cv2.rectangle(img_cp, (x1, y1), (x2, y2), (0, 255, 255), 2)  # amarelo
        cv2.circle(img_cp, (center_x, center_y), 5, (0, 255, 255), -1)

        goalpost_positions.append(np.array([center_x, center_y], dtype=float))

    return img_cp, goalpost_positions

def drawLBox(img, results, l_indices):
    """Desenha caixas para landmarks L"""
    img_cp = img.copy()
    l_positions = []
    
    for idx in l_indices:
        l_box_xywh = results.boxes[idx].xywhn.numpy()
        array_box_xywh = np.reshape(l_box_xywh, -1)
        
        center_x = int(array_box_xywh[0] * img.shape[1])
        center_y = int(array_box_xywh[1] * img.shape[0])
        width = int(array_box_xywh[2] * img.shape[1])
        height = int(array_box_xywh[3] * img.shape[0])
        
        x1 = int(center_x - width / 2)
        y1 = int(center_y - height / 2)
        x2 = int(center_x + width / 2)
        y2 = int(center_y + height / 2)
        
        # Desenha retângulo VERDE para L
        cv2.rectangle(img_cp, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(img_cp, (center_x, center_y), 5, (0, 255, 0), -1)
        
        l_positions.append(np.array([center_x, center_y], dtype=float))
    
    return img_cp, l_positions

def drawTBox(img, results, t_indices):
    """Desenha caixas para landmarks T - COR ROXA"""
    img_cp = img.copy()
    t_positions = []
    
    for idx in t_indices:
        t_box_xywh = results.boxes[idx].xywhn.numpy()
        array_box_xywh = np.reshape(t_box_xywh, -1)
        
        center_x = int(array_box_xywh[0] * img.shape[1])
        center_y = int(array_box_xywh[1] * img.shape[0])
        width = int(array_box_xywh[2] * img.shape[1])
        height = int(array_box_xywh[3] * img.shape[0])
        
        x1 = int(center_x - width / 2)
        y1 = int(center_y - height / 2)
        x2 = int(center_x + width / 2)
        y2 = int(center_y + height / 2)
        
        # ROXO (255, 0, 255)
        cv2.rectangle(img_cp, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.circle(img_cp, (center_x, center_y), 5, (255, 0, 255), -1)
        
        t_positions.append(np.array([center_x, center_y], dtype=float))
    
    return img_cp, t_positions

def drawXBox(img, results, x_indices):
    """Desenha caixas para landmarks X - COR LARANJA"""
    img_cp = img.copy()
    x_positions = []
    
    for idx in x_indices:
        x_box_xywh = results.boxes[idx].xywhn.numpy()
        array_box_xywh = np.reshape(x_box_xywh, -1)
        
        center_x = int(array_box_xywh[0] * img.shape[1])
        center_y = int(array_box_xywh[1] * img.shape[0])
        width = int(array_box_xywh[2] * img.shape[1])
        height = int(array_box_xywh[3] * img.shape[0])
        
        x1 = int(center_x - width / 2)
        y1 = int(center_y - height / 2)
        x2 = int(center_x + width / 2)
        y2 = int(center_y + height / 2)
        
        # LARANJA (0, 165, 255)
        cv2.rectangle(img_cp, (x1, y1), (x2, y2), (0, 165, 255), 2)
        cv2.circle(img_cp, (center_x, center_y), 5, (0, 165, 255), -1)
        
        x_positions.append(np.array([center_x, center_y], dtype=float))
    
    return img_cp, x_positions

class BoxSmoother:
    """Suaviza uma bounding box ao longo dos frames. Usado só para o desenho,
    não altera o que é publicado nos tópicos."""

    def __init__(self, alpha=0.4, hold_frames=5):
        self.alpha = alpha              # 0..1, maior = segue mais rápido
        self.hold_frames = hold_frames  # quantos frames mantém a box se a detecção piscar
        self.box = None                 # [cx, cy, w, h] em pixels
        self.missed = 0

    def update(self, box):
        if box is None:
            self.missed += 1
            if self.missed > self.hold_frames:
                self.box = None
            return self.box

        box = np.asarray(box, dtype=float)
        if self.box is None:
            self.box = box
        else:
            # Se pulou muito, reinicia em vez de "arrastar" a box pela tela
            jump = np.hypot(box[0] - self.box[0], box[1] - self.box[1])
            if jump > 0.5 * max(self.box[2], self.box[3]):
                self.box = box
            else:
                self.box = self.alpha * box + (1 - self.alpha) * self.box

        self.missed = 0
        return self.box

def _box_px(results, idx, img):
    """Retorna (cx, cy, w, h) em pixels da imagem de saída."""
    cx, cy, w, h = np.reshape(results.boxes[idx].xywhn.numpy(), -1)
    return (float(cx * img.shape[1]), float(cy * img.shape[0]),
            float(w * img.shape[1]), float(h * img.shape[0]))

def _touches_edge(x1, y1, x2, y2, img, margin=3):
    h, w = img.shape[:2]
    return x1 <= margin or y1 <= margin or x2 >= w - 1 - margin or y2 >= h - 1 - margin

def _draw_corner_box(img, x1, y1, x2, y2, color, thickness=2, frac=0.2):
    """Desenha só os cantos da box (mais limpo que o retângulo inteiro)."""
    lx = max(int((x2 - x1) * frac), 4)
    ly = max(int((y2 - y1) * frac), 4)
    for px, py, dx, dy in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]:
        cv2.line(img, (px, py), (px + dx * lx, py), color, thickness, cv2.LINE_AA)
        cv2.line(img, (px, py), (px, py + dy * ly), color, thickness, cv2.LINE_AA)

def drawElipseBox(img, results, elipse_indices, smoother=None):
    """Desenha a elipse do círculo central - COR CIANO.
    Elipse inteira na imagem: desenha a elipse.
    Elipse cortada pela borda: desenha só os cantos da box."""
    img_cp = img.copy()
    elipse_positions = []
    color = (255, 255, 0)

    best_idx, best_conf = None, -1.0
    for idx in elipse_indices:
        cx, cy, w, h = _box_px(results, idx, img)
        elipse_positions.append(np.array([cx, cy], dtype=float))

        conf = float(results.boxes[idx].conf)
        if conf > best_conf:
            best_idx, best_conf = idx, conf

    raw = _box_px(results, best_idx, img) if best_idx is not None else None
    shown = smoother.update(raw) if smoother is not None else raw

    if shown is None:
        return img_cp, elipse_positions

    cx, cy, w, h = shown
    x1, y1 = int(cx - w / 2), int(cy - h / 2)
    x2, y2 = int(cx + w / 2), int(cy + h / 2)

    if _touches_edge(x1, y1, x2, y2, img):
        _draw_corner_box(img_cp, x1, y1, x2, y2, color)
    else:
        cv2.ellipse(img_cp, (int(cx), int(cy)), (max(int(w / 2), 1), max(int(h / 2), 1)),
                    0, 0, 360, color, 2, cv2.LINE_AA)

    cv2.circle(img_cp, (int(cx), int(cy)), 3, color, -1, cv2.LINE_AA)

    return img_cp, elipse_positions

def resize_image(img, scale_percent=100):

    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    
    return resized

def findBall(img, results, classesValues):
    img_cp = img.copy()
    
    # Return the most confidence ball index, if ball not found index equal -1
    try:
        ball_index = np.where(results.boxes.cls == classesValues['ball'])[0][0]
    except: 
        ball_index = -1
    
    img_cp, ballPositionPx = drawBallBox(img_cp, results, ball_index)
        
    return img_cp, ballPositionPx
 
def findGoalpost(img, results, classesValues):
    img_cp = img.copy()

    try:
        goalpost_indices = np.where(results.boxes.cls == classesValues['goalpost'])[0]
    except:
        goalpost_indices = []

    img_cp, goalpost_positions = drawGoalpostBox(img_cp, results, goalpost_indices)
    
    return img_cp, goalpost_positions

def findL(img, results, classesValues):
    img_cp = img.copy()
    try:
        l_indices = np.where(results.boxes.cls == classesValues['L'])[0]
    except:
        l_indices = []
    
    img_cp, l_positions = drawLBox(img_cp, results, l_indices)
    return img_cp, l_positions

def findT(img, results, classesValues):
    img_cp = img.copy()
    try:
        t_indices = np.where(results.boxes.cls == classesValues['T'])[0]
    except:
        t_indices = []
    
    img_cp, t_positions = drawTBox(img_cp, results, t_indices)
    return img_cp, t_positions

def findX(img, results, classesValues):
    img_cp = img.copy()
    try:
        x_indices = np.where(results.boxes.cls == classesValues['X'])[0]
    except:
        x_indices = []
    
    img_cp, x_positions = drawXBox(img_cp, results, x_indices)
    return img_cp, x_positions

def findElipse(img, results, classesValues, smoother=None):
    img_cp = img.copy()

    # Busca a classe sem diferenciar maiúscula/minúscula ('Elipse', 'elipse', ...)
    key = next((k for k in classesValues if k.lower() == 'elipse'), None)

    if key is None:
        elipse_indices = []
    else:
        elipse_indices = np.where(results.boxes.cls == classesValues[key])[0]
    
    img_cp, elipse_positions = drawElipseBox(img_cp, results, elipse_indices, smoother)
    return img_cp, elipse_positions
