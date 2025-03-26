import cv2
import numpy as np
import threading
import queue
import json
import os
import math
import matplotlib.pyplot as plt

# RTSP-адрес камеры
rtsp_url = "rtsp://admin:UrFU_ISIT@10.32.9.223:554/Streaming/channels/101"

# Инициализация параметров калибровки
fx = 1500.0
fy = 1470.0
cx = 980.0
cy = 810.0
k1 = -0.610
k2 = 0.280
p1 = 0.089
p2 = 0.008

# Список параметров для настройки
params = [
    {'name': 'fx', 'value': fx, 'step': 10.0},
    {'name': 'fy', 'value': fy, 'step': 10.0},
    {'name': 'cx', 'value': cx, 'step': 10.0},
    {'name': 'cy', 'value': cy, 'step': 10.0},
    {'name': 'k1', 'value': k1, 'step': 0.01},
    {'name': 'k2', 'value': k2, 'step': 0.01},
    {'name': 'p1', 'value': p1, 'step': 0.001},
    {'name': 'p2', 'value': p2, 'step': 0.001},
]

current_param = 0  # Индекс текущего параметра

# Параметры ROI (Region of Interest)
roi_enabled = False
roi_x, roi_y, roi_w, roi_h = 0, 0, 0, 0
selecting_roi = False
roi_start_point = (0, 0)

# Файл для сохранения настроек ROI
ROI_CONFIG_FILE = "roi_config.json"

# Параметры для A*
path_planning_enabled = False
start_point = None
end_point = None
obstacles = []
path = []
coordinates = []
show_animation = False

class AStarPath:
    def __init__(self, robot_radius, grid_size, x_obstacle, y_obstacle):
        self.grid_size = grid_size
        self.robot_radius = robot_radius
        self.create_obstacle_map(x_obstacle, y_obstacle)
        self.path = self.get_path()

    class Node:
        def __init__(self, x, y, cost, path):
            self.x = x
            self.y = y
            self.cost = cost
            self.path = path
        
        def __str__(self):
            return str(self.x)+'+'+str(self.y)+','+str(self.cost)+','+str(self.path)
    
    @staticmethod
    def calc_heuristic(num1, num2):
        weight = 1.0
        return weight * math.sqrt((num1.x-num2.x)**2 + (num1.y-num2.y)**2)

    def calc_grid_position(self, idx, p):
        return idx * self.grid_size + p

    def calc_xy(self, position, min_position):
        return round((position-min_position) / self.grid_size)
    
    def calc_grid_idx(self, node):
        return (node.y-self.y_min) * self.x_width + (node.x-self.x_min)
    
    def check_validity(self, node):
        x_position = self.calc_grid_position(node.x, self.x_min)
        y_position = self.calc_grid_position(node.y, self.y_min)

        if x_position < self.x_min:
            return False
        elif y_position < self.y_min:
            return False
        elif x_position >= self.x_max:
            return False
        elif y_position >= self.y_max:
            return False
        if self.obstacle_pos[node.x][node.y]:
            return False
        return True

    def create_obstacle_map(self, x_obstacle, y_obstacle):
        self.x_min = round(min(x_obstacle))
        self.y_min = round(min(y_obstacle))
        self.x_max = round(max(x_obstacle))
        self.y_max = round(max(y_obstacle))
        self.x_width = round((self.x_max - self.x_min) / self.grid_size)
        self.y_width = round((self.y_max - self.y_min) / self.grid_size)
        self.obstacle_pos = [[False for i in range(self.y_width)] for i in range(self.x_width)]
        for idx_x in range(self.x_width):
            x = self.calc_grid_position(idx_x, self.x_min)
            for idx_y in range(self.y_width):
                y = self.calc_grid_position(idx_y, self.y_min)
                for idx_x_obstacle, idx_y_obstacle in zip(x_obstacle, y_obstacle):
                    d = math.sqrt((idx_x_obstacle - x)**2 + (idx_y_obstacle - y)**2)
                    if d <= self.robot_radius:
                        self.obstacle_pos[idx_x][idx_y] = True
                        break
    
    @staticmethod
    def get_path():
        path = [[1, 0, 1],
                [0, 1, 1],
                [-1, 0, 1],
                [0, -1, 1],
                [-1, -1, math.sqrt(2)],
                [-1, 1, math.sqrt(2)],
                [1, -1, math.sqrt(2)],
                [1, 1, math.sqrt(2)]]
        return path

    def calc_final_path(self, end_node, record_closed):
        x_out_path, y_out_path = [self.calc_grid_position(end_node.x, self.x_min)], [self.calc_grid_position(end_node.y, self.y_min)]
        path = end_node.path
        while path != -1:
            n = record_closed[path]
            x_out_path.append(self.calc_grid_position(n.x, self.x_min))
            y_out_path.append(self.calc_grid_position(n.y, self.y_min))
            path = n.path

        return x_out_path, y_out_path

    def a_star_search(self, start_x, start_y, end_x, end_y):
        start_node = self.Node(self.calc_xy(start_x, self.x_min), self.calc_xy(start_y, self.y_min), 0.0, -1)
        end_node = self.Node(self.calc_xy(end_x, self.x_min), self.calc_xy(end_y, self.y_min), 0.0, -1)

        record_open, record_closed = dict(), dict()
        record_open[self.calc_grid_idx(start_node)] = start_node

        while True:
            if len(record_open) == 0:
                print('Check Record Validity')
                break

            total_cost = min(record_open, key=lambda x: record_open[x].cost + self.calc_heuristic(end_node, record_open[x]))
            cost_collection = record_open[total_cost]
        
            if show_animation:
                plt.plot(self.calc_grid_position(cost_collection.x, self.x_min),  self.calc_grid_position(cost_collection.y, self.y_min), "xy")
                if len(record_closed.keys())%10 == 0:
                    plt.pause(0.001)
            
            if cost_collection.x == end_node.x and cost_collection.y == end_node.y:
                print("Finished!")
                end_node.path = cost_collection.path
                end_node.cost = cost_collection.cost
                break

            del record_open[total_cost]
            record_closed[total_cost] = cost_collection

            for i, _ in enumerate(self.path):
                node = self.Node(cost_collection.x + self.path[i][0], cost_collection.y + self.path[i][1], cost_collection.cost + self.path[i][2], total_cost)
                idx_node = self.calc_grid_idx(node)

                if not self.check_validity(node):
                    continue

                if idx_node in record_closed:
                    continue

                if idx_node not in record_open:
                    record_open[idx_node] = node
                else:
                    if record_open[idx_node].cost > node.cost:
                        record_open[idx_node] = node

        x_out_path, y_out_path = self.calc_final_path(end_node, record_closed)

        return x_out_path, y_out_path

def save_roi_config():
    config = {
        'roi_enabled': roi_enabled,
        'roi_x': roi_x,
        'roi_y': roi_y,
        'roi_w': roi_w,
        'roi_h': roi_h
    }
    with open(ROI_CONFIG_FILE, 'w') as f:
        json.dump(config, f)

def load_roi_config():
    global roi_enabled, roi_x, roi_y, roi_w, roi_h
    if os.path.exists(ROI_CONFIG_FILE):
        try:
            with open(ROI_CONFIG_FILE, 'r') as f:
                config = json.load(f)
                roi_enabled = config.get('roi_enabled', False)
                roi_x = config.get('roi_x', 0)
                roi_y = config.get('roi_y', 0)
                roi_w = config.get('roi_w', 0)
                roi_h = config.get('roi_h', 0)
                return True
        except:
            return False
    return False

def detect_obstacles(frame):
    # Преобразуем в оттенки серого
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Применяем размытие для уменьшения шума
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Применяем пороговую обработку
    _, threshold = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)
    
    # Находим контуры
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    obstacles = []
    for contour in contours:
        # Фильтруем маленькие контуры
        if cv2.contourArea(contour) > 100:
            # Получаем ограничивающий прямоугольник
            x, y, w, h = cv2.boundingRect(contour)
            obstacles.append((x, y, w, h))
    
    return obstacles

def run_path_planning(frame, start_x, start_y, end_x, end_y):
    global coordinates
    
    # Определяем препятствия
    obstacles = detect_obstacles(frame)
    if not obstacles:
        return []
    
    # Подготавливаем данные для A*
    x_obstacle, y_obstacle = [], []
    for (x, y, w, h) in obstacles:
        for i in range(w):
            for j in range(h):
                x_obstacle.append(x + i)
                y_obstacle.append(y + j)
    
    # Параметры A*
    grid_size = 5.0
    robot_radius = 5.0
    
    # Запускаем A*
    a_star = AStarPath(robot_radius, grid_size, x_obstacle, y_obstacle)
    x_out_path, y_out_path = a_star.a_star_search(start_x, start_y, end_x, end_y)
    
    coordinates = list(zip(x_out_path, y_out_path))
    return coordinates

def sharpen_image(image):
    gaussian = cv2.GaussianBlur(image, (0,0), 3)
    return cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)

def mouse_callback(event, x, y, flags, param):
    global roi_enabled, roi_x, roi_y, roi_w, roi_h, selecting_roi, roi_start_point
    global start_point, end_point, path_planning_enabled, path
    
    if event == cv2.EVENT_LBUTTONDOWN:
        selecting_roi = True
        roi_start_point = (x, y)
        roi_x, roi_y, roi_w, roi_h = 0, 0, 0, 0
        
    elif event == cv2.EVENT_MOUSEMOVE and selecting_roi:
        roi_x = min(roi_start_point[0], x)
        roi_y = min(roi_start_point[1], y)
        roi_w = abs(x - roi_start_point[0])
        roi_h = abs(y - roi_start_point[1])
        
    elif event == cv2.EVENT_LBUTTONUP:
        selecting_roi = False
        if roi_w > 10 and roi_h > 10:  # Минимальный размер ROI
            roi_enabled = True
            save_roi_config()  # Сохраняем параметры ROI
        else:
            roi_enabled = False
    
    # Обработка выбора точек для планирования пути
    if event == cv2.EVENT_RBUTTONDOWN:
        if start_point is None:
            start_point = (x, y)
            end_point = None
            path = []
        else:
            end_point = (x, y)
            path_planning_enabled = True

# Загружаем сохраненные параметры ROI при старте
load_roi_config()

# Создаем объект для захвата видео
cap = cv2.VideoCapture(rtsp_url)

# Проверяем успешность подключения
if not cap.isOpened():
    print("Ошибка: Не удалось подключиться к камере")
    exit()

# Получаем разрешение кадра
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Размеры окна для отображения
window_width = 800
window_height = 600

# Создаем окно с именем 'IP Camera Stream'
cv2.namedWindow('IP Camera Stream', cv2.WINDOW_NORMAL)
cv2.resizeWindow('IP Camera Stream', window_width, window_height)

# Центрируем окно на экране
screen_width = cv2.getWindowImageRect('IP Camera Stream')[2]
screen_height = cv2.getWindowImageRect('IP Camera Stream')[3]
x = (screen_width - window_width) // 2
y = (screen_height - window_height) // 2
cv2.moveWindow('IP Camera Stream', x, y)

# Создаем очередь для передачи кадров между потоками
frame_queue = queue.Queue(maxsize=10)

# Функция для чтения кадров в отдельном потоке
def capture_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Ошибка: Не удалось получить кадр")
            break
        if not frame_queue.full():
            frame_queue.put(frame)

# Запускаем поток для чтения кадров
thread = threading.Thread(target=capture_frames)
thread.daemon = True
thread.start()

# Устанавливаем обработчик событий мыши
cv2.setMouseCallback('IP Camera Stream', mouse_callback)

try:
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()

            # Обновляем матрицы калибровки из параметров
            camera_matrix = np.array([
                [params[0]['value'], 0, params[2]['value']],
                [0, params[1]['value'], params[3]['value']],
                [0, 0, 1]
            ], dtype=np.float32)

            dist_coeffs = np.array([
                [params[4]['value'], params[5]['value'], 
                [params[6]['value'], params[7]['value']]
            ], dtype=np.float32)

            # Коррекция эффекта "рыбьего глаза"
            undistorted_frame = cv2.undistort(
                frame, 
                camera_matrix, 
                dist_coeffs
            )

            # Повышение четкости изображения
            sharpened_frame = sharpen_image(undistorted_frame)

            # Изменяем размер кадра под размер окна
            resized_frame = cv2.resize(sharpened_frame, (window_width, window_height))
            
            # Создаем копию для отображения
            display_frame = resized_frame.copy()
            
            # Если ROI включена, показываем только обрезанную область
            if roi_enabled:
                # Убедимся, что ROI не выходит за границы кадра
                roi_x = max(0, min(roi_x, window_width - 1))
                roi_y = max(0, min(roi_y, window_height - 1))
                roi_w = max(1, min(roi_w, window_width - roi_x))
                roi_h = max(1, min(roi_h, window_height - roi_y))
                
                # Обрезаем кадр по ROI (без масштабирования)
                display_frame = display_frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
                
                # Создаем черное изображение того же размера, что и оригинальное окно
                black_frame = np.zeros_like(resized_frame)
                
                # Помещаем обрезанную область в центр черного изображения
                y_offset = (window_height - roi_h) // 2
                x_offset = (window_width - roi_w) // 2
                black_frame[y_offset:y_offset+roi_h, x_offset:x_offset+roi_w] = display_frame
                
                display_frame = black_frame

            # Если идет выделение ROI, рисуем прямоугольник
            if selecting_roi:
                cv2.rectangle(resized_frame, (roi_x, roi_y), 
                            (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)
                display_frame = resized_frame

            # Отображение информации о параметрах
            y_start = 30
            cv2.putText(display_frame, 
                       f"Selected: {params[current_param]['name']} ({current_param+1})", 
                       (10, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            
            cv2.putText(display_frame, 
                       f"Value: {params[current_param]['value']:.3f} [ +/- to adjust ]", 
                       (10, y_start+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            
            # Информация о ROI
            cv2.putText(display_frame, 
                       f"ROI: {'ON' if roi_enabled else 'OFF'} [r to reset, s to save]", 
                       (10, y_start+60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

            # Список всех параметров
            for i, param in enumerate(params):
                text = f"{i+1}: {param['name']} = {param['value']:.3f}"
                color = (0, 255, 0) if i == current_param else (0, 0, 255)
                cv2.putText(display_frame, text, 
                           (10, y_start + 100 + i*30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

            # Планирование пути
            if start_point is not None:
                cv2.circle(display_frame, start_point, 5, (0, 255, 0), -1)
                if end_point is not None:
                    cv2.circle(display_frame, end_point, 5, (0, 0, 255), -1)
                    
                    if path_planning_enabled:
                        # Запускаем планирование пути
                        coordinates = run_path_planning(resized_frame, start_point[0], start_point[1], end_point[0], end_point[1])
                        path_planning_enabled = False
                        
                        # Рисуем путь
                        if coordinates:
                            for i in range(1, len(coordinates)):
                                cv2.line(display_frame, coordinates[i-1], coordinates[i], (255, 0, 0), 2)

            # Отображаем кадр в окне
            cv2.imshow('IP Camera Stream', display_frame)

        # Обработка клавиш
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif ord('1') <= key <= ord('8'):
            current_param = key - ord('1')
        elif key == ord('+') or key == ord('='):
            params[current_param]['value'] += params[current_param]['step']
        elif key == ord('-') or key == ord('_'):
            params[current_param]['value'] -= params[current_param]['step']
        elif key == ord('r'):  # Сброс ROI
            roi_enabled = False
            roi_x, roi_y, roi_w, roi_h = 0, 0, 0, 0
        elif key == ord('s'):  # Сохранить ROI
            if roi_enabled:
                save_roi_config()
        elif key == ord('p'):  # Сброс точек пути
            start_point = None
            end_point = None
            path = []

finally:
    # Освобождаем ресурсы и закрываем окна
    cap.release()
    cv2.destroyAllWindows()
