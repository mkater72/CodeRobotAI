import cv2
import numpy as np
import threading
import queue
import json
import os

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

# Функция для повышения резкости
def sharpen_image(image):
    gaussian = cv2.GaussianBlur(image, (0,0), 3)
    return cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)

# Функция обработки событий мыши
def mouse_callback(event, x, y, flags, param):
    global roi_enabled, roi_x, roi_y, roi_w, roi_h, selecting_roi, roi_start_point
    
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
                params[6]['value'], params[7]['value']]
            ], dtype=np.float32)

            # Коррекция эффекта "рыбьего глаза"
            undistorted_frame = cv2.undistort(
                frame, 
                camera_matrix, 
                dist_coeffs
            )

            # Повышение четкости изображения
            sharpened_frame = sharpen_image(undistorted_frame)

            # Записываем кадр в видеофайл

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

finally:
    # Освобождаем ресурсы и закрываем окна
    cap.release()
    cv2.destroyAllWindows()
