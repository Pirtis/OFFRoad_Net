from vehicle import Driver
import socket
import os
import numpy as np
import cv2
from PIL import Image
import threading
import time

# =============================================================================
# НАСТРОЙКИ
# =============================================================================
FRAME_DIR = "camera_frames"
os.makedirs(FRAME_DIR, exist_ok=True)

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
SAVE_EVERY_N_FRAMES = 50  # Сохранять каждый 50-й кадр
DETECT_RED_OBSTACLES = True

# =============================================================================
# ИНИЦИАЛИЗАЦИЯ WEBOTS
# =============================================================================
driver = Driver()
timestep = int(driver.getBasicTimeStep())
print(f"Basic timestep: {timestep} ms")

# Получаем камеру
camera = driver.getDevice("map_camera")
if camera is None:
    print("ERROR: map_camera not found!")
    exit(1)

camera.enable(timestep)
print(f"Camera found: {camera.getName()}")
print(f"Resolution: {camera.getWidth()} x {camera.getHeight()}")
print(f"Field of view: {camera.getFov():.2f} rad")

# =============================================================================
# UDP ДЛЯ СВЯЗИ С CAR_RESET
# =============================================================================
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.settimeout(0.5)
CAR_RESET_ADDR = ("127.0.0.1", 6006)


def send_waypoints(waypoints):
    """Отправка путевых точек в супервизор"""
    if not waypoints:
        return
    waypoints_str = ";".join([f"{wp[0]:.2f},{wp[1]:.2f}" for wp in waypoints])
    try:
        sock.sendto(f"WAYPOINTS:{waypoints_str}".encode(), CAR_RESET_ADDR)
        print(f"  Sent {len(waypoints)} waypoints")
    except Exception as e:
        print(f"  Error: {e}")


def clear_waypoints():
    """Очистка путевых точек"""
    try:
        sock.sendto(b"CLEAR_WAYPOINTS", CAR_RESET_ADDR)
    except:
        pass


# =============================================================================
# ОБРАБОТКА ИЗОБРАЖЕНИЯ
# =============================================================================
def detect_red_obstacles(img_bgr):
    """Обнаружение красных препятствий на изображении"""
    # Конвертируем в HSV для лучшего обнаружения
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Диапазоны красного цвета
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Морфологическая очистка
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

    # Находим контуры
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    obstacles = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  # Минимальная площадь
            x, y, w, h = cv2.boundingRect(contour)
            obstacles.append({
                'bbox': (x, y, w, h),
                'center': (x + w // 2, y + h // 2),
                'area': area
            })

    return obstacles, red_mask


def draw_obstacles(img, obstacles):
    """Отрисовка обнаруженных препятствий"""
    for obs in obstacles:
        x, y, w, h = obs['bbox']
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(img, f"Red", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return img


def process_and_save_frame(img_data, width, height, frame_count):
    """Обработка и сохранение кадра"""
    # Преобразуем в numpy массив
    img_array = np.frombuffer(img_data, dtype=np.uint8).reshape(height, width, 4)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_BGRA2BGR)

    # Обнаружение красных препятствий
    red_obstacles, red_mask = detect_red_obstacles(img_bgr)

    # Рисуем препятствия на изображении
    img_with_obstacles = draw_obstacles(img_bgr.copy(), red_obstacles)

    # Добавляем информацию на изображение
    info_text = [
        f"Frame: {frame_count}",
        f"Red obstacles: {len(red_obstacles)}",
        f"Resolution: {width}x{height}",
        f"Press 's' - save, 'q' - quit, 'w' - send waypoints"
    ]

    for i, text in enumerate(info_text):
        cv2.putText(img_with_obstacles, text, (10, 30 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Сохраняем кадр (каждый N-й кадр)
    if frame_count % SAVE_EVERY_N_FRAMES == 0:
        filename = f"{FRAME_DIR}/frame_{frame_count:06d}.jpg"
        cv2.imwrite(filename, img_with_obstacles)
        print(f"  Saved: {filename}")

    # Сохраняем маску красных пикселей при обнаружении
    if red_obstacles and frame_count % 10 == 0:
        mask_filename = f"{FRAME_DIR}/red_mask_{frame_count:06d}.jpg"
        cv2.imwrite(mask_filename, red_mask)
        print(f"  Red mask saved: {mask_filename}")

    return img_with_obstacles, len(red_obstacles)


def get_waypoints_from_obstacles(obstacles, start_pos, goal_pos):
    """Построение маршрута в обход препятствий"""
    if not obstacles:
        return [start_pos, goal_pos]

    waypoints = [start_pos]

    for obs in obstacles:
        cx, cy = obs['center']
        w, h = obs['bbox'][2], obs['bbox'][3]

        # Преобразуем пиксельные координаты в примерные мировые
        # (здесь нужно добавить калибровку)
        world_cx = cx * 0.1 - 100
        world_cy = cy * 0.1 - 100

        # Добавляем обходные точки
        waypoints.append((world_cx, world_cy - 20))
        waypoints.append((world_cx, world_cy + 20))

    waypoints.append(goal_pos)

    # Упрощаем путь
    if len(waypoints) > 2:
        simplified = [waypoints[0]]
        for i in range(1, len(waypoints) - 1):
            if (abs(waypoints[i][0] - simplified[-1][0]) > 10 or
                    abs(waypoints[i][1] - simplified[-1][1]) > 10):
                simplified.append(waypoints[i])
        simplified.append(waypoints[-1])
        waypoints = simplified

    return waypoints


# =============================================================================
# ОСНОВНОЙ ЦИКЛ
# =============================================================================
print("=" * 60)
print("CAMERA VIEWER - Full Version")
print("=" * 60)
print(f"Frame save directory: {FRAME_DIR}")
print(f"Save every: {SAVE_EVERY_N_FRAMES} frames")
print("=" * 60)
print("Controls:")
print("  's' - Save current frame")
print("  'w' - Send waypoints to supervisor")
print("  'q' - Quit")
print("=" * 60)
print("Waiting for simulation to start...\n")

# Ждём стабилизации
for _ in range(50):
    driver.step()

# Создаём окно OpenCV
cv2.namedWindow("Camera View", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Camera View", 800, 600)

frame_count = 0
step = 0
car_speed = 0
car_steering = 0

# Начальные путевые точки (прямая линия)
current_waypoints = [(-77.2, 114.5), (155.2, -155.8)]

while driver.step() != -1:
    step += 1

    # Получаем изображение с камеры
    img_data = camera.getImage()
    if img_data is None:
        continue

    frame_count += 1
    width = camera.getWidth()
    height = camera.getHeight()

    # Обрабатываем кадр
    display_img, red_count = process_and_save_frame(img_data, width, height, frame_count)

    # Показываем изображение в окне
    cv2.imshow("Camera View", display_img)

    # Обработка клавиш
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("\nQuitting...")
        break
    elif key == ord('s'):
        filename = f"{FRAME_DIR}/manual_save_{frame_count}.jpg"
        cv2.imwrite(filename, display_img)
        print(f"Manual save: {filename}")
    elif key == ord('w'):
        # Отправляем текущие путевые точки
        send_waypoints(current_waypoints)

    # Автоматическая отправка путевых точек каждые 200 кадров
    if frame_count % 200 == 0:
        # Пример: отправляем прямую линию
        waypoints = [(-77.2, 114.5), (155.2, -155.8)]
        send_waypoints(waypoints)
        current_waypoints = waypoints

    # Логирование
    if frame_count % 100 == 0:
        print(f"\n[Frame {frame_count}] Red obstacles: {red_count}")
        print(f"  Step: {step}, Car stationary")

    # Машина не двигается (только просмотр камеры)
    driver.setCruisingSpeed(0)
    driver.setSteeringAngle(0)

# =============================================================================
# ЗАВЕРШЕНИЕ
# =============================================================================
print("\n" + "=" * 60)
print("CAMERA VIEWER SHUTDOWN")
print("=" * 60)
print(f"Total frames captured: {frame_count}")
print(f"Images saved to: {FRAME_DIR}")
cv2.destroyAllWindows()
print("Done.")