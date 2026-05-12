from controller import Robot
import numpy as np
import cv2
import os
import math
from collections import deque

# =============================================================================
# НАСТРОЙКИ КАМЕРЫ
# =============================================================================
WORLD_MIN_X = -100
WORLD_MAX_X = 200
WORLD_MIN_Y = -200
WORLD_MAX_Y = 150

# Фиксированные точки
START_POINT = (WORLD_MAX_X - 10, WORLD_MIN_Y + 10)
FINISH_POINT = (155.2, -155.8)

# =============================================================================
# НАСТРОЙКИ ДЕТЕКЦИИ
# =============================================================================
RED_LOWER1 = np.array([0, 50, 50])
RED_UPPER1 = np.array([10, 255, 255])
RED_LOWER2 = np.array([160, 50, 50])
RED_UPPER2 = np.array([179, 255, 255])

GREEN_LOWER = np.array([40, 50, 50])
GREEN_UPPER = np.array([80, 255, 255])

MIN_OBSTACLE_AREA = 30
GRID_RESOLUTION = 5  # метров на ячейку сетки

# =============================================================================
# ИНИЦИАЛИЗАЦИЯ
# =============================================================================
robot = Robot()
timestep = int(robot.getBasicTimeStep())

camera = robot.getDevice("satellite_camera")
if camera is None:
    print("ERROR: satellite_camera not found!")
    exit(1)

camera.enable(timestep)
print(f"Satellite camera OK: {camera.getWidth()}x{camera.getHeight()}")

os.makedirs("sputnik_frames", exist_ok=True)


# =============================================================================
# ГЕОМЕТРИЧЕСКИЕ ФУНКЦИИ
# =============================================================================
def pixel_to_world(px, py, img_width, img_height):
    scale_x = (WORLD_MAX_X - WORLD_MIN_X) / img_width
    scale_y = (WORLD_MAX_Y - WORLD_MIN_Y) / img_height
    world_x = WORLD_MIN_X + px * scale_x
    world_y = WORLD_MIN_Y + (img_height - py) * scale_y
    return world_x, world_y


def world_to_pixel(wx, wy, img_width, img_height):
    scale_x = (WORLD_MAX_X - WORLD_MIN_X) / img_width
    scale_y = (WORLD_MAX_Y - WORLD_MIN_Y) / img_height
    px = int((wx - WORLD_MIN_X) / scale_x)
    py = int(img_height - (wy - WORLD_MIN_Y) / scale_y)
    return max(0, min(px, img_width - 1)), max(0, min(py, img_height - 1))


def point_to_segment_distance(px, py, x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(px - x1, py - y1)
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    t = max(0, min(1, t))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.hypot(px - proj_x, py - proj_y)


def line_intersects_obstacle(x1, y1, x2, y2, obstacle):
    """Проверяет, пересекает ли отрезок препятствие"""
    ox, oy, w, h = obstacle['bbox_px']
    # Проверяем пересечение с прямоугольником
    if (x1 < ox + w and x2 < ox + w and x1 > ox and x2 > ox):
        return False
    if (y1 < oy + h and y2 < oy + h and y1 > oy and y2 > oy):
        return False

    # Проверяем расстояние от отрезка до центра
    cx, cy = obstacle['center_px']
    dist = point_to_segment_distance(cx, cy, x1, y1, x2, y2)
    return dist < max(w, h) / 2


# =============================================================================
# A* АЛГОРИТМ
# =============================================================================
def create_occupancy_grid(obstacles, width, height):
    """Создание карты занятости для A*"""
    grid_width = int((WORLD_MAX_X - WORLD_MIN_X) / GRID_RESOLUTION) + 1
    grid_height = int((WORLD_MAX_Y - WORLD_MIN_Y) / GRID_RESOLUTION) + 1
    grid = np.zeros((grid_width, grid_height))

    for obs in obstacles:
        wx, wy = obs['center_world']
        ww, wh = obs['size_world']

        gx1 = max(0, int((wx - ww / 2 - WORLD_MIN_X) / GRID_RESOLUTION))
        gx2 = min(grid_width - 1, int((wx + ww / 2 - WORLD_MIN_X) / GRID_RESOLUTION))
        gy1 = max(0, int((wy - wh / 2 - WORLD_MIN_Y) / GRID_RESOLUTION))
        gy2 = min(grid_height - 1, int((wy + wh / 2 - WORLD_MIN_Y) / GRID_RESOLUTION))

        grid[gx1:gx2 + 1, gy1:gy2 + 1] = 1

    return grid


def world_to_grid(wx, wy):
    gx = int((wx - WORLD_MIN_X) / GRID_RESOLUTION)
    gy = int((wy - WORLD_MIN_Y) / GRID_RESOLUTION)
    return gx, gy


def grid_to_world(gx, gy):
    wx = WORLD_MIN_X + (gx + 0.5) * GRID_RESOLUTION
    wy = WORLD_MIN_Y + (gy + 0.5) * GRID_RESOLUTION
    return wx, wy


def heuristic(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def a_star_search(grid, start, goal):
    """A* поиск кратчайшего пути"""
    start_grid = world_to_grid(start[0], start[1])
    goal_grid = world_to_grid(goal[0], goal[1])

    # Проверка границ
    if not (0 <= start_grid[0] < grid.shape[0] and 0 <= start_grid[1] < grid.shape[1]):
        start_grid = (max(0, min(start_grid[0], grid.shape[0] - 1)),
                      max(0, min(start_grid[1], grid.shape[1] - 1)))
    if not (0 <= goal_grid[0] < grid.shape[0] and 0 <= goal_grid[1] < grid.shape[1]):
        goal_grid = (max(0, min(goal_grid[0], grid.shape[0] - 1)),
                     max(0, min(goal_grid[1], grid.shape[1] - 1)))

    open_set = {start_grid}
    came_from = {}
    g_score = {start_grid: 0}
    f_score = {start_grid: heuristic(start_grid, goal_grid)}

    while open_set:
        current = min(open_set, key=lambda x: f_score.get(x, float('inf')))

        if current == goal_grid:
            # Восстанавливаем путь
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start_grid)
            path.reverse()
            return path

        open_set.remove(current)

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)

            if not (0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]):
                continue
            if grid[neighbor] == 1:
                continue

            move_cost = math.hypot(dx, dy)
            tentative_g = g_score[current] + move_cost

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal_grid)
                if neighbor not in open_set:
                    open_set.add(neighbor)

    return None


# =============================================================================
# УПРОЩЕНИЕ ПУТИ (ТОЛЬКО ПОВОРОТЫ)
# =============================================================================
def simplify_path(waypoints):
    """Оставляет только точки на поворотах"""
    if len(waypoints) < 3:
        return waypoints

    simplified = [waypoints[0]]

    for i in range(1, len(waypoints) - 1):
        p1 = np.array(simplified[-1])
        p2 = np.array(waypoints[i])
        p3 = np.array(waypoints[i + 1])

        v1 = p2 - p1
        v2 = p3 - p2

        if np.linalg.norm(v1) < 0.5 or np.linalg.norm(v2) < 0.5:
            continue

        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)

        dot = np.dot(v1, v2)

        # Добавляем точку только если направление меняется (поворот > 15 градусов)
        if abs(dot - 1) > 0.03:
            simplified.append(p2)

    simplified.append(waypoints[-1])

    # Удаляем слишком близкие точки
    filtered = [simplified[0]]
    for p in simplified[1:]:
        if np.linalg.norm(np.array(p) - np.array(filtered[-1])) > 8.0:
            filtered.append(p)

    return filtered


# =============================================================================
# ДЕТЕКЦИЯ
# =============================================================================
def detect_red_obstacles(hsv_image):
    mask1 = cv2.inRange(hsv_image, RED_LOWER1, RED_UPPER1)
    mask2 = cv2.inRange(hsv_image, RED_LOWER2, RED_UPPER2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((3, 3), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    obstacles = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > MIN_OBSTACLE_AREA:
            x, y, w, h = cv2.boundingRect(contour)
            center_px = (x + w // 2, y + h // 2)
            center_world = pixel_to_world(center_px[0], center_px[1],
                                          camera.getWidth(), camera.getHeight())

            obstacles.append({
                'bbox_px': (x, y, w, h),
                'center_px': center_px,
                'center_world': center_world,
                'size_px': (w, h),
                'size_world': (w * (WORLD_MAX_X - WORLD_MIN_X) / camera.getWidth(),
                               h * (WORLD_MAX_Y - WORLD_MIN_Y) / camera.getHeight()),
                'area': area
            })

    return obstacles, red_mask


def detect_green_finish(hsv_image):
    green_mask = cv2.inRange(hsv_image, GREEN_LOWER, GREEN_UPPER)
    kernel = np.ones((3, 3), np.uint8)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    finish_positions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:
            x, y, w, h = cv2.boundingRect(contour)
            center_px = (x + w // 2, y + h // 2)
            center_world = pixel_to_world(center_px[0], center_px[1],
                                          camera.getWidth(), camera.getHeight())
            finish_positions.append({
                'center_world': center_world,
                'area': area
            })

    return finish_positions, green_mask


# =============================================================================
# ОСНОВНОЙ ЦИКЛ
# =============================================================================
def draw_detections(image, red_obstacles, finish_positions, waypoints, path_px):
    img = image.copy()

    # Старт
    start_px = world_to_pixel(START_POINT[0], START_POINT[1], img.shape[1], img.shape[0])
    cv2.circle(img, start_px, 12, (255, 165, 0), -1)
    cv2.putText(img, "START", (start_px[0] - 25, start_px[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)

    # Красные препятствия
    for obs in red_obstacles:
        x, y, w, h = obs['bbox_px']
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Финиш
    for finish in finish_positions:
        fx, fy = world_to_pixel(finish['center_world'][0], finish['center_world'][1], img.shape[1], img.shape[0])
        cv2.circle(img, (fx, fy), 15, (0, 255, 0), -1)
        cv2.putText(img, "FINISH", (fx - 30, fy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Путь A*
    if len(path_px) > 1:
        for i in range(len(path_px) - 1):
            cv2.line(img, path_px[i], path_px[i + 1], (255, 255, 0), 2)

    # Путевые точки (только на поворотах)
    for i, wp in enumerate(waypoints):
        wp_px = world_to_pixel(wp[0], wp[1], img.shape[1], img.shape[0])
        cv2.circle(img, wp_px, 6, (0, 255, 255), -1)
        cv2.putText(img, str(i + 1), (wp_px[0] - 5, wp_px[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    return img


print("=" * 60)
print("A* SHORTEST PATH ALGORITHM")
print("=" * 60)
print(f"Start: {START_POINT}")
print(f"Finish: {FINISH_POINT}")
print(f"Grid resolution: {GRID_RESOLUTION}m")
print("Controls: 'q' - quit, 's' - save")
print("=" * 60)

cv2.namedWindow("A* Shortest Path", cv2.WINDOW_NORMAL)
cv2.resizeWindow("A* Shortest Path", 1024, 1024)

frame_count = 0

for _ in range(50):
    robot.step()

while robot.step() != -1:
    frame_count += 1

    img_data = camera.getImage()
    if img_data is None:
        continue

    width = camera.getWidth()
    height = camera.getHeight()

    img_array = np.frombuffer(img_data, dtype=np.uint8).reshape(height, width, 4)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_BGRA2BGR)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Детекция
    red_obstacles, _ = detect_red_obstacles(img_hsv)
    finish_positions, _ = detect_green_finish(img_hsv)

    # Цель
    if finish_positions:
        goal = finish_positions[0]['center_world']
    else:
        goal = FINISH_POINT

    # A* поиск
    grid = create_occupancy_grid(red_obstacles, width, height)
    path_grid = a_star_search(grid, START_POINT, goal)

    if path_grid:
        path_world = [grid_to_world(gx, gy) for gx, gy in path_grid]
        path_px = [world_to_pixel(wx, wy, width, height) for wx, wy in path_world]

        # Упрощаем путь до точек поворота
        waypoints = simplify_path(path_world)

        print(f"\n[Frame {frame_count}] Path found!")
        print(f"  Grid cells: {len(path_grid)}")
        print(f"  Waypoints: {len(waypoints)}")
        for i, wp in enumerate(waypoints):
            print(f"    {i + 1}: ({wp[0]:.1f}, {wp[1]:.1f})")
    else:
        path_px = []
        waypoints = [START_POINT, goal]
        print(f"\n[Frame {frame_count}] Path not found, using direct line")

    # Визуализация
    display_img = draw_detections(img_bgr, red_obstacles, finish_positions, waypoints, path_px)

    cv2.imshow("A* Shortest Path", display_img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite(f"sputnik_frames/frame_{frame_count}.jpg", display_img)
        print(f"Saved: sputnik_frames/frame_{frame_count}.jpg")

cv2.destroyAllWindows()
print("Done")