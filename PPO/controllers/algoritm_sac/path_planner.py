"""
Модуль планирования пути с использованием A* алгоритма.
Строит маршрут от старта до финиша в обход препятствий.
"""

import numpy as np
import math


class PathPlanner:
    """Класс для планирования пути с обходом препятствий."""

    def __init__(self, grid_resolution=3.0):
        """
        Инициализация планировщика пути.

        Args:
            grid_resolution: Разрешение сетки в метрах
        """
        self.grid_resolution = grid_resolution
        self.obstacle_buffer = 2.0

    def plan_path(self, start, goal, obstacles):
        """
        Построение маршрута от старта до финиша.

        Args:
            start: Кортеж (x, y) начальной позиции
            goal: Кортеж (x, y) целевой позиции
            obstacles: Список препятствий [x, y, width, depth]

        Returns:
            List of tuples: Список путевых точек маршрута
        """

        if not obstacles:
            print("   Препятствия отсутствуют, построение прямой линии")
            return self.smooth_path([start, goal])

        # Определение границ карты
        all_points = [start, goal]
        for obs in obstacles:
            all_points.append((obs[0] - obs[2] / 2, obs[1] - obs[3] / 2))
            all_points.append((obs[0] + obs[2] / 2, obs[1] + obs[3] / 2))

        min_x = min(p[0] for p in all_points) - 15
        max_x = max(p[0] for p in all_points) + 15
        min_y = min(p[1] for p in all_points) - 15
        max_y = max(p[1] for p in all_points) + 15

        # Создание карты препятствий
        width = int((max_x - min_x) / self.grid_resolution) + 1
        height = int((max_y - min_y) / self.grid_resolution) + 1
        grid = np.zeros((width, height))

        # Отметка препятствий с буфером
        for obs in obstacles:
            self._mark_obstacle(grid, obs, min_x, min_y)

        # Поиск пути
        start_grid = self._world_to_grid(start, min_x, min_y)
        goal_grid = self._world_to_grid(goal, min_x, min_y)

        waypoints_grid = self._a_star(grid, start_grid, goal_grid)

        if waypoints_grid is None or len(waypoints_grid) < 2:
            print("   Путь не найден, используется прямая линия")
            return self.smooth_path([start, goal])

        # Преобразование в мировые координаты
        waypoints = [self._grid_to_world(p, min_x, min_y) for p in waypoints_grid]

        # Упрощение пути до ключевых точек поворота
        simplified = self._simplify_to_turns(waypoints)

        # Фильтрация точек, находящихся внутри препятствий
        filtered = self._filter_collisions(simplified, obstacles)

        print(f"   Построено {len(waypoints)} точек, упрощено до {len(filtered)}")
        return filtered

    def _mark_obstacle(self, grid, obstacle, min_x, min_y):
        """
        Отметка препятствия на карте с добавлением буфера.

        Args:
            grid: 2D массив карты
            obstacle: Препятствие [x, y, width, depth]
            min_x: Минимальная координата X карты
            min_y: Минимальная координата Y карты
        """
        x, y, w, d = obstacle
        buffer = self.obstacle_buffer
        x_min = x - w / 2 - buffer
        x_max = x + w / 2 + buffer
        y_min = y - d / 2 - buffer
        y_max = y + d / 2 + buffer

        gx_min = max(0, int((x_min - min_x) / self.grid_resolution))
        gx_max = min(grid.shape[0] - 1, int((x_max - min_x) / self.grid_resolution))
        gy_min = max(0, int((y_min - min_y) / self.grid_resolution))
        gy_max = min(grid.shape[1] - 1, int((y_max - min_y) / self.grid_resolution))

        grid[gx_min:gx_max + 1, gy_min:gy_max + 1] = 1

    def _a_star(self, grid, start, goal):
        """
        Реализация A* алгоритма поиска пути.

        Args:
            grid: 2D массив карты (0 - свободно, 1 - препятствие)
            start: Кортеж (x, y) начальной позиции в координатах сетки
            goal: Кортеж (x, y) целевой позиции в координатах сетки

        Returns:
            List of tuples: Путь в координатах сетки или None
        """
        open_set = {start}
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, goal)}

        while open_set:
            current = min(open_set, key=lambda x: f_score.get(x, float('inf')))
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path

            open_set.remove(current)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if not (0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]):
                    continue
                if grid[neighbor] == 1:
                    continue

                move_cost = math.sqrt(dx * dx + dy * dy)
                tentative_g = g_score[current] + move_cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal)
                    if neighbor not in open_set:
                        open_set.add(neighbor)
        return None

    def _heuristic(self, a, b):
        """Евклидово расстояние между двумя точками для эвристики A*."""
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def _world_to_grid(self, point, min_x, min_y):
        """Преобразование мировых координат в координаты сетки."""
        x = int((point[0] - min_x) / self.grid_resolution)
        y = int((point[1] - min_y) / self.grid_resolution)
        return (x, y)

    def _grid_to_world(self, grid_point, min_x, min_y):
        """Преобразование координат сетки в мировые координаты."""
        x = min_x + (grid_point[0] + 0.5) * self.grid_resolution
        y = min_y + (grid_point[1] + 0.5) * self.grid_resolution
        return (x, y)

    def smooth_path(self, waypoints):
        """Базовое сглаживание пути."""
        if len(waypoints) < 3:
            return waypoints
        return waypoints

    def _simplify_to_turns(self, waypoints):
        """
        Упрощение пути до ключевых точек поворота.
        Оставляются только точки, где направление движения значительно меняется.

        Args:
            waypoints: Список путевых точек

        Returns:
            List of tuples: Упрощенный список путевых точек
        """
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

            # Добавление точки при значительном изменении направления (более 30 градусов)
            if abs(dot - 1) > 0.15:
                simplified.append(p2)

        simplified.append(waypoints[-1])

        # Удаление слишком близких точек
        filtered = [simplified[0]]
        for p in simplified[1:]:
            if np.linalg.norm(np.array(p) - np.array(filtered[-1])) > 8.0:
                filtered.append(p)

        return filtered

    def _filter_collisions(self, waypoints, obstacles):
        """
        Удаление точек, находящихся внутри препятствий.

        Args:
            waypoints: Список путевых точек
            obstacles: Список препятствий

        Returns:
            List of tuples: Отфильтрованный список путевых точек
        """
        if not waypoints:
            return waypoints

        filtered = []
        for wp in waypoints:
            collision = False
            for obs in obstacles:
                x, y, w, d = obs
                if (abs(wp[0] - x) < w / 2 + 1.0 and
                        abs(wp[1] - y) < d / 2 + 1.0):
                    collision = True
                    break
            if not collision:
                filtered.append(wp)

        if len(filtered) < 2 and len(waypoints) >= 2:
            return [waypoints[0], waypoints[-1]]

        return filtered