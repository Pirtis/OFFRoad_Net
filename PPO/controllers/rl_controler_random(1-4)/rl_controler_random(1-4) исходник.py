#!/usr/bin/env python3
"""
РL КОНТРОЛЛЕР ДЛЯ BMW X5 В WEBOTS
==================================
Данный файл реализует среду обучения с подкреплением для автомобиля BMW X5
в симуляторе Webots. Автомобиль учится объезжать препятствия и достигать цели,
используя алгоритм PPO (Proximal Policy Optimization).

Автор: Студенческий проект
Дата: 2026
"""

# =============================================================================
# ИМПОРТ БИБЛИОТЕК
# =============================================================================
from vehicle import Car  # Класс для управления автомобилем в Webots
import numpy as np  # Работа с массивами и математика
import math  # Математические функции
import os  # Работа с файловой системой
import sys  # Системные функции
import random  # Генерация случайных чисел
from collections import deque  # Очередь для хранения истории
import gymnasium as gym  # Стандартный интерфейс для RL-сред
from gymnasium import spaces  # Классы для определения пространств

from stable_baselines3 import PPO  # Алгоритм PPO
from stable_baselines3.common.callbacks import BaseCallback  # Базовый класс для колбэков
import torch as th  # Фреймворк для нейросетей


# =============================================================================
# КЛАСС СРЕДЫ ДЛЯ ОБУЧЕНИЯ
# =============================================================================
class BMWRLEnvironment(gym.Env):
    """
    Среда для обучения с подкреплением автомобиля BMW X5 в Webots.

    Среда реализует интерфейс Gymnasium, что позволяет использовать её
    с различными алгоритмами RL (PPO, SAC, TD3 и др.).

    Особенности:
    - Рандомизация положения препятствий и цели в каждом эпизоде
    - Поддержка заднего хода (отрицательные значения скорости)
    - Детектирование столкновений по лидарам
    - Подробная система наград
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode="human"):
        """
        Инициализация среды.

        Параметры:
            render_mode (str): режим отображения ("human" для визуализации)
        """
        super().__init__()

        print("=" * 60)
        print("BMW X5 — СРЕДА ДЛЯ ОБУЧЕНИЯ С ПОДКРЕПЛЕНИЕМ")
        print("=" * 60)

        # =====================================================================
        # ИНИЦИАЛИЗАЦИЯ WEBOTS
        # =====================================================================

        # Создаём объект автомобиля (предоставляется Webots)
        self.car = Car()

        # Временной шаг симуляции (определяется в world-файле)
        self.timestep = int(self.car.getBasicTimeStep())

        # =====================================================================
        # ПОИСК НОД В МИРЕ WEBOTS
        # =====================================================================
        print("\nПоиск нод в мире:")

        # Поиск автомобиля по идентификатору DEF
        self.car_node = self.car.getFromDef("CAR")
        if self.car_node:
            print(f"  CAR: найдена")
        else:
            # Если не найдена по DEF, используем текущего робота
            print(f"  CAR: не найдена, использую getSelf()")
            self.car_node = self.car.getSelf()

        # Поиск препятствия (опционально, может создаваться программно)
        self.obstacle_node = self.car.getFromDef("OBSTACLE")
        if self.obstacle_node:
            print(f"  OBSTACLE: найдена")
        else:
            print(f"  OBSTACLE: не найдена")

        # Поиск цели
        self.target_node = self.car.getFromDef("TARGET")
        self.use_virtual_target = False

        if self.target_node:
            target_pos = self.target_node.getPosition()
            print(f"TARGET: найдена на ({target_pos[0]:.1f}, {target_pos[1]:.1f}, {target_pos[2]:.1f})")

            # Отключаем физику цели, чтобы машина могла проходить сквозь неё
            try:
                self.target_node.removeBoundingObject()
                self.target_node.enableContactPointsTracking(False)
                self.target_node.resetPhysics()
                print("Физика цели отключена")
            except Exception as e:
                print(f" Не удалось отключить физику: {e}")
        else:
            # Если цель не найдена, используем виртуальную
            print(f"TARGET: не найдена, использую виртуальную цель")
            self.use_virtual_target = True
            self.virtual_target_pos = [10, 0.5, 0.4]

        print()

        # =====================================================================
        # ПАРАМЕТРЫ СРЕДЫ
        # =====================================================================

        # Начальная позиция автомобиля (определена в world-файле)
        self.START_POS = [-11.5, 0.5, 0.5]

        # Начальная ориентация автомобиля (поворот вокруг оси Y)
        self.START_ROT = [0, 1, 0, 0]

        # Расстояние до цели, считающееся успехом (в метрах)
        self.GOAL_DISTANCE = 4.0

        # Расстояние до препятствия, считающееся столкновением (в метрах)
        self.COLLISION_DISTANCE = 0.5

        # Максимальное количество шагов в одном эпизоде
        self.MAX_STEPS = 2500

        # =====================================================================
        # ПАРАМЕТРЫ РАНДОМИЗАЦИИ СЦЕНАРИЕВ
        # =====================================================================

        # Включить/выключить рандомизацию
        self.randomize_scenarios = True

        # Диапазоны для рандомизации положения препятствий
        # X - вправо/влево, Y - вперёд/назад, Z - высота (фиксирована)
        self.OBSTACLE_X_RANGE = (-8, 8)  # От -8 до 8 метров по X
        self.OBSTACLE_Y_RANGE = (-8, 8)  # От -8 до 8 метров по Y
        self.OBSTACLE_Z = 0.5  # Фиксированная высота препятствий

        # Диапазоны для рандомизации положения цели
        self.TARGET_X_RANGE = (-12, 12)  # От -12 до 12 метров по X
        self.TARGET_Y_RANGE = (-10, 10)  # От -10 до 10 метров по Y
        self.TARGET_Z = 0.5  # Фиксированная высота цели

        # Минимальное расстояние от старта до объектов
        self.MIN_DIST_FROM_START = 6.0

        # Минимальное расстояние между объектами
        self.MIN_DIST_BETWEEN_OBJ = 3.0

        # =====================================================================
        # ПАРАМЕТРЫ ДЛЯ НЕСКОЛЬКИХ ПРЕПЯТСТВИЙ
        # =====================================================================

        # Включить режим нескольких препятствий
        self.multiple_obstacles = True

        # Количество препятствий
        self.num_obstacles = 4

        # Список для хранения созданных препятствий
        self.obstacle_nodes = []

        # Список размеров препятствий
        self.obstacle_sizes = []

        # Диапазон размеров препятствий
        self.OBSTACLE_SIZE_RANGE = (0.8, 2.0)

        # =====================================================================
        # СТАТИСТИКА СЦЕНАРИЕВ
        # =====================================================================

        # Счётчики для статистики сложности
        self.scenario_stats = {
            'easy': 0,
            'medium': 0,
            'hard': 0
        }

        # =====================================================================
        # ИНИЦИАЛИЗАЦИЯ СЕНСОРОВ
        # =====================================================================
        self._setup_sensors()

        # =====================================================================
        # ОПРЕДЕЛЕНИЕ ПРОСТРАНСТВА НАБЛЮДЕНИЙ
        # =====================================================================
        # Наблюдение содержит 12 параметров:
        # [dx, dy, dz, yaw, angle_to_target, front, left, right, min_lidar,
        #  current_speed, dist_to_target, prev_steering]
        #
        # dx, dy, dz - расстояние до цели по осям (в метрах)
        # yaw - текущий угол поворота автомобиля (в радианах)
        # angle_to_target - угол до цели относительно направления авто (в радианах)
        # front, left, right - расстояния до препятствий с трёх лидаров (в метрах)
        # min_lidar - минимальное расстояние до любого препятствия (в метрах)
        # current_speed - текущая скорость автомобиля (м/с, отрицательная = назад)
        # dist_to_target - евклидово расстояние до цели (в метрах)
        # prev_steering - предыдущее значение руля (для сглаживания)

        self.observation_space = spaces.Box(
            low=np.array([-30, -10, -30, -np.pi, -np.pi,
                          0, 0, 0, 0, 0, 0, -0.8], dtype=np.float32),
            high=np.array([30, 10, 30, np.pi, np.pi,
                           25, 25, 25, 25, 30, 1000, 0.8], dtype=np.float32),
            dtype=np.float32
        )

        # =====================================================================
        # ОПРЕДЕЛЕНИЕ ПРОСТРАНСТВА ДЕЙСТВИЙ
        # =====================================================================
        # Действие состоит из двух параметров:
        # [steering, speed]
        #
        # steering - угол поворота руля: от -0.8 (влево) до 0.8 (вправо)
        # speed - скорость: от -5.0 (задний ход) до 20.0 (вперёд)

        self.action_space = spaces.Box(
            low=np.array([-0.8, -5.0], dtype=np.float32),
            high=np.array([0.8, 20.0], dtype=np.float32),
            dtype=np.float32
        )

        # =====================================================================
        # СЧЁТЧИКИ ДЛЯ СТАТИСТИКИ
        # =====================================================================
        self.step_count = 0  # Текущий шаг в эпизоде
        self.episode_reward = 0  # Суммарная награда в эпизоде
        self.episode_count = 0  # Количество завершённых эпизодов
        self.success_count = 0  # Количество успешных эпизодов
        self.collision_count = 0  # Количество эпизодов со столкновением

        # Хранение предыдущих действий для сглаживания
        self.prev_action = np.array([0.0, 0.0], dtype=np.float32)
        self.prev_speed = 0.0

        # Лучшее достигнутое расстояние до цели (для бонуса)
        self.best_distance = float('inf')

        self.render_mode = render_mode

        # =====================================================================
        # ВЫВОД ИНФОРМАЦИИ О СРЕДЕ
        # =====================================================================
        print(f"✅ Goal distance: {self.GOAL_DISTANCE}m")
        print(f"✅ Collision distance: {self.COLLISION_DISTANCE}m")
        print(f"✅ Max steps: {self.MAX_STEPS}")
        print(f"✅ Randomization: {'ON' if self.randomize_scenarios else 'OFF'}")
        print(f"✅ Reverse gear: ENABLED (speed range: -5 to 20)")
        print(f"✅ Multiple obstacles: {self.num_obstacles}")
        if self.randomize_scenarios:
            print(f"   Obstacle X: {self.OBSTACLE_X_RANGE}")
            print(f"   Obstacle Y: {self.OBSTACLE_Y_RANGE}")
            print(f"   Obstacle Z: {self.OBSTACLE_Z}")
            print(f"   Target X: {self.TARGET_X_RANGE}")
            print(f"   Target Y: {self.TARGET_Y_RANGE}")
            print(f"   Target Z: {self.TARGET_Z}")
        print(f"📊 Observation space: {self.observation_space.shape}")
        print(f"🎮 Action space: steering={self.action_space.low[0]:.1f}..{self.action_space.high[0]:.1f}, "
              f"speed={self.action_space.low[1]:.1f}..{self.action_space.high[1]:.1f}")
        print("=" * 60)

    # =========================================================================
    # МЕТОДЫ ДЛЯ РАБОТЫ С СЕНСОРАМИ
    # =========================================================================

    def _setup_sensors(self):
        """
        Инициализация и включение всех сенсоров автомобиля.

        Включает:
        - Три лидара (спереди, слева, справа)
        - GPS для определения позиции
        - IMU для определения ориентации
        """
        self.lidar_front = self.car.getDevice("lidar_front")
        self.lidar_left = self.car.getDevice("lidar_left")
        self.lidar_right = self.car.getDevice("lidar_right")
        self.gps = self.car.getDevice("gps")
        self.imu = self.car.getDevice("imu")

        for device in [self.lidar_front, self.lidar_left, self.lidar_right, self.gps, self.imu]:
            if device:
                device.enable(self.timestep)
                print(f"  {device.getName()}")

    def _sanitize_value(self, value, default=0.0, min_val=-1e6, max_val=1e6):
        """
        Очистка значения от NaN и бесконечности.

        Параметры:
            value: исходное значение
            default: значение по умолчанию при ошибке
            min_val: минимальное допустимое значение
            max_val: максимальное допустимое значение

        Возвращает:
            float: очищенное значение
        """
        if value is None or not isinstance(value, (int, float)):
            return default
        if math.isnan(value) or math.isinf(value):
            return default
        return float(np.clip(value, min_val, max_val))

    def _get_lidar_min_distance(self, lidar):
        """
        Получение минимального расстояния до препятствия с лидара.

        Лидар возвращает массив расстояний по всем направлениям.
        Функция фильтрует некорректные значения (inf, nan) и возвращает минимум.

        Параметры:
            lidar: объект лидара из Webots

        Возвращает:
            float: минимальное расстояние в метрах (или 25.0 если нет данных)
        """
        if not lidar:
            return 25.0
        try:
            data = lidar.getRangeImage()
            if not data or len(data) == 0:
                return 25.0
            valid = [x for x in data if x is not None and 0.1 < x < 100
                     and not math.isnan(x) and not math.isinf(x)]
            return float(min(valid)) if valid else 25.0
        except:
            return 25.0

    def _get_position(self):
        """
        Получение текущей позиции автомобиля из GPS.

        Возвращает:
            list: [x, y, z] координаты в метрах
        """
        if self.gps:
            try:
                vals = self.gps.getValues()
                if vals and len(vals) >= 3:
                    return [
                        self._sanitize_value(vals[0], 0, -1000, 1000),
                        self._sanitize_value(vals[1], 0.5, -100, 100),
                        self._sanitize_value(vals[2], 0, -1000, 1000)
                    ]
            except:
                pass
        return [0.0, 0.5, 0.0]

    def _get_yaw(self):
        """
        Получение текущего угла поворота автомобиля (yaw) из IMU.

        Yaw - это поворот вокруг вертикальной оси:
        0 = направление вдоль оси X
        π/2 = направление вдоль оси Y
        π = направление назад

        Возвращает:
            float: угол в радианах от -π до π
        """
        if self.imu:
            try:
                rpy = self.imu.getRollPitchYaw()
                if rpy and len(rpy) >= 3:
                    return self._sanitize_value(rpy[2], 0, -np.pi, np.pi)
            except:
                pass
        return 0.0

    def _get_target_position(self):
        """
        Получение позиции цели (реальной или виртуальной).

        Возвращает:
            list: [x, y, z] координаты цели
        """
        if self.use_virtual_target:
            return self.virtual_target_pos
        elif self.target_node:
            return self.target_node.getPosition()
        return [10, 0.5, 0.4]

    def _get_angle_to_target(self):
        """
        Вычисление угла до цели относительно направления автомобиля.

        Угол 0 означает, что автомобиль смотрит прямо на цель.
        Положительный угол - цель справа, отрицательный - слева.

        Возвращает:
            float: угол в радианах от -π до π
        """
        car_pos = self._get_position()
        target_pos = self._get_target_position()
        dx = target_pos[0] - car_pos[0]
        dz = target_pos[2] - car_pos[2]
        target_angle = math.atan2(dz, dx)
        yaw = self._get_yaw()
        angle = target_angle - yaw
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -np.pi:
            angle += 2 * math.pi
        return self._sanitize_value(angle, 0, -np.pi, np.pi)

    def _check_goal_3d(self):
        """
        Проверка достижения цели по трёхмерному расстоянию.

        Возвращает:
            tuple: (достигнута_ли_цель, расстояние_до_цели)
        """
        car_pos = self._get_position()
        target_pos = self._get_target_position()
        dx = car_pos[0] - target_pos[0]
        dy = car_pos[1] - target_pos[1]
        dz = car_pos[2] - target_pos[2]
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        if math.isnan(distance):
            distance = 999.0
        return distance < self.GOAL_DISTANCE, distance

    def _check_collision_lidar(self):
        """
        Проверка столкновения с препятствиями по данным лидаров.

        Возвращает:
            tuple: (столкновение, информация_о_столкновении, минимальное_расстояние)
        """
        front = self._get_lidar_min_distance(self.lidar_front)
        left = self._get_lidar_min_distance(self.lidar_left)
        right = self._get_lidar_min_distance(self.lidar_right)
        min_dist = min(front, left, right)

        if front < self.COLLISION_DISTANCE:
            return True, f"FRONT:{front:.2f}m", min_dist
        elif left < self.COLLISION_DISTANCE:
            return True, f"LEFT:{left:.2f}m", min_dist
        elif right < self.COLLISION_DISTANCE:
            return True, f"RIGHT:{right:.2f}m", min_dist
        return False, None, min_dist

    def _get_observation(self):
        """
        Формирование вектора наблюдения для агента.

        Возвращает:
            np.array: вектор из 12 параметров (см. описание в __init__)
        """
        car_pos = self._get_position()
        target_pos = self._get_target_position()
        yaw = self._get_yaw()

        dx = self._sanitize_value(target_pos[0] - car_pos[0], 0, -100, 100)
        dy = self._sanitize_value(target_pos[1] - car_pos[1], 0, -10, 10)
        dz = self._sanitize_value(target_pos[2] - car_pos[2], 0, -100, 100)

        angle_to_target = self._get_angle_to_target()

        front = self._get_lidar_min_distance(self.lidar_front)
        left = self._get_lidar_min_distance(self.lidar_left)
        right = self._get_lidar_min_distance(self.lidar_right)
        min_lidar = min(front, left, right)

        current_speed = 0.0
        if self.car_node:
            try:
                vel = self.car_node.getVelocity()
                if vel:
                    # Используем компоненту Z для скорости вперёд/назад
                    current_speed = vel[2]
                    current_speed = self._sanitize_value(current_speed, 0, -100, 100)
            except:
                pass

        _, dist_to_target = self._check_goal_3d()
        dist_to_target = self._sanitize_value(dist_to_target, 999, 0, 1000)

        prev_steering = float(self.prev_action[0])

        obs = np.array([
            dx, dy, dz,
            yaw, angle_to_target,
            front, left, right, min_lidar,
            current_speed,
            dist_to_target,
            prev_steering
        ], dtype=np.float32)

        # Финальная очистка от NaN/Inf
        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            obs = np.nan_to_num(obs, nan=0.0, posinf=25.0, neginf=-25.0)

        return obs

    # =========================================================================
    # МЕТОДЫ ДЛЯ РАБОТЫ С НЕСКОЛЬКИМИ ПРЕПЯТСТВИЯМИ
    # =========================================================================

    def _create_obstacle(self, position, size=1.0, index=0):
        """
        Создание нового препятствия в мире Webots программно.

        Параметры:
            position: [x, y, z] координаты препятствия
            size: размер препятствия (куб со стороной size)
            index: индекс препятствия для идентификации

        Возвращает:
            node: объект созданного препятствия или None при ошибке
        """
        try:
            # Получаем корневой узел сцены
            root = self.car.getRoot()
            children = root.getField("children")

            # Случайный цвет для визуального разнообразия
            color = [random.uniform(0.5, 1.0),
                     random.uniform(0, 0.5),
                     random.uniform(0, 0.5)]

            # Формируем строку описания препятствия на языке VRML
            obstacle_str = f"""
            DEF OBSTACLE_{index} Solid {{
                translation {position[0]} {position[1]} {position[2]}
                children [
                    Shape {{
                        appearance Appearance {{
                            material Material {{
                                diffuseColor {color[0]} {color[1]} {color[2]}
                            }}
                        }}
                        geometry Box {{
                            size {size} {size} {size}
                        }}
                    }}
                ]
                name "obstacle_{index}"
                boundingObject Box {{
                    size {size} {size} {size}
                }}
            }}
            """

            # Добавляем препятствие в сцену
            children.importMFNodeFromString(-1, obstacle_str)

            # Находим созданный объект по DEF
            obs_node = self.car.getFromDef(f"OBSTACLE_{index}")
            return obs_node
        except Exception as e:
            print(f"Ошибка создания препятствия: {e}")
            return None

    def _remove_all_obstacles(self):
        """
        Удаление всех ранее созданных препятствий из мира.
        """
        if hasattr(self, 'obstacle_nodes'):
            for obs_node in self.obstacle_nodes:
                if obs_node:
                    try:
                        obs_node.remove()
                    except:
                        pass
        self.obstacle_nodes = []
        self.obstacle_sizes = []

    def _generate_random_position(self, for_target=True, exclude_positions=None):
        """
        Генерация случайной позиции для объекта с проверкой ограничений.

        Параметры:
            for_target: True для цели, False для препятствия
            exclude_positions: список позиций, которых нужно избегать

        Возвращает:
            list: [x, y, z] координаты
        """
        max_attempts = 100
        attempt = 0

        while attempt < max_attempts:
            if for_target:
                x = random.uniform(*self.TARGET_X_RANGE)
                y = random.uniform(*self.TARGET_Y_RANGE)
                z = self.TARGET_Z
            else:
                x = random.uniform(*self.OBSTACLE_X_RANGE)
                y = random.uniform(*self.OBSTACLE_Y_RANGE)
                z = self.OBSTACLE_Z

            # Проверка расстояния от старта
            dist_from_start = math.sqrt(
                (x - self.START_POS[0]) ** 2 +
                (y - self.START_POS[1]) ** 2
            )

            if dist_from_start < self.MIN_DIST_FROM_START:
                attempt += 1
                continue

            # Проверка расстояния от других объектов
            valid = True
            if exclude_positions:
                for pos in exclude_positions:
                    # Сравниваем только X и Y (горизонтальная плоскость)
                    dist = math.sqrt(
                        (x - pos[0]) ** 2 +
                        (y - pos[1]) ** 2
                    )
                    if dist < self.MIN_DIST_BETWEEN_OBJ:
                        valid = False
                        break

            if valid:
                return [x, y, z]

            attempt += 1

        # Запасной вариант, если не удалось найти подходящую позицию
        print(f"⚠️ Не удалось найти позицию, использую значение по умолчанию")
        if for_target:
            return [10, 0.5, 0.4]
        else:
            return [0, 0, 1]

    def _randomize_scenario(self):
        """
        Создание нового случайного сценария с несколькими препятствиями.

        Генерирует новые позиции для всех препятствий и цели,
        удаляет старые препятствия и создаёт новые.
        """
        if not self.randomize_scenarios:
            return

        print("\n  🎲 Генерация нового сценария с несколькими препятствиями...")

        # Удаляем старые препятствия
        self._remove_all_obstacles()

        # Список для хранения позиций всех препятствий
        all_obstacle_positions = []

        # Генерируем несколько препятствий
        for i in range(self.num_obstacles):
            # Случайный размер
            size = random.uniform(*self.OBSTACLE_SIZE_RANGE)

            # Генерируем позицию с учётом уже созданных препятствий и старта
            obs_pos = self._generate_random_position(
                for_target=False,
                exclude_positions=all_obstacle_positions + [self.START_POS]
            )

            # Создаём препятствие
            obs_node = self._create_obstacle(obs_pos, size, i)

            if obs_node:
                self.obstacle_nodes.append(obs_node)
                self.obstacle_sizes.append(size)
                all_obstacle_positions.append(obs_pos)
                print(f"     Препятствие {i + 1}: X={obs_pos[0]:5.1f}, Y={obs_pos[1]:5.1f}, размер={size:.1f}м")

        # Генерируем позицию цели (избегая всех препятствий)
        target_pos = self._generate_random_position(
            for_target=True,
            exclude_positions=all_obstacle_positions
        )

        # Перемещаем цель
        if self.target_node:
            try:
                self.target_node.getField("translation").setSFVec3f(target_pos)
                print(f"     Цель: X={target_pos[0]:5.1f}, Y={target_pos[1]:5.1f}")
            except Exception as e:
                print(f"     Ошибка перемещения цели: {e}")
        elif self.use_virtual_target:
            self.virtual_target_pos = target_pos

        # Определяем сложность сценария по минимальному расстоянию до цели
        min_dist_to_target = float('inf')
        for obs_pos in all_obstacle_positions:
            dist = math.sqrt(
                (obs_pos[0] - target_pos[0]) ** 2 +
                (obs_pos[1] - target_pos[1]) ** 2
            )
            min_dist_to_target = min(min_dist_to_target, dist)

        if min_dist_to_target > 8:
            difficulty = "easy"
        elif min_dist_to_target > 4:
            difficulty = "medium"
        else:
            difficulty = "hard"

        self.scenario_stats[difficulty] += 1
        print(f"     Сложность: {difficulty.upper()} (мин. дист. до цели: {min_dist_to_target:.1f}м)")

        # Вывод статистики каждые 100 эпизодов
        if self.episode_count % 100 == 0 and self.episode_count > 0:
            total = sum(self.scenario_stats.values())
            print(f"\n  📊 Статистика сценариев:")
            for diff, count in self.scenario_stats.items():
                percentage = (count / total) * 100 if total > 0 else 0
                print(f"     {diff}: {count} ({percentage:.1f}%)")

    # =========================================================================
    # МЕТОДЫ ДЛЯ РАСЧЁТА НАГРАДЫ
    # =========================================================================

    def _calculate_reward(self, action):
        """
        Расчёт награды за текущий шаг.

        Система наград поощряет:
        - Движение к цели (прогресс)
        - Правильное направление
        - Использование заднего хода в нужных ситуациях
        - Достижение цели

        Штрафует:
        - Удаление от цели
        - Близость к препятствиям
        - Резкие манёвры
        - Бездействие

        Параметры:
            action: действие, применённое агентом [steering, speed]

        Возвращает:
            tuple: (награда, достигнута_ли_цель, столкновение, информация, скорость)
        """
        is_goal, dist_to_target = self._check_goal_3d()
        is_collision, collision_info, min_lidar = self._check_collision_lidar()

        # Текущая скорость с учётом направления
        current_speed = 0.0
        if self.car_node:
            try:
                vel = self.car_node.getVelocity()
                if vel:
                    current_speed = vel[2]  # Z - вперёд/назад
            except:
                pass

        reward = 0.0

        # Базовый штраф за время (побуждает к быстрым решениям)
        reward -= 0.1

        # Награда за любое движение (вперёд или назад)
        if abs(current_speed) > 0.5:
            reward += 1.0
        else:
            reward -= 1.0  # Штраф за остановку

        # Прогресс к цели (главный источник награды)
        if hasattr(self, 'prev_dist_to_target'):
            progress = self.prev_dist_to_target - dist_to_target
            if progress > 0.01:  # Приближаемся
                reward += progress * 50.0
            elif progress < -0.01:  # Удаляемся
                reward += progress * 20.0  # Отрицательная награда

        # Бонус за правильное использование заднего хода
        angle_to_target = abs(self._get_angle_to_target())

        # Если цель сзади (угол > 90 градусов) и близко
        if angle_to_target > 1.5 and dist_to_target < 5.0:
            if current_speed < -0.5:  # Едет назад
                reward += 5.0  # Молодец, что сдаёшь назад
            elif current_speed > 0.5:  # Едет вперёд от цели
                reward -= 3.0  # Плохо, удаляешься

        # Если цель спереди (угол < 45 градусов)
        elif angle_to_target < 0.8:
            if current_speed > 0.5:  # Едет вперёд
                reward += 2.0
            elif current_speed < -0.5:  # Едет назад от цели
                reward -= 3.0

        # Штраф за резкую смену направления (плавность вождения)
        if hasattr(self, 'prev_speed'):
            speed_change = abs(current_speed - self.prev_speed)
            if speed_change > 8.0:  # Очень резко
                reward -= 3.0
            elif speed_change > 3.0:  # Умеренно резко
                reward -= 1.0

        # Штраф за близость к препятствию
        if min_lidar < 1.5:
            danger = (1.5 - min_lidar) / 1.5
            reward -= danger * 15.0

        # Бонус за новый рекорд (самое близкое расстояние к цели)
        if dist_to_target < self.best_distance:
            self.best_distance = dist_to_target
            reward += 10.0

        self.prev_dist_to_target = dist_to_target
        self.prev_speed = current_speed

        if math.isnan(reward) or math.isinf(reward):
            reward = 0.0

        return reward, is_goal, is_collision, collision_info, current_speed

    # =========================================================================
    # ОСНОВНЫЕ МЕТОДЫ GYMNASIUM
    # =========================================================================

    def reset(self, seed=None, options=None):
        """
        Сброс среды к начальному состоянию.

        Вызывается в начале каждого эпизода.
        Телепортирует автомобиль на старт, генерирует новый сценарий,
        сбрасывает счётчики.

        Параметры:
            seed: сид для генератора случайных чисел
            options: дополнительные опции

        Возвращает:
            tuple: (наблюдение, информация)
        """
        super().reset(seed=seed)

        # Генерация нового случайного сценария
        if self.randomize_scenarios:
            self._randomize_scenario()

        self.episode_count += 1

        # Вывод статистики предыдущего эпизода
        if self.episode_count > 1:
            success_rate = (self.success_count / (self.episode_count - 1)) * 100 if self.episode_count > 1 else 0
            avg_reward = self.episode_reward / max(self.step_count, 1)
            print(f"\nЭпизод {self.episode_count - 1}: награда={self.episode_reward:.1f}, "
                  f"средняя={avg_reward:.2f}/шаг, успех={success_rate:.1f}%")

        print(f"\nСброс эпизода {self.episode_count}...")

        # Остановка автомобиля
        self.car.setCruisingSpeed(0)
        self.car.setSteeringAngle(0)

        # Сброс физики симуляции
        self.car.simulationResetPhysics()
        for _ in range(5):
            self.car.step()

        # Телепортация на стартовую позицию
        if self.car_node:
            try:
                # Сначала устанавливаем поворот
                self.car_node.getField("rotation").setSFRotation(self.START_ROT)
                for _ in range(3):
                    self.car.step()
                # Затем позицию
                self.car_node.getField("translation").setSFVec3f(self.START_POS)
                for _ in range(10):
                    self.car.step()
            except Exception as e:
                print(f"Ошибка телепортации: {e}")

        # Стабилизация после телепортации
        for _ in range(20):
            self.car.setCruisingSpeed(0)
            self.car.setSteeringAngle(0)
            self.car.step()

        # Сброс счётчиков
        self.step_count = 0
        self.episode_reward = 0
        self.prev_action = np.array([0.0, 0.0], dtype=np.float32)
        self.prev_speed = 0.0
        self.prev_dist_to_target = self._check_goal_3d()[1]
        self.best_distance = float('inf')

        observation = self._get_observation()
        return observation, {}

    def step(self, action):
        """
        Выполнение одного шага симуляции.

        Применяет действие агента, обновляет состояние среды,
        вычисляет награду и проверяет завершение эпизода.

        Параметры:
            action: действие агента [steering, speed]

        Возвращает:
            tuple: (наблюдение, награда, завершено, обрезано, информация)
        """
        # Очистка действия от NaN и бесконечности
        action = np.array(action, dtype=np.float32)
        action = np.nan_to_num(action, nan=0.0, posinf=0.8, neginf=-0.8)

        # Исследование только в первых 20 эпизодах (для начального обучения)
        if self.episode_count < 20:
            action[0] += np.random.normal(0, 0.2)
            action[1] += np.random.normal(0, 2)

        # Сглаживание управления для плавности
        smoothed_steering = 0.7 * self.prev_action[0] + 0.3 * action[0]
        steering = float(np.clip(smoothed_steering, -0.8, 0.8))

        smoothed_speed = 0.7 * self.prev_speed + 0.3 * action[1]
        speed = float(np.clip(smoothed_speed, -5.0, 20.0))

        self.prev_action = np.array([steering, speed], dtype=np.float32)

        # Применение действия к автомобилю
        self.car.setSteeringAngle(steering)
        self.car.setCruisingSpeed(speed)
        self.car.step()

        # Получение нового наблюдения и награды
        observation = self._get_observation()
        reward, is_goal, is_collision, collision_info, current_speed = self._calculate_reward(action)

        terminated = False
        truncated = False
        info = {
            'reason': 'unknown',
            'final_distance': self.prev_dist_to_target if (is_goal or is_collision) else None,
            'current_speed': current_speed,
            'min_lidar': min(
                self._get_lidar_min_distance(self.lidar_front),
                self._get_lidar_min_distance(self.lidar_left),
                self._get_lidar_min_distance(self.lidar_right)
            ),
            'angle_to_target': abs(self._get_angle_to_target()),
            'dist_to_target': self.prev_dist_to_target
        }

        # Проверка достижения цели
        if is_goal:
            reward += 500
            terminated = True
            self.success_count += 1
            info['reason'] = 'goal'
            direction = "вперёд" if current_speed >= 0 else "назад"
            print(
                f"\nGOAL! dist={self.prev_dist_to_target:.2f}m, направление={direction}, reward={self.episode_reward + reward:.1f}")

        # Проверка столкновения
        elif is_collision:
            reward -= 200
            terminated = True
            self.collision_count += 1
            info['reason'] = 'collision'
            print(f"\nCOLLISION! {collision_info}, reward={self.episode_reward + reward:.1f}")

        # Проверка таймаута
        elif self.step_count >= self.MAX_STEPS:
            reward -= 50
            truncated = True
            info['reason'] = 'timeout'
            print(f"\nTIMEOUT! dist={self.prev_dist_to_target:.1f}m, reward={self.episode_reward + reward:.1f}")

        self.step_count += 1
        self.episode_reward += reward

        # Логирование каждые 100 шагов
        if self.step_count % 100 == 0:
            _, dist_to_target = self._check_goal_3d()
            _, _, min_lidar = self._check_collision_lidar()
            direction = "ВПЕРЁД" if current_speed >= 0 else "НАЗАД"
            print(f"  Step {self.step_count:4d}: r={reward:6.1f} total={self.episode_reward:7.1f} "
                  f"speed={current_speed:5.2f} {direction} goal={dist_to_target:5.1f}m lidar={min_lidar:4.1f}m")

        return observation, reward, terminated, truncated, info

    def render(self):
        """Метод рендеринга (не используется, Webots рендерит сам)."""
        pass

    def close(self):
        """
        Закрытие среды и освобождение ресурсов.

        Удаляет все созданные препятствия и выводит итоговую статистику.
        """
        self._remove_all_obstacles()
        print("Закрытие среды")
        if self.randomize_scenarios:
            total = sum(self.scenario_stats.values())
            print("\n📊 ИТОГОВАЯ СТАТИСТИКА СЦЕНАРИЕВ:")
            for diff, count in self.scenario_stats.items():
                percentage = (count / total) * 100 if total > 0 else 0
                print(f"   {diff}: {count} ({percentage:.1f}%)")


# =============================================================================
# КОЛБЭК ДЛЯ ЛОГИРОВАНИЯ МЕТРИК
# =============================================================================
class ModelMetricsCallback(BaseCallback):
    """
    Колбэк для логирования метрик модели в TensorBoard.

    Собирает и записывает:
    - Среднюю награду за эпизод
    - Длину эпизода
    - Процент успехов, столкновений и таймаутов
    - Финальное расстояние до цели
    """

    def __init__(self, verbose=0):
        """
        Инициализация колбэка.

        Параметры:
            verbose: уровень детализации вывода
        """
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_success = []
        self.episode_collisions = []
        self.episode_timeouts = []

    def _on_step(self) -> bool:
        """
        Вызывается на каждом шаге обучения.
        Проверяет завершение эпизодов и логирует метрики.

        Возвращает:
            bool: True для продолжения обучения
        """
        if self.locals.get("dones") and any(self.locals["dones"]):
            dones = self.locals["dones"]
            infos = self.locals["infos"]

            for i, done in enumerate(dones):
                if done and infos[i]:
                    info = infos[i]

                    # Логирование информации об эпизоде
                    if "episode" in info:
                        ep_info = info["episode"]
                        self.episode_rewards.append(ep_info["r"])
                        self.episode_lengths.append(ep_info["l"])

                        self.logger.record("rollout/ep_rew_mean",
                                           np.mean(self.episode_rewards[-100:]))
                        self.logger.record("rollout/ep_len_mean",
                                           np.mean(self.episode_lengths[-100:]))
                        self.logger.record("rollout/ep_rew_max",
                                           np.max(self.episode_rewards[-100:]))
                        self.logger.record("rollout/ep_rew_min",
                                           np.min(self.episode_rewards[-100:]))

                    # Логирование причины завершения
                    if "reason" in info:
                        reason = info["reason"]
                        self.logger.record(f"terminal/{reason}", 1)

                        if reason == "goal":
                            self.episode_success.append(1)
                            self.episode_collisions.append(0)
                            self.episode_timeouts.append(0)
                        elif reason == "collision":
                            self.episode_success.append(0)
                            self.episode_collisions.append(1)
                            self.episode_timeouts.append(0)
                        elif reason == "timeout":
                            self.episode_success.append(0)
                            self.episode_collisions.append(0)
                            self.episode_timeouts.append(1)
                        else:
                            self.episode_success.append(0)
                            self.episode_collisions.append(0)
                            self.episode_timeouts.append(0)

                        if len(self.episode_success) > 0:
                            self.logger.record("stats/success_rate",
                                               np.mean(self.episode_success[-100:]) * 100)
                            self.logger.record("stats/collision_rate",
                                               np.mean(self.episode_collisions[-100:]) * 100)
                            self.logger.record("stats/timeout_rate",
                                               np.mean(self.episode_timeouts[-100:]) * 100)

                    # Логирование финального расстояния
                    if "final_distance" in info and info["final_distance"] is not None:
                        self.logger.record("environment/final_distance", info["final_distance"])

        return True


# =============================================================================
# ФУНКЦИЯ ОБУЧЕНИЯ МОДЕЛИ
# =============================================================================
def train():
    """
    Основная функция для запуска обучения.

    Создаёт среду, инициализирует модель PPO,
    запускает обучение и сохраняет результаты.
    """
    print("\n" + "=" * 60)
    print("ЗАПУСК ОБУЧЕНИЯ (С РАНДОМИЗАЦИЕЙ И ЗАДНИМ ХОДОМ)")
    print("=" * 60)

    # Создание среды
    env = BMWRLEnvironment()

    # =========================================================================
    # ПАРАМЕТРЫ АРХИТЕКТУРЫ НЕЙРОСЕТИ
    # =========================================================================
    policy_kwargs = dict(
        net_arch=[256, 256],  # Два скрытых слоя по 256 нейронов
        activation_fn=th.nn.ReLU,  # Функция активации ReLU
    )

    # =========================================================================
    # ПАРАМЕТРЫ АЛГОРИТМА PPO
    # =========================================================================
    # learning_rate: скорость обучения (3e-4 = 0.0003)
    # n_steps: количество шагов между обновлениями политики
    # batch_size: размер порции данных для обучения
    # n_epochs: количество эпох обучения на каждом обновлении
    # gamma: коэффициент дисконтирования (важность будущих наград)
    # gae_lambda: параметр для GAE (Generalized Advantage Estimation)
    # clip_range: ограничение на изменение политики
    # ent_coef: коэффициент энтропии (поощрение исследования)
    # vf_coef: коэффициент функции ценности
    # max_grad_norm: максимальная норма градиента (для стабильности)

    model = PPO(
        "MlpPolicy",  # Тип политики (MLP = многослойный персептрон)
        env,  # Среда для обучения
        verbose=1,  # Уровень детализации вывода
        learning_rate=3e-4,  # Скорость обучения
        n_steps=2048,  # Шагов между обновлениями
        batch_size=64,  # Размер батча
        n_epochs=10,  # Количество эпох
        gamma=0.99,  # Коэффициент дисконтирования
        gae_lambda=0.95,  # Параметр GAE
        clip_range=0.2,  # Ограничение клиппинга
        ent_coef=0.01,  # Коэффициент энтропии
        vf_coef=0.5,  # Коэффициент функции ценности
        max_grad_norm=0.5,  # Максимальная норма градиента
        tensorboard_log="./logs/",  # Директория для логов TensorBoard
        policy_kwargs=policy_kwargs,  # Параметры архитектуры
        device="cpu"  # Устройство для вычислений (cpu/cuda)
    )

    # Создание директории для сохранения моделей
    os.makedirs("./models", exist_ok=True)

    print("\nНачинаем обучение на 1,500,000 шагов...")
    print("✅ Рандомизация сценариев ВКЛЮЧЕНА")
    print("✅ Задний ход ВКЛЮЧЕН (скорость от -5 до 20)")
    print(f"✅ Количество препятствий: {env.num_obstacles}")
    print("🎲 Каждый эпизод - новая позиция препятствий и цели")
    print("📊 Все метрики модели логируются в TensorBoard")
    print("⏱️  Ожидаемое время: ~3-4 часа")
    print("Нажмите Ctrl+C для остановки\n")
    print("💡 Запустите TensorBoard: tensorboard --logdir=./logs\n")

    try:
        # Запуск обучения
        model.learn(
            total_timesteps=1_500_000,  # Общее количество шагов обучения
            tb_log_name="BMW_X5_RL_REVERSE",  # Имя для логов TensorBoard
            reset_num_timesteps=False,  # Не сбрасывать счётчик шагов
            callback=ModelMetricsCallback()  # Колбэк для логирования
        )

        # Сохранение финальной модели
        final_path = "./models/bmw_rl_reverse()"
        model.save(final_path)
        print(f"\nФинальная модель сохранена: {final_path}")

    except KeyboardInterrupt:
        print("\n\nОбучение прервано пользователем")
        interrupt_path = "./models/bmw_rl_reverse_interrupted"
        model.save(interrupt_path)
        print(f"Модель сохранена: {interrupt_path}")

    # Вывод финальной статистики
    print("\n" + "=" * 60)
    print("ФИНАЛЬНАЯ СТАТИСТИКА")
    print("=" * 60)
    print(f"Всего эпизодов: {env.episode_count}")
    print(f"Успехов: {env.success_count}")
    print(f"Столкновений: {env.collision_count}")
    if env.episode_count > 0:
        success_rate = (env.success_count / env.episode_count) * 100
        print(f"Успешность: {success_rate:.1f}%")

    return model


# =============================================================================
# ФУНКЦИЯ ТЕСТИРОВАНИЯ МОДЕЛИ
# =============================================================================
def test(model_path="./models/bmw_rl(random 1-4).zip"):
    """
    Функция для тестирования обученной модели.

    Параметры:
        model_path: путь к файлу с сохранённой моделью
    """
    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ МОДЕЛИ")
    print("=" * 60)

    env = BMWRLEnvironment()

    # Проверка существования файла модели
    if os.path.exists(model_path):
        model = PPO.load(model_path)
        print(f"Модель загружена: {model_path}")
    else:
        print(f"Модель не найдена: {model_path}")
        return

    input("\nНажмите Enter для начала тестирования...")

    obs, _ = env.reset()
    total_reward = 0
    step = 0

    while True:
        # Получение детерминированного действия от модели
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        # Логирование каждые 50 шагов
        if step % 50 == 0:
            _, dist_target = env._check_goal_3d()
            _, _, min_lidar = env._check_collision_lidar()
            print(f"Step {step}: reward={reward:.2f}, total={total_reward:.1f}, "
                  f"goal={dist_target:.1f}m, lidar={min_lidar:.1f}m")

        # Завершение эпизода
        if terminated or truncated:
            print(f"\n📊 Эпизод завершен! Шагов: {step}, награда: {total_reward:.1f}")
            print(f"Причина: {info.get('reason', 'unknown')}")
            break


# =============================================================================
# ТОЧКА ВХОДА
# =============================================================================
if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            model_path = sys.argv[2] if len(sys.argv) > 2 else "./models/bmw_rl_reverse_final.zip"
            test(model_path)
        else:
            print("Использование:")
            print("  python rl_controller.py           # обучение")
            print("  python rl_controller.py --test    # тестирование")
    else:
        train()