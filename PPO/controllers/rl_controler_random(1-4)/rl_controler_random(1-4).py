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
from vehicle import Car
import numpy as np
import math
import os
import sys
import random
from collections import deque
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import torch as th


# =============================================================================
# КЛАСС СРЕДЫ ДЛЯ ОБУЧЕНИЯ
# =============================================================================
class BMWRLEnvironment(gym.Env):
    """
    Среда для обучения с подкреплением автомобиля BMW X5 в Webots.

    Особенности:
    - Рандомизация положения препятствий и цели в каждом эпизоде
    - Поддержка заднего хода (отрицательные значения скорости)
    - Детектирование столкновений по лидарам
    - Улучшенная система наград против зигзагов
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode="human"):
        super().__init__()

        print("=" * 60)
        print("BMW X5 — СРЕДА ДЛЯ ОБУЧЕНИЯ С ПОДКРЕПЛЕНИЕМ")
        print("=" * 60)

        # =====================================================================
        # ИНИЦИАЛИЗАЦИЯ WEBOTS
        # =====================================================================
        self.car = Car()
        self.timestep = int(self.car.getBasicTimeStep())

        # =====================================================================
        # ПОИСК НОД В МИРЕ WEBOTS
        # =====================================================================
        print("\nПоиск нод в мире:")

        self.car_node = self.car.getFromDef("CAR")
        if self.car_node:
            print(f"  CAR: найдена")
        else:
            print(f"  CAR: не найдена, использую getSelf()")
            self.car_node = self.car.getSelf()

        self.obstacle_node = self.car.getFromDef("OBSTACLE")
        if self.obstacle_node:
            print(f"  OBSTACLE: найдена")
        else:
            print(f"  OBSTACLE: не найдена")

        self.target_node = self.car.getFromDef("TARGET")
        self.use_virtual_target = False

        if self.target_node:
            target_pos = self.target_node.getPosition()
            print(f"TARGET: найдена на ({target_pos[0]:.1f}, {target_pos[1]:.1f}, {target_pos[2]:.1f})")
            try:
                self.target_node.removeBoundingObject()
                self.target_node.enableContactPointsTracking(False)
                self.target_node.resetPhysics()
                print("Физика цели отключена")
            except Exception as e:
                print(f" Не удалось отключить физику: {e}")
        else:
            print(f"TARGET: не найдена, использую виртуальную цель")
            self.use_virtual_target = True
            self.virtual_target_pos = [10, 0.5, 0.4]

        print()

        # =====================================================================
        # ПАРАМЕТРЫ СРЕДЫ
        # =====================================================================
        self.START_POS = [-11.5, 0.5, 0.5]
        self.START_ROT = [0, 1, 0, 0]
        self.GOAL_DISTANCE = 4.0
        self.COLLISION_DISTANCE = 0.5
        self.MAX_STEPS = 2500

        # =====================================================================
        # ПАРАМЕТРЫ РАНДОМИЗАЦИИ СЦЕНАРИЕВ
        # =====================================================================
        self.randomize_scenarios = True
        self.OBSTACLE_X_RANGE = (-8, 8)
        self.OBSTACLE_Y_RANGE = (-8, 8)
        self.OBSTACLE_Z = 0.5
        self.TARGET_X_RANGE = (-12, 12)
        self.TARGET_Y_RANGE = (-10, 10)
        self.TARGET_Z = 0.5
        self.MIN_DIST_FROM_START = 6.0
        self.MIN_DIST_BETWEEN_OBJ = 3.0

        # =====================================================================
        # ПАРАМЕТРЫ ДЛЯ НЕСКОЛЬКИХ ПРЕПЯТСТВИЙ
        # =====================================================================
        self.multiple_obstacles = True
        self.num_obstacles = 4
        self.obstacle_nodes = []
        self.obstacle_sizes = []
        self.OBSTACLE_SIZE_RANGE = (0.8, 2.0)

        # =====================================================================
        # СТАТИСТИКА СЦЕНАРИЕВ
        # =====================================================================
        self.scenario_stats = {'easy': 0, 'medium': 0, 'hard': 0}

        # =====================================================================
        # ИНИЦИАЛИЗАЦИЯ СЕНСОРОВ
        # =====================================================================
        self._setup_sensors()

        # =====================================================================
        # ОПРЕДЕЛЕНИЕ ПРОСТРАНСТВА НАБЛЮДЕНИЙ
        # =====================================================================
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
        self.action_space = spaces.Box(
            low=np.array([-0.8, -5.0], dtype=np.float32),
            high=np.array([0.8, 20.0], dtype=np.float32),
            dtype=np.float32
        )

        # =====================================================================
        # СЧЁТЧИКИ ДЛЯ СТАТИСТИКИ
        # =====================================================================
        self.step_count = 0
        self.episode_reward = 0
        self.episode_count = 0
        self.success_count = 0
        self.collision_count = 0
        self.prev_action = np.array([0.0, 0.0], dtype=np.float32)
        self.prev_speed = 0.0
        self.best_distance = float('inf')

        # НОВОЕ: для отслеживания зигзагов и прогресса
        self.steps_without_progress = 0
        self.action_history = deque(maxlen=50)
        self.last_progress_distance = float('inf')
        self.steering_changes = 0
        self.last_steering = 0.0

        self.render_mode = render_mode

        print(f"\n✅ Goal distance: {self.GOAL_DISTANCE}m")
        print(f"✅ Collision distance: {self.COLLISION_DISTANCE}m")
        print(f"✅ Max steps: {self.MAX_STEPS}")
        print(f"✅ Randomization: {'ON' if self.randomize_scenarios else 'OFF'}")
        print(f"✅ Reverse gear: ENABLED (speed: -5 to 20)")
        print(f"✅ Multiple obstacles: {self.num_obstacles}")
        print("=" * 60)

    # =========================================================================
    # МЕТОДЫ ДЛЯ РАБОТЫ С СЕНСОРАМИ
    # =========================================================================
    def _setup_sensors(self):
        """Инициализация и включение всех сенсоров"""
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
        """Очистка значения от NaN и бесконечности"""
        if value is None or not isinstance(value, (int, float)):
            return default
        if math.isnan(value) or math.isinf(value):
            return default
        return float(np.clip(value, min_val, max_val))

    def _get_lidar_min_distance(self, lidar):
        """Получение минимального расстояния с лидара"""
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
        """Получение текущей позиции автомобиля из GPS"""
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
        """Получение текущего угла поворота автомобиля"""
        if self.imu:
            try:
                rpy = self.imu.getRollPitchYaw()
                if rpy and len(rpy) >= 3:
                    return self._sanitize_value(rpy[2], 0, -np.pi, np.pi)
            except:
                pass
        return 0.0

    def _get_target_position(self):
        """Получение позиции цели"""
        if self.use_virtual_target:
            return self.virtual_target_pos
        elif self.target_node:
            return self.target_node.getPosition()
        return [10, 0.5, 0.4]

    def _get_angle_to_target(self):
        """Вычисление угла до цели"""
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
        """Проверка достижения цели"""
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
        """Проверка столкновения с препятствиями"""
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
        """Формирование вектора наблюдения"""
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

        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            obs = np.nan_to_num(obs, nan=0.0, posinf=25.0, neginf=-25.0)

        return obs

    # =========================================================================
    # МЕТОДЫ ДЛЯ РАБОТЫ С НЕСКОЛЬКИМИ ПРЕПЯТСТВИЯМИ
    # =========================================================================
    def _create_obstacle(self, position, size=1.0, index=0):
        """Создание нового препятствия в мире Webots"""
        try:
            root = self.car.getRoot()
            children = root.getField("children")
            color = [random.uniform(0.5, 1.0),
                     random.uniform(0, 0.5),
                     random.uniform(0, 0.5)]

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

            children.importMFNodeFromString(-1, obstacle_str)
            obs_node = self.car.getFromDef(f"OBSTACLE_{index}")
            return obs_node
        except Exception as e:
            print(f"Ошибка создания препятствия: {e}")
            return None

    def _remove_all_obstacles(self):
        """Удаление всех созданных препятствий"""
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
        """Генерация случайной позиции с проверкой ограничений"""
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

            dist_from_start = math.sqrt(
                (x - self.START_POS[0]) ** 2 +
                (y - self.START_POS[1]) ** 2
            )

            if dist_from_start < self.MIN_DIST_FROM_START:
                attempt += 1
                continue

            valid = True
            if exclude_positions:
                for pos in exclude_positions:
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

        print(f"⚠️ Не удалось найти позицию, использую значение по умолчанию")
        if for_target:
            return [10, 0.5, 0.4]
        else:
            return [0, 0, 1]

    def _randomize_scenario(self):
        """Создание нового случайного сценария"""
        if not self.randomize_scenarios:
            return

        print("\n  🎲 Генерация нового сценария с несколькими препятствиями...")
        self._remove_all_obstacles()

        all_obstacle_positions = []
        for i in range(self.num_obstacles):
            size = random.uniform(*self.OBSTACLE_SIZE_RANGE)
            obs_pos = self._generate_random_position(
                for_target=False,
                exclude_positions=all_obstacle_positions + [self.START_POS]
            )
            obs_node = self._create_obstacle(obs_pos, size, i)
            if obs_node:
                self.obstacle_nodes.append(obs_node)
                self.obstacle_sizes.append(size)
                all_obstacle_positions.append(obs_pos)
                print(f"     Препятствие {i + 1}: X={obs_pos[0]:5.1f}, Y={obs_pos[1]:5.1f}, размер={size:.1f}м")

        target_pos = self._generate_random_position(
            for_target=True,
            exclude_positions=all_obstacle_positions
        )

        if self.target_node:
            try:
                self.target_node.getField("translation").setSFVec3f(target_pos)
                print(f"     Цель: X={target_pos[0]:5.1f}, Y={target_pos[1]:5.1f}")
            except Exception as e:
                print(f"     Ошибка перемещения цели: {e}")
        elif self.use_virtual_target:
            self.virtual_target_pos = target_pos

        if all_obstacle_positions:
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

    # =========================================================================
    # УЛУЧШЕННЫЙ МЕТОД РАСЧЁТА НАГРАДЫ (ПРОТИВ ЗИГЗАГОВ)
    # =========================================================================
    def _calculate_reward(self, action):
        """
        СБАЛАНСИРОВАННАЯ система наград

        Принципы:
        1. Машина должна получать маленькую награду за само движение
        2. Прогресс к цели - главный источник награды, но не доминирующий
        3. Штрафы должны быть умеренными, чтобы не парализовать агента
        """
        is_goal, dist_to_target = self._check_goal_3d()
        is_collision, collision_info, min_lidar = self._check_collision_lidar()

        # Текущая скорость
        current_speed = 0.0
        if self.car_node:
            try:
                vel = self.car_node.getVelocity()
                if vel:
                    current_speed = vel[2]
            except:
                pass

        reward = 0.0

        # =====================================================================
        # 1. НАГРАДА ЗА ДВИЖЕНИЕ (УМЕРЕННАЯ)
        # =====================================================================
        # Без этой награды агент боится двигаться
        if abs(current_speed) > 0.5:
            reward += 0.5  # Небольшая награда за любое движение
        else:
            reward -= 0.2  # Мягкий штраф за остановку

        # =====================================================================
        # 2. БАЗОВЫЙ ШТРАФ ЗА ВРЕМЯ (МЯГКИЙ)
        # =====================================================================
        reward -= 0.1  # Вернул к исходному значению

        # =====================================================================
        # 3. ПРОГРЕСС К ЦЕЛИ (УМЕРЕННЫЙ)
        # =====================================================================
        if hasattr(self, 'prev_dist_to_target'):
            progress = self.prev_dist_to_target - dist_to_target

            if progress > 0.01:  # Приближаемся
                reward += progress * 30.0  # Уменьшил с 80
                self.steps_without_progress = 0
            elif progress < -0.01:  # Удаляемся
                reward += progress * 15.0  # Уменьшил с 40
                self.steps_without_progress += 1
            else:
                self.steps_without_progress += 1

        # =====================================================================
        # 4. ШТРАФ ЗА ОТСУТСТВИЕ ПРОГРЕССА (МЯГКИЙ)
        # =====================================================================
        if self.steps_without_progress > 100:  # Увеличил порог
            reward -= 1.0  # Уменьшил с 5
        elif self.steps_without_progress > 50:
            reward -= 0.5  # Уменьшил с 2

        # =====================================================================
        # 5. НАГРАДА ЗА НАПРАВЛЕНИЕ
        # =====================================================================
        angle_to_target = abs(self._get_angle_to_target())
        # Небольшой бонус за правильное направление
        angle_score = 1.0 - min(angle_to_target / math.pi, 1.0)
        reward += angle_score * 0.5  # Уменьшил с 2

        # =====================================================================
        # 6. ШТРАФ ЗА БЛИЗОСТЬ К ПРЕПЯТСТВИЮ
        # =====================================================================
        if min_lidar < 1.5:
            danger = (1.5 - min_lidar) / 1.5
            reward -= danger * 5.0  # Уменьшил с 12

        # =====================================================================
        # 7. БОНУС ЗА НОВЫЙ РЕКОРД
        # =====================================================================
        if dist_to_target < self.best_distance:
            improvement = self.best_distance - dist_to_target
            self.best_distance = dist_to_target
            reward += improvement * 20.0  # Уменьшил с 50

        self.prev_dist_to_target = dist_to_target
        self.prev_speed = current_speed

        if math.isnan(reward) or math.isinf(reward):
            reward = 0.0

        return reward, is_goal, is_collision, collision_info, current_speed

    # =========================================================================
    # ОСНОВНЫЕ МЕТОДЫ GYMNASIUM
    # =========================================================================
    def reset(self, seed=None, options=None):
        """Сброс среды к начальному состоянию"""
        super().reset(seed=seed)

        if self.randomize_scenarios:
            self._randomize_scenario()

        self.episode_count += 1

        if self.episode_count > 1:
            success_rate = (self.success_count / (self.episode_count - 1)) * 100 if self.episode_count > 1 else 0
            avg_reward = self.episode_reward / max(self.step_count, 1)
            print(f"\n📊 Эпизод {self.episode_count - 1}: награда={self.episode_reward:.1f}, "
                  f"средняя={avg_reward:.2f}/шаг, успех={success_rate:.1f}%")

        print(f"\n🔄 Сброс эпизода {self.episode_count}...")

        self.car.setCruisingSpeed(0)
        self.car.setSteeringAngle(0)

        self.car.simulationResetPhysics()
        for _ in range(5):
            self.car.step()

        if self.car_node:
            try:
                self.car_node.getField("rotation").setSFRotation(self.START_ROT)
                for _ in range(3):
                    self.car.step()
                self.car_node.getField("translation").setSFVec3f(self.START_POS)
                for _ in range(10):
                    self.car.step()
            except Exception as e:
                print(f"Ошибка телепортации: {e}")

        for _ in range(20):
            self.car.setCruisingSpeed(0)
            self.car.setSteeringAngle(0)
            self.car.step()

        self.step_count = 0
        self.episode_reward = 0
        self.prev_action = np.array([0.0, 0.0], dtype=np.float32)
        self.prev_speed = 0.0
        self.prev_dist_to_target = self._check_goal_3d()[1]
        self.best_distance = float('inf')
        self.steps_without_progress = 0
        self.action_history.clear()
        self.steering_changes = 0
        self.last_steering = 0.0

        observation = self._get_observation()
        return observation, {}

    def step(self, action):
        """
        Выполнение одного шага симуляции.

        Применяет действие агента, обновляет состояние среды,
        вычисляет награду и проверяет завершение эпизода.

        Параметры:
            action: действие агента [steering, speed] от -0.8 до 0.8 и от -5 до 20

        Возвращает:
            tuple: (наблюдение, награда, завершено, обрезано, информация)
        """
        # =====================================================================
        # 1. ПОДГОТОВКА ДЕЙСТВИЯ
        # =====================================================================
        # Преобразуем действие в numpy массив и очищаем от NaN/Inf
        action = np.array(action, dtype=np.float32)
        action = np.nan_to_num(action, nan=0.0, posinf=0.8, neginf=-0.8)

        # Ограничиваем действие в допустимых пределах
        action[0] = np.clip(action[0], -0.8, 0.8)  # steering
        action[1] = np.clip(action[1], -5.0, 20.0)  # speed

        # =====================================================================
        # 2. ИССЛЕДОВАНИЕ (ТОЛЬКО В НАЧАЛЕ)
        # =====================================================================
        # Добавляем шум для исследования только в первых 20 эпизодах
        if self.episode_count < 20:
            action[0] += np.random.normal(0, 0.2)  # Шум на руль
            action[1] += np.random.normal(0, 2)  # Шум на скорость
            # Снова ограничиваем после добавления шума
            action[0] = np.clip(action[0], -0.8, 0.8)
            action[1] = np.clip(action[1], -5.0, 20.0)

        # =====================================================================
        # 3. СГЛАЖИВАНИЕ УПРАВЛЕНИЯ
        # =====================================================================
        # Плавное изменение руля (70% предыдущего + 30% нового)
        smoothed_steering = 0.7 * self.prev_action[0] + 0.3 * action[0]
        steering = float(np.clip(smoothed_steering, -0.8, 0.8))

        # Плавное изменение скорости
        smoothed_speed = 0.7 * self.prev_speed + 0.3 * action[1]
        speed = float(np.clip(smoothed_speed, -5.0, 20.0))

        # Сохраняем применённое действие для следующего шага
        self.prev_action = np.array([steering, speed], dtype=np.float32)

        # =====================================================================
        # 4. ПРИМЕНЕНИЕ ДЕЙСТВИЯ К АВТОМОБИЛЮ
        # =====================================================================
        self.car.setSteeringAngle(steering)
        self.car.setCruisingSpeed(speed)
        self.car.step()  # Один шаг симуляции Webots

        # =====================================================================
        # 5. ПОЛУЧЕНИЕ НОВОГО СОСТОЯНИЯ И НАГРАДЫ
        # =====================================================================
        observation = self._get_observation()
        reward, is_goal, is_collision, collision_info, current_speed = self._calculate_reward(action)

        # =====================================================================
        # 6. ПОДГОТОВКА ИНФОРМАЦИИ О ШАГЕ
        # =====================================================================
        terminated = False
        truncated = False
        info = {
            'reason': 'unknown',  # Причина завершения эпизода
            'final_distance': self.prev_dist_to_target if (is_goal or is_collision) else None,
            'current_speed': current_speed,
            'min_lidar': min(
                self._get_lidar_min_distance(self.lidar_front),
                self._get_lidar_min_distance(self.lidar_left),
                self._get_lidar_min_distance(self.lidar_right)
            ),
            'angle_to_target': abs(self._get_angle_to_target()),
            'dist_to_target': self.prev_dist_to_target,
            'steering': steering,
            'speed': speed
        }

        # =====================================================================
        # 7. ПРОВЕРКА ДОСТИЖЕНИЯ ЦЕЛИ
        # =====================================================================
        if is_goal:
            # Базовая награда за цель + небольшой бонус за быстроту
            speed_bonus = max(0, (self.MAX_STEPS - self.step_count) * 0.1)
            reward += 500 + speed_bonus
            terminated = True
            self.success_count += 1
            info['reason'] = 'goal'
            direction = "вперёд" if current_speed >= 0 else "назад"
            print(f"\n🎯 GOAL! Шагов: {self.step_count}, скорость={current_speed:.1f}, "
                  f"бонус={speed_bonus:.1f}, reward={self.episode_reward + reward:.1f}")

        # =====================================================================
        # 8. ПРОВЕРКА СТОЛКНОВЕНИЯ
        # =====================================================================
        elif is_collision:
            reward -= 200
            terminated = True
            self.collision_count += 1
            info['reason'] = 'collision'
            print(f"\n💥 COLLISION! {collision_info}, reward={self.episode_reward + reward:.1f}")

        # =====================================================================
        # 9. ПРОВЕРКА ТАЙМАУТА
        # =====================================================================
        elif self.step_count >= self.MAX_STEPS:
            reward -= 50
            truncated = True
            info['reason'] = 'timeout'
            print(f"\n⏱️ TIMEOUT! dist={self.prev_dist_to_target:.1f}m, reward={self.episode_reward + reward:.1f}")

        # =====================================================================
        # 10. ОБНОВЛЕНИЕ СЧЁТЧИКОВ
        # =====================================================================
        self.step_count += 1
        self.episode_reward += reward

        # =====================================================================
        # 11. ЛОГИРОВАНИЕ КАЖДЫЕ 100 ШАГОВ
        # =====================================================================
        if self.step_count % 100 == 0:
            _, dist_to_target = self._check_goal_3d()
            _, _, min_lidar = self._check_collision_lidar()
            direction = "ВПЕРЁД" if current_speed >= 0 else "НАЗАД"

            # Прогресс-бар для визуализации
            progress_bar = '█' * int((self.step_count / self.MAX_STEPS) * 20)
            progress_bar = progress_bar.ljust(20, '░')

            print(f"  Step {self.step_count:4d}/{[self.MAX_STEPS]} [{progress_bar}] "
                  f"r={reward:6.1f} total={self.episode_reward:7.1f} "
                  f"goal={dist_to_target:5.1f}m lidar={min_lidar:4.1f}m {direction}")

        return observation, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        """Закрытие среды и освобождение ресурсов"""
        self._remove_all_obstacles()
        print("\n👋 Закрытие среды")
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
    """Колбэк для логирования метрик модели в TensorBoard"""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_success = []
        self.episode_collisions = []
        self.episode_timeouts = []

    def _on_step(self) -> bool:
        if self.locals.get("dones") and any(self.locals["dones"]):
            dones = self.locals["dones"]
            infos = self.locals["infos"]

            for i, done in enumerate(dones):
                if done and infos[i]:
                    info = infos[i]

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

                        if len(self.episode_success) > 0:
                            self.logger.record("stats/success_rate",
                                               np.mean(self.episode_success[-100:]) * 100)
                            self.logger.record("stats/collision_rate",
                                               np.mean(self.episode_collisions[-100:]) * 100)
                            self.logger.record("stats/timeout_rate",
                                               np.mean(self.episode_timeouts[-100:]) * 100)

                    if "final_distance" in info and info["final_distance"] is not None:
                        self.logger.record("environment/final_distance", info["final_distance"])

        return True


# =============================================================================
# ФУНКЦИЯ ОБУЧЕНИЯ МОДЕЛИ
# =============================================================================
def train():
    print("\n" + "=" * 60)
    print("ЗАПУСК ОБУЧЕНИЯ (УЛУЧШЕННЫЕ НАГРАДЫ ПРОТИВ ЗИГЗАГОВ)")
    print("=" * 60)

    env = BMWRLEnvironment()

    policy_kwargs = dict(
        net_arch=[256, 256],
        activation_fn=th.nn.ReLU,
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,  # Увеличено для лучшего исследования в начале
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log="./logs/",
        policy_kwargs=policy_kwargs,
        device="cpu"
    )

    os.makedirs("./models", exist_ok=True)

    print("\nНачинаем обучение на 1,500,000 шагов...")
    print("✅ Рандомизация сценариев ВКЛЮЧЕНА")
    print("✅ Задний ход ВКЛЮЧЕН")
    print(f"✅ Количество препятствий: {env.num_obstacles}")
    print("✅ УЛУЧШЕННАЯ СИСТЕМА НАГРАД:")
    print("   - Убрана награда за простое движение")
    print("   - Увеличен штраф за время: -0.3/шаг")
    print("   - Прогресс к цели: +80/метр")
    print("   - Штраф за зигзаги")
    print("   - Штраф за отсутствие прогресса")
    print("   - Достижение цели: +1000 + бонус")
    print("\n⏱️ Ожидаемое время: ~3-4 часа")
    print("Нажмите Ctrl+C для остановки\n")
    print("💡 Запустите TensorBoard: tensorboard --logdir=./logs\n")

    try:
        model.learn(
            total_timesteps=1_500_000,
            tb_log_name="BMW_X5_RL_IMPROVED",
            reset_num_timesteps=False,
            callback=ModelMetricsCallback()
        )

        final_path = "./models/bmw_rl_improved(v2)"
        model.save(final_path)
        print(f"\n✅ Финальная модель сохранена: {final_path}")

    except KeyboardInterrupt:
        print("\n\n🛑 Обучение прервано пользователем")
        interrupt_path = "./models/bmw_rl_improved_interrupted"
        model.save(interrupt_path)
        print(f"✅ Модель сохранена: {interrupt_path}")

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
def test(model_path="./models/bmw_rl_improved(v2).zip"):
    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ МОДЕЛИ")
    print("=" * 60)

    env = BMWRLEnvironment()

    if os.path.exists(model_path):
        model = PPO.load(model_path)
        print(f"✅ Модель загружена: {model_path}")
    else:
        print(f"❌ Модель не найдена: {model_path}")
        print("   Ищу другие модели...")

        alt_paths = [
            "./models/bmw_rl_reverse_final.zip",
            "./models/bmw_rl_randomized_final.zip",
            "./models/bmw_rl_balanced.zip"
        ]

        for path in alt_paths:
            if os.path.exists(path):
                model = PPO.load(path)
                print(f"✅ Загружена альтернативная модель: {path}")
                break
        else:
            print("❌ Модель не найдена!")
            return

    input("\nНажмите Enter для начала тестирования...")

    obs, _ = env.reset()
    total_reward = 0
    step = 0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        if step % 50 == 0:
            _, dist_target = env._check_goal_3d()
            _, _, min_lidar = env._check_collision_lidar()
            print(f"Step {step:4d}: reward={reward:6.1f} total={total_reward:7.1f} "
                  f"goal={dist_target:5.1f}m lidar={min_lidar:4.1f}m")

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
            model_path = sys.argv[2] if len(sys.argv) > 2 else "./models/bmw_rl_improved(v2).zip"
            test(model_path)
        else:
            print("Использование:")
            print("  python rl_controller.py           # обучение")
            print("  python rl_controller.py --test    # тестирование")
    else:
        train()#!/usr/bin/env python3
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
from vehicle import Car
import numpy as np
import math
import os
import sys
import random
from collections import deque
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import torch as th


# =============================================================================
# КЛАСС СРЕДЫ ДЛЯ ОБУЧЕНИЯ
# =============================================================================
class BMWRLEnvironment(gym.Env):
    """
    Среда для обучения с подкреплением автомобиля BMW X5 в Webots.

    Особенности:
    - Рандомизация положения препятствий и цели в каждом эпизоде
    - Поддержка заднего хода (отрицательные значения скорости)
    - Детектирование столкновений по лидарам
    - Улучшенная система наград против зигзагов
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode="human"):
        super().__init__()

        print("=" * 60)
        print("BMW X5 — СРЕДА ДЛЯ ОБУЧЕНИЯ С ПОДКРЕПЛЕНИЕМ")
        print("=" * 60)

        # =====================================================================
        # ИНИЦИАЛИЗАЦИЯ WEBOTS
        # =====================================================================
        self.car = Car()
        self.timestep = int(self.car.getBasicTimeStep())

        # =====================================================================
        # ПОИСК НОД В МИРЕ WEBOTS
        # =====================================================================
        print("\nПоиск нод в мире:")

        self.car_node = self.car.getFromDef("CAR")
        if self.car_node:
            print(f"  CAR: найдена")
        else:
            print(f"  CAR: не найдена, использую getSelf()")
            self.car_node = self.car.getSelf()

        self.obstacle_node = self.car.getFromDef("OBSTACLE")
        if self.obstacle_node:
            print(f"  OBSTACLE: найдена")
        else:
            print(f"  OBSTACLE: не найдена")

        self.target_node = self.car.getFromDef("TARGET")
        self.use_virtual_target = False

        if self.target_node:
            target_pos = self.target_node.getPosition()
            print(f"TARGET: найдена на ({target_pos[0]:.1f}, {target_pos[1]:.1f}, {target_pos[2]:.1f})")
            try:
                self.target_node.removeBoundingObject()
                self.target_node.enableContactPointsTracking(False)
                self.target_node.resetPhysics()
                print("Физика цели отключена")
            except Exception as e:
                print(f" Не удалось отключить физику: {e}")
        else:
            print(f"TARGET: не найдена, использую виртуальную цель")
            self.use_virtual_target = True
            self.virtual_target_pos = [10, 0.5, 0.4]

        print()

        # =====================================================================
        # ПАРАМЕТРЫ СРЕДЫ
        # =====================================================================
        self.START_POS = [-11.5, 0.5, 0.5]
        self.START_ROT = [0, 1, 0, 0]
        self.GOAL_DISTANCE = 4.0
        self.COLLISION_DISTANCE = 0.5
        self.MAX_STEPS = 2500

        # =====================================================================
        # ПАРАМЕТРЫ РАНДОМИЗАЦИИ СЦЕНАРИЕВ
        # =====================================================================
        self.randomize_scenarios = True
        self.OBSTACLE_X_RANGE = (-8, 8)
        self.OBSTACLE_Y_RANGE = (-8, 8)
        self.OBSTACLE_Z = 0.5
        self.TARGET_X_RANGE = (-12, 12)
        self.TARGET_Y_RANGE = (-10, 10)
        self.TARGET_Z = 0.5
        self.MIN_DIST_FROM_START = 6.0
        self.MIN_DIST_BETWEEN_OBJ = 3.0

        # =====================================================================
        # ПАРАМЕТРЫ ДЛЯ НЕСКОЛЬКИХ ПРЕПЯТСТВИЙ
        # =====================================================================
        self.multiple_obstacles = True
        self.num_obstacles = 4
        self.obstacle_nodes = []
        self.obstacle_sizes = []
        self.OBSTACLE_SIZE_RANGE = (0.8, 2.0)

        # =====================================================================
        # СТАТИСТИКА СЦЕНАРИЕВ
        # =====================================================================
        self.scenario_stats = {'easy': 0, 'medium': 0, 'hard': 0}

        # =====================================================================
        # ИНИЦИАЛИЗАЦИЯ СЕНСОРОВ
        # =====================================================================
        self._setup_sensors()

        # =====================================================================
        # ОПРЕДЕЛЕНИЕ ПРОСТРАНСТВА НАБЛЮДЕНИЙ
        # =====================================================================
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
        self.action_space = spaces.Box(
            low=np.array([-0.8, -5.0], dtype=np.float32),
            high=np.array([0.8, 20.0], dtype=np.float32),
            dtype=np.float32
        )

        # =====================================================================
        # СЧЁТЧИКИ ДЛЯ СТАТИСТИКИ
        # =====================================================================
        self.step_count = 0
        self.episode_reward = 0
        self.episode_count = 0
        self.success_count = 0
        self.collision_count = 0
        self.prev_action = np.array([0.0, 0.0], dtype=np.float32)
        self.prev_speed = 0.0
        self.best_distance = float('inf')

        # НОВОЕ: для отслеживания зигзагов и прогресса
        self.steps_without_progress = 0
        self.action_history = deque(maxlen=50)
        self.last_progress_distance = float('inf')
        self.steering_changes = 0
        self.last_steering = 0.0

        self.render_mode = render_mode

        print(f"\n✅ Goal distance: {self.GOAL_DISTANCE}m")
        print(f"✅ Collision distance: {self.COLLISION_DISTANCE}m")
        print(f"✅ Max steps: {self.MAX_STEPS}")
        print(f"✅ Randomization: {'ON' if self.randomize_scenarios else 'OFF'}")
        print(f"✅ Reverse gear: ENABLED (speed: -5 to 20)")
        print(f"✅ Multiple obstacles: {self.num_obstacles}")
        print("=" * 60)

    # =========================================================================
    # МЕТОДЫ ДЛЯ РАБОТЫ С СЕНСОРАМИ
    # =========================================================================
    def _setup_sensors(self):
        """Инициализация и включение всех сенсоров"""
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
        """Очистка значения от NaN и бесконечности"""
        if value is None or not isinstance(value, (int, float)):
            return default
        if math.isnan(value) or math.isinf(value):
            return default
        return float(np.clip(value, min_val, max_val))

    def _get_lidar_min_distance(self, lidar):
        """Получение минимального расстояния с лидара"""
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
        """Получение текущей позиции автомобиля из GPS"""
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
        """Получение текущего угла поворота автомобиля"""
        if self.imu:
            try:
                rpy = self.imu.getRollPitchYaw()
                if rpy and len(rpy) >= 3:
                    return self._sanitize_value(rpy[2], 0, -np.pi, np.pi)
            except:
                pass
        return 0.0

    def _get_target_position(self):
        """Получение позиции цели"""
        if self.use_virtual_target:
            return self.virtual_target_pos
        elif self.target_node:
            return self.target_node.getPosition()
        return [10, 0.5, 0.4]

    def _get_angle_to_target(self):
        """Вычисление угла до цели"""
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
        """Проверка достижения цели"""
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
        """Проверка столкновения с препятствиями"""
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
        """Формирование вектора наблюдения"""
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

        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            obs = np.nan_to_num(obs, nan=0.0, posinf=25.0, neginf=-25.0)

        return obs

    # =========================================================================
    # МЕТОДЫ ДЛЯ РАБОТЫ С НЕСКОЛЬКИМИ ПРЕПЯТСТВИЯМИ
    # =========================================================================
    def _create_obstacle(self, position, size=1.0, index=0):
        """Создание нового препятствия в мире Webots"""
        try:
            root = self.car.getRoot()
            children = root.getField("children")
            color = [random.uniform(0.5, 1.0),
                     random.uniform(0, 0.5),
                     random.uniform(0, 0.5)]

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

            children.importMFNodeFromString(-1, obstacle_str)
            obs_node = self.car.getFromDef(f"OBSTACLE_{index}")
            return obs_node
        except Exception as e:
            print(f"Ошибка создания препятствия: {e}")
            return None

    def _remove_all_obstacles(self):
        """Удаление всех созданных препятствий"""
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
        """Генерация случайной позиции с проверкой ограничений"""
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

            dist_from_start = math.sqrt(
                (x - self.START_POS[0]) ** 2 +
                (y - self.START_POS[1]) ** 2
            )

            if dist_from_start < self.MIN_DIST_FROM_START:
                attempt += 1
                continue

            valid = True
            if exclude_positions:
                for pos in exclude_positions:
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

        print(f"⚠️ Не удалось найти позицию, использую значение по умолчанию")
        if for_target:
            return [10, 0.5, 0.4]
        else:
            return [0, 0, 1]

    def _randomize_scenario(self):
        """Создание нового случайного сценария"""
        if not self.randomize_scenarios:
            return

        print("\n  🎲 Генерация нового сценария с несколькими препятствиями...")
        self._remove_all_obstacles()

        all_obstacle_positions = []
        for i in range(self.num_obstacles):
            size = random.uniform(*self.OBSTACLE_SIZE_RANGE)
            obs_pos = self._generate_random_position(
                for_target=False,
                exclude_positions=all_obstacle_positions + [self.START_POS]
            )
            obs_node = self._create_obstacle(obs_pos, size, i)
            if obs_node:
                self.obstacle_nodes.append(obs_node)
                self.obstacle_sizes.append(size)
                all_obstacle_positions.append(obs_pos)
                print(f"     Препятствие {i + 1}: X={obs_pos[0]:5.1f}, Y={obs_pos[1]:5.1f}, размер={size:.1f}м")

        target_pos = self._generate_random_position(
            for_target=True,
            exclude_positions=all_obstacle_positions
        )

        if self.target_node:
            try:
                self.target_node.getField("translation").setSFVec3f(target_pos)
                print(f"     Цель: X={target_pos[0]:5.1f}, Y={target_pos[1]:5.1f}")
            except Exception as e:
                print(f"     Ошибка перемещения цели: {e}")
        elif self.use_virtual_target:
            self.virtual_target_pos = target_pos

        if all_obstacle_positions:
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

    # =========================================================================
    # УЛУЧШЕННЫЙ МЕТОД РАСЧЁТА НАГРАДЫ (ПРОТИВ ЗИГЗАГОВ)
    # =========================================================================
    def _calculate_reward(self, action):
        """
        Улучшенная система наград против зигзагов и петель.

        Ключевые изменения:
        1. Убрана награда за простое движение (было +1.0)
        2. Увеличен штраф за время для ускорения
        3. Добавлен штраф за зигзаги
        4. Увеличен бонус за достижение цели
        5. Добавлен прогрессивный штраф за отсутствие прогресса
        """
        is_goal, dist_to_target = self._check_goal_3d()
        is_collision, collision_info, min_lidar = self._check_collision_lidar()

        # Текущая скорость
        current_speed = 0.0
        if self.car_node:
            try:
                vel = self.car_node.getVelocity()
                if vel:
                    current_speed = vel[2]
            except:
                pass

        reward = 0.0

        # =====================================================================
        # 1. БАЗОВЫЙ ШТРАФ ЗА ВРЕМЯ (УВЕЛИЧЕН)
        # =====================================================================
        # Каждый шаг стоит дороже, чтобы агент не затягивал эпизод
        reward -= 0.3  # Было -0.1

        # =====================================================================
        # 2. УБРАНА НАГРАДА ЗА ПРОСТОЕ ДВИЖЕНИЕ (БЫЛО +1.0)
        # =====================================================================
        # Это главное изменение! Агент больше не получает награду за бесцельное движение

        # =====================================================================
        # 3. УСИЛЕННАЯ НАГРАДА ЗА ПРОГРЕСС К ЦЕЛИ
        # =====================================================================
        if hasattr(self, 'prev_dist_to_target'):
            progress = self.prev_dist_to_target - dist_to_target

            if progress > 0.01:  # Приближаемся
                reward += progress * 80.0  # Увеличено с 50
                self.steps_without_progress = 0

                # Дополнительный бонус за прямолинейное движение
                angle_to_target = abs(self._get_angle_to_target())
                if angle_to_target < 0.3:  # Почти прямо на цель
                    reward += progress * 40.0

            elif progress < -0.01:  # Удаляемся
                reward += progress * 40.0  # Штраф за удаление (было 20)
                self.steps_without_progress += 1
            else:
                self.steps_without_progress += 1

        # =====================================================================
        # 4. ШТРАФ ЗА ДЛИТЕЛЬНОЕ ОТСУТСТВИЕ ПРОГРЕССА (НОВОЕ)
        # =====================================================================
        if self.steps_without_progress > 50:
            reward -= 5.0  # Сильный штраф за застревание
            if self.steps_without_progress % 20 == 0:
                print(f"     ⚠️ Штраф за отсутствие прогресса: {self.steps_without_progress} шагов")
        elif self.steps_without_progress > 30:
            reward -= 2.0
        elif self.steps_without_progress > 15:
            reward -= 1.0

        # =====================================================================
        # 5. ШТРАФ ЗА ЗИГЗАГИ (НОВОЕ)
        # =====================================================================
        # Отслеживаем резкие изменения руля
        current_steering = action[0]
        if hasattr(self, 'last_steering'):
            steering_change = abs(current_steering - self.last_steering)
            if steering_change > 0.3:  # Резкое изменение руля
                reward -= 1.0
                self.steering_changes += 1
            elif steering_change > 0.1:
                reward -= 0.3

        self.last_steering = current_steering

        # Если слишком много изменений руля за последние 50 шагов
        if self.steering_changes > 15:
            reward -= 2.0

        # =====================================================================
        # 6. УЛУЧШЕННАЯ НАГРАДА ЗА НАПРАВЛЕНИЕ
        # =====================================================================
        angle_to_target = abs(self._get_angle_to_target())
        # Бонус за правильное направление только если есть движение
        if abs(current_speed) > 1.0:
            angle_score = 1.0 - min(angle_to_target / math.pi, 1.0)
            reward += angle_score * 2.0  # Увеличено с 1.0

        # =====================================================================
        # 7. ШТРАФ ЗА БЛИЗОСТЬ К ПРЕПЯТСТВИЮ
        # =====================================================================
        if min_lidar < 2.0:
            danger = (2.0 - min_lidar) / 2.0
            reward -= danger * 12.0  # Увеличено с 8.0

        # =====================================================================
        # 8. УЛУЧШЕННАЯ НАГРАДА ЗА НОВЫЙ РЕКОРД
        # =====================================================================
        if dist_to_target < self.best_distance:
            improvement = self.best_distance - dist_to_target
            self.best_distance = dist_to_target
            reward += improvement * 50.0  # Увеличено с 30.0
            print(f"     🏆 Новый рекорд! Дистанция: {dist_to_target:.2f}м")

        self.prev_dist_to_target = dist_to_target
        self.prev_speed = current_speed

        if math.isnan(reward) or math.isinf(reward):
            reward = 0.0

        return reward, is_goal, is_collision, collision_info, current_speed

    # =========================================================================
    # ОСНОВНЫЕ МЕТОДЫ GYMNASIUM
    # =========================================================================
    def reset(self, seed=None, options=None):
        """Сброс среды к начальному состоянию"""
        super().reset(seed=seed)

        if self.randomize_scenarios:
            self._randomize_scenario()

        self.episode_count += 1

        if self.episode_count > 1:
            success_rate = (self.success_count / (self.episode_count - 1)) * 100 if self.episode_count > 1 else 0
            avg_reward = self.episode_reward / max(self.step_count, 1)
            print(f"\n📊 Эпизод {self.episode_count - 1}: награда={self.episode_reward:.1f}, "
                  f"средняя={avg_reward:.2f}/шаг, успех={success_rate:.1f}%")

        print(f"\n🔄 Сброс эпизода {self.episode_count}...")

        self.car.setCruisingSpeed(0)
        self.car.setSteeringAngle(0)

        self.car.simulationResetPhysics()
        for _ in range(5):
            self.car.step()

        if self.car_node:
            try:
                self.car_node.getField("rotation").setSFRotation(self.START_ROT)
                for _ in range(3):
                    self.car.step()
                self.car_node.getField("translation").setSFVec3f(self.START_POS)
                for _ in range(10):
                    self.car.step()
            except Exception as e:
                print(f"Ошибка телепортации: {e}")

        for _ in range(20):
            self.car.setCruisingSpeed(0)
            self.car.setSteeringAngle(0)
            self.car.step()

        self.step_count = 0
        self.episode_reward = 0
        self.prev_action = np.array([0.0, 0.0], dtype=np.float32)
        self.prev_speed = 0.0
        self.prev_dist_to_target = self._check_goal_3d()[1]
        self.best_distance = float('inf')
        self.steps_without_progress = 0
        self.action_history.clear()
        self.steering_changes = 0
        self.last_steering = 0.0

        observation = self._get_observation()
        return observation, {}

    def step(self, action):
        """Выполнение одного шага симуляции"""
        action = np.array(action, dtype=np.float32)
        action = np.nan_to_num(action, nan=0.0, posinf=0.8, neginf=-0.8)

        if self.episode_count < 20:
            action[0] += np.random.normal(0, 0.2)
            action[1] += np.random.normal(0, 2)

        smoothed_steering = 0.7 * self.prev_action[0] + 0.3 * action[0]
        steering = float(np.clip(smoothed_steering, -0.8, 0.8))
        smoothed_speed = 0.7 * self.prev_speed + 0.3 * action[1]
        speed = float(np.clip(smoothed_speed, -5.0, 20.0))

        self.prev_action = np.array([steering, speed], dtype=np.float32)

        self.car.setSteeringAngle(steering)
        self.car.setCruisingSpeed(speed)
        self.car.step()

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

        # =====================================================================
        # УЛУЧШЕННЫЕ НАГРАДЫ ЗА ЗАВЕРШЕНИЕ
        # =====================================================================
        if is_goal:
            # Бонус за быстроту: чем меньше шагов, тем больше бонус
            speed_bonus = max(0, (self.MAX_STEPS - self.step_count) * 0.3)
            reward += 1000 + speed_bonus  # Увеличено с 500
            terminated = True
            self.success_count += 1
            info['reason'] = 'goal'
            direction = "вперёд" if current_speed >= 0 else "назад"
            print(f"\n🎯 GOAL! Шагов: {self.step_count}, скорость={current_speed:.1f}, "
                  f"бонус={speed_bonus:.1f}, reward={self.episode_reward + reward:.1f}")

        elif is_collision:
            reward -= 400  # Увеличено с 200
            terminated = True
            self.collision_count += 1
            info['reason'] = 'collision'
            print(f"\n💥 COLLISION! {collision_info}, reward={self.episode_reward + reward:.1f}")

        elif self.step_count >= self.MAX_STEPS:
            reward -= 150  # Увеличено с 50
            truncated = True
            info['reason'] = 'timeout'
            print(f"\n⏱️ TIMEOUT! dist={self.prev_dist_to_target:.1f}m, reward={self.episode_reward + reward:.1f}")

        self.step_count += 1
        self.episode_reward += reward

        if self.step_count % 100 == 0:
            _, dist_to_target = self._check_goal_3d()
            _, _, min_lidar = self._check_collision_lidar()
            direction = "ВПЕРЁД" if current_speed >= 0 else "НАЗАД"
            print(f"  Step {self.step_count:4d}: r={reward:6.1f} total={self.episode_reward:7.1f} "
                  f"goal={dist_to_target:5.1f}m lidar={min_lidar:4.1f}m {direction}")

        return observation, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        """Закрытие среды и освобождение ресурсов"""
        self._remove_all_obstacles()
        print("\n👋 Закрытие среды")
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
    """Колбэк для логирования метрик модели в TensorBoard"""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_success = []
        self.episode_collisions = []
        self.episode_timeouts = []

    def _on_step(self) -> bool:
        if self.locals.get("dones") and any(self.locals["dones"]):
            dones = self.locals["dones"]
            infos = self.locals["infos"]

            for i, done in enumerate(dones):
                if done and infos[i]:
                    info = infos[i]

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

                        if len(self.episode_success) > 0:
                            self.logger.record("stats/success_rate",
                                               np.mean(self.episode_success[-100:]) * 100)
                            self.logger.record("stats/collision_rate",
                                               np.mean(self.episode_collisions[-100:]) * 100)
                            self.logger.record("stats/timeout_rate",
                                               np.mean(self.episode_timeouts[-100:]) * 100)

                    if "final_distance" in info and info["final_distance"] is not None:
                        self.logger.record("environment/final_distance", info["final_distance"])

        return True


# =============================================================================
# ФУНКЦИЯ ОБУЧЕНИЯ МОДЕЛИ
# =============================================================================
def train():
    print("\n" + "=" * 60)
    print("ЗАПУСК ОБУЧЕНИЯ (УЛУЧШЕННЫЕ НАГРАДЫ ПРОТИВ ЗИГЗАГОВ)")
    print("=" * 60)

    env = BMWRLEnvironment()

    policy_kwargs = dict(
        net_arch=[256, 256],
        activation_fn=th.nn.ReLU,
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,  # Увеличено для лучшего исследования в начале
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log="./logs/",
        policy_kwargs=policy_kwargs,
        device="cpu"
    )

    os.makedirs("./models", exist_ok=True)

    print("\nНачинаем обучение на 1,500,000 шагов...")
    print("✅ Рандомизация сценариев ВКЛЮЧЕНА")
    print("✅ Задний ход ВКЛЮЧЕН")
    print(f"✅ Количество препятствий: {env.num_obstacles}")
    print("✅ УЛУЧШЕННАЯ СИСТЕМА НАГРАД:")
    print("   - Убрана награда за простое движение")
    print("   - Увеличен штраф за время: -0.3/шаг")
    print("   - Прогресс к цели: +80/метр")
    print("   - Штраф за зигзаги")
    print("   - Штраф за отсутствие прогресса")
    print("   - Достижение цели: +1000 + бонус")
    print("\n⏱️ Ожидаемое время: ~3-4 часа")
    print("Нажмите Ctrl+C для остановки\n")
    print("💡 Запустите TensorBoard: tensorboard --logdir=./logs\n")

    try:
        model.learn(
            total_timesteps=1_500_000,
            tb_log_name="BMW_X5_RL_IMPROVED",
            reset_num_timesteps=False,
            callback=ModelMetricsCallback()
        )

        final_path = "./models/bmw_rl_improved(v2)"
        model.save(final_path)
        print(f"\n✅ Финальная модель сохранена: {final_path}")

    except KeyboardInterrupt:
        print("\n\n🛑 Обучение прервано пользователем")
        interrupt_path = "./models/bmw_rl_improved_interrupted"
        model.save(interrupt_path)
        print(f"✅ Модель сохранена: {interrupt_path}")

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
def test(model_path="./models/bmw_rl_improved(v2).zip"):
    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ МОДЕЛИ")
    print("=" * 60)

    env = BMWRLEnvironment()

    if os.path.exists(model_path):
        model = PPO.load(model_path)
        print(f"✅ Модель загружена: {model_path}")
    else:
        print(f"❌ Модель не найдена: {model_path}")
        print("   Ищу другие модели...")

        alt_paths = [
            "./models/bmw_rl_reverse_final.zip",
            "./models/bmw_rl_randomized_final.zip",
            "./models/bmw_rl_balanced.zip"
        ]

        for path in alt_paths:
            if os.path.exists(path):
                model = PPO.load(path)
                print(f"✅ Загружена альтернативная модель: {path}")
                break
        else:
            print("❌ Модель не найдена!")
            return

    input("\nНажмите Enter для начала тестирования...")

    obs, _ = env.reset()
    total_reward = 0
    step = 0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        if step % 50 == 0:
            _, dist_target = env._check_goal_3d()
            _, _, min_lidar = env._check_collision_lidar()
            print(f"Step {step:4d}: reward={reward:6.1f} total={total_reward:7.1f} "
                  f"goal={dist_target:5.1f}m lidar={min_lidar:4.1f}m")

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
            model_path = sys.argv[2] if len(sys.argv) > 2 else "./models/bmw_rl_improved(v2).zip"
            test(model_path)
        else:
            print("Использование:")
            print("  python rl_controller.py           # обучение")
            print("  python rl_controller.py --test    # тестирование")
    else:
        train()