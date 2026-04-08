#!/usr/bin/env python3
"""
RL КОНТРОЛЛЕР ДЛЯ BMW X5 В WEBOTS
===================================
Обучение на карте с множеством препятствий.
Цель генерируется с изменением X и Y (высота), Z фиксирован.
СБАЛАНСИРОВАННАЯ система наград (столкновение сильно наказывается).

Автор: Студенческий проект
Дата: 2026
"""

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


class BMWRLEnvironment(gym.Env):
    """
    Среда для обучения на карте с множеством препятствий.
    Цель генерируется с проверкой, что не пересекается с препятствиями.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode="human"):
        super().__init__()

        print("=" * 70)
        print("BMW X5 — ОБУЧЕНИЕ НА КАРТЕ С МНОЖЕСТВОМ ПРЕПЯТСТВИЙ")
        print("=" * 70)

        # =====================================================================
        # ИНИЦИАЛИЗАЦИЯ WEBOTS
        # =====================================================================
        self.car = Car()
        self.timestep = int(self.car.getBasicTimeStep())

        # =====================================================================
        # ПОИСК НОД В МИРЕ WEBOTS
        # =====================================================================
        print("\nПоиск нод в мире:")

        # Поиск автомобиля
        self.car_node = self.car.getFromDef("CAR")
        if self.car_node:
            print("  ✅ CAR: найдена")
            start_pos = self.car_node.getPosition()
            print(f"     Начальная позиция: X={start_pos[0]:.2f}, Y={start_pos[1]:.2f}, Z={start_pos[2]:.2f}")
        else:
            print("  ⚠️ CAR: не найдена, использую getSelf()")
            self.car_node = self.car.getSelf()

        # Поиск цели
        self.target_node = self.car.getFromDef("TARGET")
        self.use_virtual_target = False

        if self.target_node:
            target_pos = self.target_node.getPosition()
            print(f"  ✅ TARGET: найдена на ({target_pos[0]:.1f}, {target_pos[1]:.1f}, {target_pos[2]:.1f})")
            try:
                self.target_node.removeBoundingObject()
                self.target_node.enableContactPointsTracking(False)
                self.target_node.resetPhysics()
                print("     👻 Физика цели отключена")
            except Exception as e:
                print(f"     ⚠️ Не удалось отключить физику: {e}")
        else:
            print("  ⚠️ TARGET: не найдена, использую виртуальную цель")
            self.use_virtual_target = True
            self.virtual_target_pos = [0, 0.5, 0]

        # =====================================================================
        # СБОР ВСЕХ ПРЕПЯТСТВИЙ В МИРЕ
        # =====================================================================
        print("\nПоиск препятствий в мире...")

        self.obstacle_nodes = []
        self.obstacle_positions = []  # [x, z, size] для проверки
        self.obstacle_sizes = []  # размеры препятствий

        # Функция для получения размера препятствия
        def get_obstacle_size(obs_node):
            try:
                # Пытаемся получить размер из boundingObject
                bounding = obs_node.getField("boundingObject")
                if bounding:
                    box = bounding.getMFNode(0)
                    if box:
                        size_field = box.getField("size")
                        if size_field:
                            size = size_field.getSFVec3f()
                            return max(size[0], size[2])  # берем максимальный размер по X или Z
            except:
                pass
            return 1.0  # размер по умолчанию

        # Поиск всех препятствий с DEF "OBSTACLE"
        obstacle_index = 0
        while True:
            obs_node = self.car.getFromDef(f"OBSTACLE")
            if obs_node is None:
                obs_node = self.car.getFromDef(f"OBSTACLE({obstacle_index})")
            if obs_node is None:
                obs_node = self.car.getFromDef(f"obstacle({obstacle_index})")
            if obs_node is None:
                break

            try:
                pos = obs_node.getPosition()
                if pos and len(pos) >= 3:
                    size = get_obstacle_size(obs_node)
                    self.obstacle_nodes.append(obs_node)
                    self.obstacle_positions.append([pos[0], pos[2], size])
                    self.obstacle_sizes.append(size)
                    print(
                        f"     Препятствие {obstacle_index + 1}: X={pos[0]:6.1f}, Z={pos[2]:6.1f}, размер={size:.1f}м")
            except:
                pass

            obstacle_index += 1
            if obstacle_index > 50:
                break

        # Поиск препятствий по именам
        obstacle_names = ["OBSTACLE", "obstacle", "OBSTACLE(1)", "obstacle(1)",
                          "OBSTACLE(2)", "obstacle(2)", "OBSTACLE(3)", "obstacle(3)",
                          "OBSTACLE(4)", "obstacle(4)", "OBSTACLE(5)", "obstacle(5)",
                          "OBSTACLE(6)", "obstacle(6)", "OBSTACLE(7)", "obstacle(7)",
                          "OBSTACLE(8)", "obstacle(8)", "OBSTACLE(9)", "obstacle(9)",
                          "OBSTACLE(10)", "obstacle(10)", "OBSTACLE(11)", "obstacle(11)",
                          "OBSTACLE(12)", "obstacle(12)", "OBSTACLE(13)", "obstacle(13)",
                          "OBSTACLE(14)", "obstacle(14)", "OBSTACLE(15)", "obstacle(15)",
                          "OBSTACLE(16)", "obstacle(16)", "OBSTACLE(17)", "obstacle(17)",
                          "OBSTACLE(18)", "obstacle(18)"]

        for name in obstacle_names:
            obs_node = self.car.getFromDef(name)
            if obs_node and obs_node not in self.obstacle_nodes:
                try:
                    pos = obs_node.getPosition()
                    if pos and len(pos) >= 3:
                        size = get_obstacle_size(obs_node)
                        self.obstacle_nodes.append(obs_node)
                        self.obstacle_positions.append([pos[0], pos[2], size])
                        self.obstacle_sizes.append(size)
                        print(f"     Препятствие {name}: X={pos[0]:6.1f}, Z={pos[2]:6.1f}, размер={size:.1f}м")
                except:
                    pass

        print(f"\n  ✅ Найдено препятствий: {len(self.obstacle_nodes)}")

        # =====================================================================
        # ПАРАМЕТРЫ СРЕДЫ
        # =====================================================================
        # Начальная позиция автомобиля (из скриншота)
        self.START_POS = [-21.0651, 0.5, 0.313462]
        self.START_ROT = [0, 1, 0, 0]

        self.GOAL_DISTANCE = 4.0
        self.COLLISION_DISTANCE = 0.8
        self.MAX_STEPS = 5000

        # =====================================================================
        # ПАРАМЕТРЫ РАНДОМИЗАЦИИ ЦЕЛИ
        # =====================================================================
        self.randomize_target = True

        # Диапазоны для рандомизации цели
        self.TARGET_X_RANGE = (-23.0, 23.0)
        self.TARGET_Y_RANGE = (-23.0, 23.0)
        self.TARGET_Z = 0.5

        # Минимальные расстояния
        self.MIN_DIST_FROM_START = 8.0
        self.MIN_DIST_FROM_OBSTACLE = 1.5  # Минимальное расстояние от края препятствия

        # =====================================================================
        # СТАТИСТИКА
        # =====================================================================
        self.scenario_stats = {'easy': 0, 'medium': 0, 'hard': 0}

        # =====================================================================
        # ИНИЦИАЛИЗАЦИЯ СЕНСОРОВ
        # =====================================================================
        self._setup_sensors()

        # =====================================================================
        # ПРОСТРАНСТВА
        # =====================================================================
        self.observation_space = spaces.Box(
            low=np.array([-50, -10, -50, -np.pi, -np.pi,
                          0, 0, 0, 0, 0, 0, -0.8], dtype=np.float32),
            high=np.array([50, 10, 50, np.pi, np.pi,
                           25, 25, 25, 25, 30, 1000, 0.8], dtype=np.float32),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=np.array([-0.8, -5.0], dtype=np.float32),
            high=np.array([0.8, 20.0], dtype=np.float32),
            dtype=np.float32
        )

        # =====================================================================
        # СЧЁТЧИКИ
        # =====================================================================
        self.step_count = 0
        self.episode_reward = 0
        self.episode_count = 0
        self.success_count = 0
        self.collision_count = 0
        self.prev_action = np.array([0.0, 0.0], dtype=np.float32)
        self.prev_speed = 0.0
        self.best_distance = float('inf')
        self.steps_without_progress = 0

        self.render_mode = render_mode

        print(f"\n📊 Параметры среды:")
        print(f"  Start position: X={self.START_POS[0]:.1f}, Z={self.START_POS[2]:.1f}")
        print(f"  Goal distance: {self.GOAL_DISTANCE} м")
        print(f"  Collision distance: {self.COLLISION_DISTANCE} м")
        print(f"  Max steps: {self.MAX_STEPS}")
        print(f"  Target X range: {self.TARGET_X_RANGE}")
        print(f"  Target Y range: {self.TARGET_Y_RANGE}")
        print(f"  Target Z (fixed): {self.TARGET_Z} м")
        print(f"  Min dist from obstacle: {self.MIN_DIST_FROM_OBSTACLE} м")
        print(f"  Obstacles found: {len(self.obstacle_nodes)}")
        print("=" * 70)

    # =========================================================================
    # МЕТОДЫ ДЛЯ РАБОТЫ С СЕНСОРАМИ
    # =========================================================================
    def _setup_sensors(self):
        self.lidar_front = self.car.getDevice("lidar_front")
        self.lidar_left = self.car.getDevice("lidar_left")
        self.lidar_right = self.car.getDevice("lidar_right")
        self.gps = self.car.getDevice("gps")
        self.imu = self.car.getDevice("imu")

        for device in [self.lidar_front, self.lidar_left, self.lidar_right, self.gps, self.imu]:
            if device:
                device.enable(self.timestep)
                print(f"  ✅ {device.getName()}")

    def _sanitize_value(self, value, default=0.0, min_val=-1e6, max_val=1e6):
        if value is None or not isinstance(value, (int, float)):
            return default
        if math.isnan(value) or math.isinf(value):
            return default
        return float(np.clip(value, min_val, max_val))

    def _get_lidar_min_distance(self, lidar):
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
        if self.imu:
            try:
                rpy = self.imu.getRollPitchYaw()
                if rpy and len(rpy) >= 3:
                    return self._sanitize_value(rpy[2], 0, -np.pi, np.pi)
            except:
                pass
        return 0.0

    def _get_target_position(self):
        if self.use_virtual_target:
            return self.virtual_target_pos
        elif self.target_node:
            return self.target_node.getPosition()
        return [0, 0.5, 0]

    def _get_angle_to_target(self):
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

    def _check_collision_position(self):
        """Проверка столкновения по позиции (для случая, когда машина упёрлась в препятствие)"""
        car_pos = self._get_position()

        for i, obs in enumerate(self.obstacle_nodes):
            try:
                obs_pos = obs.getPosition()
                obs_size = self.obstacle_sizes[i] if i < len(self.obstacle_sizes) else 1.0

                dx = abs(car_pos[0] - obs_pos[0])
                dz = abs(car_pos[2] - obs_pos[2])

                car_half_width = 1.2
                car_half_length = 2.5
                obs_half = obs_size / 2

                if dx < (car_half_width + obs_half + 0.2) and dz < (car_half_length + obs_half + 0.2):
                    return True, f"POSITION with obstacle at ({obs_pos[0]:.1f}, {obs_pos[2]:.1f})"
            except:
                pass

        return False, None

    def _get_observation(self):
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
    # МЕТОДЫ ДЛЯ БЕЗОПАСНОЙ ГЕНЕРАЦИИ ЦЕЛИ
    # =========================================================================
    def _is_position_safe(self, x, z):
        """
        Проверка безопасной позиции для цели.
        Проверяет, что цель не пересекается с препятствиями.
        """
        # Расстояние от старта
        dist_from_start = math.sqrt(
            (x - self.START_POS[0]) ** 2 +
            (z - self.START_POS[2]) ** 2
        )
        if dist_from_start < self.MIN_DIST_FROM_START:
            return False

        # Проверка пересечения с каждым препятствием
        target_radius = 0.5  # радиус цели (половина размера куба 1x1)

        for i, pos in enumerate(self.obstacle_positions):
            obs_x, obs_z, obs_size = pos[0], pos[1], pos[2]
            obs_radius = obs_size / 2

            # Расстояние между центрами
            dist = math.sqrt((x - obs_x) ** 2 + (z - obs_z) ** 2)

            # Минимальное безопасное расстояние между центрами
            min_safe_dist = target_radius + obs_radius + self.MIN_DIST_FROM_OBSTACLE

            if dist < min_safe_dist:
                return False

        return True

    def _generate_random_target_position(self):
        """Генерация случайной безопасной позиции для цели с проверкой пересечения"""
        max_attempts = 200
        attempt = 0

        while attempt < max_attempts:
            x = random.uniform(*self.TARGET_X_RANGE)
            y = random.uniform(*self.TARGET_Y_RANGE)
            z = self.TARGET_Z

            if self._is_position_safe(x, z):
                return [x, y, z]

            attempt += 1

        print(f"  ⚠️ Не удалось найти безопасную позицию после {max_attempts} попыток")
        # Возвращаем позицию в центре карты
        return [0, 0.5, 0]

    def _randomize_target(self):
        """Случайное перемещение цели с проверкой безопасности"""
        if not self.randomize_target:
            return

        print("\n  🎯 Генерация новой позиции цели...")

        target_pos = self._generate_random_target_position()

        if self.target_node:
            try:
                self.target_node.getField("translation").setSFVec3f(target_pos)
                print(f"     Цель: X={target_pos[0]:6.1f}, Y={target_pos[1]:5.2f}, Z={target_pos[2]:6.2f}")

                # Проверка безопасности
                is_safe = self._is_position_safe(target_pos[0], target_pos[2])
                if is_safe:
                    print(f"     ✅ Позиция безопасна")
                else:
                    print(f"     ⚠️ ВНИМАНИЕ: Позиция может быть небезопасна!")
            except Exception as e:
                print(f"     Ошибка перемещения цели: {e}")
        elif self.use_virtual_target:
            self.virtual_target_pos = target_pos

    # =========================================================================
    # ИСПРАВЛЕННАЯ ФУНКЦИЯ НАГРАДЫ
    # =========================================================================
    def _calculate_reward(self, action):
        """
        РАСЧЁТ НАГРАДЫ ЗА ТЕКУЩИЙ ШАГ
        =================================

        СБАЛАНСИРОВАННАЯ версия с сильным наказанием за столкновения.
        """

        # =========================================================================
        # ПОЛУЧЕНИЕ ТЕКУЩЕГО СОСТОЯНИЯ
        # =========================================================================
        is_goal, dist_to_target = self._check_goal_3d()
        is_collision, collision_info, min_lidar = self._check_collision_lidar()

        current_speed = 0.0
        if self.car_node:
            try:
                vel = self.car_node.getVelocity()
                if vel:
                    current_speed = vel[2]
            except:
                pass

        reward = 0.0

        # =========================================================================
        # 1. ШТРАФ ЗА ВРЕМЯ
        # =========================================================================
        reward -= 0.2

        # =========================================================================
        # 2. ШТРАФ/БОНУС ЗА ДВИЖЕНИЕ
        # =========================================================================
        if abs(current_speed) < 1:
            reward -= 0.5  # Штраф за остановку
        elif abs(current_speed) > 5.0:
            reward += 0.3  # Небольшой бонус за движение

        # =========================================================================
        # 3. ПРОГРЕСС К ЦЕЛИ (УМЕРЕННЫЙ)
        # =========================================================================
        if hasattr(self, 'prev_dist_to_target'):
            progress = self.prev_dist_to_target - dist_to_target

            if progress > 0.01:
                # Приближение к цели - ХОРОШО
                reward += progress * 50.0
                self.steps_without_progress = 0
            elif progress < -0.01:
                # Удаление от цели - ПЛОХО
                reward += progress * 30.0
                self.steps_without_progress += 1
            else:
                self.steps_without_progress += 1

        # =========================================================================
        # 4. ШТРАФ ЗА ОТСУТСТВИЕ ПРОГРЕССА
        # =========================================================================
        if self.steps_without_progress > 100:
            reward -= 2.0
        elif self.steps_without_progress > 50:
            reward -= 1.0
        elif self.steps_without_progress > 25:
            reward -= 0.5

        # =========================================================================
        # 5. НАГРАДА ЗА НАПРАВЛЕНИЕ
        # =========================================================================
        if abs(current_speed) > 0.5:
            angle_to_target = abs(self._get_angle_to_target())
            angle_score = 1.0 - min(angle_to_target / math.pi, 1.0)
            reward += angle_score * 1.0

        # =========================================================================
        # 6. ШТРАФ ЗА БЛИЗОСТЬ К ПРЕПЯТСТВИЮ (УСИЛЕН)
        # =========================================================================
        if min_lidar < 2.0:
            # danger = 1.0 (если расстояние 0.5м)
            # danger = 0.0 (если расстояние 2.0м)
            danger = (2.0 - min(min_lidar, 2.0)) / 2.0
            reward -= danger * 40.0

            # ДОПОЛНИТЕЛЬНЫЙ ШТРАФ если очень близко
            if min_lidar < 0.8:
                reward -= 20.0  # Сильный штраф за критическую близость

        # =========================================================================
        # 7. БОНУС ЗА НОВЫЙ РЕКОРД (УМЕРЕННЫЙ)
        # =========================================================================
        if dist_to_target < self.best_distance:
            improvement = self.best_distance - dist_to_target
            self.best_distance = dist_to_target
            reward += improvement * 30.0

        self.prev_dist_to_target = dist_to_target
        self.prev_speed = current_speed

        if math.isnan(reward) or math.isinf(reward):
            reward = 0.0

        return reward, is_goal, is_collision, collision_info, current_speed

    # =========================================================================
    # ОСНОВНЫЕ МЕТОДЫ
    # =========================================================================
    def _stabilize_car(self):
        """Стабилизация автомобиля после телепортации"""
        print("     🔧 Стабилизация автомобиля...")

        self.car.setCruisingSpeed(0)
        self.car.setSteeringAngle(0)

        for i in range(30):
            self.car.step()
            if (i + 1) % 15 == 0:
                try:
                    pos = self._get_position()
                    print(f"        Стабилизация: {i + 1}/30, высота={pos[1]:.3f}м")
                except:
                    pass

        self.car.setCruisingSpeed(0)
        self.car.setSteeringAngle(0)
        for _ in range(5):
            self.car.step()

        print("     ✅ Стабилизация завершена")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Сохраняем начальное расстояние до цели
        self.dist_at_start = self._check_goal_3d()[1]

        # Рандомизация цели
        if self.randomize_target:
            self._randomize_target()

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
                print(f"⚠️ Ошибка телепортации: {e}")

        self._stabilize_car()

        self.step_count = 0
        self.episode_reward = 0
        self.prev_action = np.array([0.0, 0.0], dtype=np.float32)
        self.prev_speed = 0.0
        self.prev_dist_to_target = self._check_goal_3d()[1]
        self.best_distance = float('inf')
        self.steps_without_progress = 0

        observation = self._get_observation()

        final_pos = self._get_position()
        print(f"     📍 Финальная позиция: X={final_pos[0]:.1f}, Z={final_pos[2]:.1f}, Y={final_pos[1]:.2f}м")

        return observation, {}

    # =========================================================================
    # ИСПРАВЛЕННЫЙ МЕТОД STEP
    # =========================================================================
    def step(self, action):
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

        # Проверка столкновения по позиции
        if not is_collision:
            is_collision_pos, collision_pos_info = self._check_collision_position()
            if is_collision_pos:
                is_collision = True
                collision_info = collision_pos_info

        # =========================================================================
        # ИСПРАВЛЕННЫЕ НАГРАДЫ ЗА ЗАВЕРШЕНИЕ
        # =========================================================================
        if is_goal:
            # Уменьшен бонус
            speed_bonus = max(0, (self.MAX_STEPS - self.step_count) * 0.1)
            reward += 1000 + speed_bonus
            terminated = True
            self.success_count += 1
            info['reason'] = 'goal'
            direction = "вперёд" if current_speed >= 0 else "назад"
            print(f"\n🎯 GOAL! Шагов: {self.step_count}, скорость={current_speed:.1f}, "
                  f"бонус={speed_bonus:.1f}, reward={self.episode_reward + reward:.1f}")

        elif is_collision:
            # Увеличен штраф
            reward -= 800
            terminated = True
            self.collision_count += 1
            info['reason'] = 'collision'
            print(f"\n💥 COLLISION! {collision_info}, reward={self.episode_reward + reward:.1f}")

        elif self.step_count >= self.MAX_STEPS:
            reward -= 100
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
                  f"speed={current_speed:5.2f} {direction} goal={dist_to_target:5.1f}m lidar={min_lidar:4.1f}m")

        return observation, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        print("\n👋 Закрытие среды")


# =============================================================================
# КОЛБЭК ДЛЯ ЛОГИРОВАНИЯ МЕТРИК
# =============================================================================
class ModelMetricsCallback(BaseCallback):
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
# ФУНКЦИИ ОБУЧЕНИЯ И ТЕСТИРОВАНИЯ
# =============================================================================
def train():
    print("\n" + "=" * 70)
    print("ЗАПУСК ОБУЧЕНИЯ НА КАРТЕ С МНОЖЕСТВОМ ПРЕПЯТСТВИЙ")
    print("=" * 70)

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
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log="./logs/",
        policy_kwargs=policy_kwargs,
        device="cpu"
    )

    os.makedirs("./models", exist_ok=True)

    print("\nНачинаем обучение на 2,000,000 шагов...")
    print("✅ Карта: с множеством препятствий")
    print("✅ Препятствия: фиксированные")
    print("✅ Цель: меняются X и Y (высота), Z фиксирован")
    print(f"   X: {env.TARGET_X_RANGE}")
    print(f"   Y: {env.TARGET_Y_RANGE}")
    print(f"   Z (fixed): {env.TARGET_Z} м")
    print(f"   Проверка: цель не пересекается с препятствиями (мин. дист. {env.MIN_DIST_FROM_OBSTACLE}м)")
    print("✅ СБАЛАНСИРОВАННАЯ система наград:")
    print("   - Штраф за шаг: -0.2")
    print("   - Штраф за остановку: -0.5")
    print("   - Бонус за движение: +0.3")
    print("   - Прогресс к цели: +50/м")
    print("   - Удаление от цели: -30/м")
    print("   - Штраф за отсутствие прогресса: до -2.0")
    print("   - Направление: до +1.0")
    print("   - Штраф за близость к препятствию: до -40 + доп. штраф")
    print("   - Штраф за столкновение: -800")
    print("   - Новый рекорд: +30/м")
    print("   - Достижение цели: +1000 + бонус")
    print("✅ Задний ход: ВКЛ (скорость от -5 до 20)")
    print(f"✅ Обнаружено препятствий: {len(env.obstacle_nodes)}")
    print("\n⏱️ Ожидаемое время: ~4-5 часов")
    print("Нажмите Ctrl+C для остановки\n")
    print("💡 Запустите TensorBoard: tensorboard --logdir=./logs\n")

    try:
        model.learn(
            total_timesteps=2_000_000,
            tb_log_name="BMW_X5_OBSTACLES",
            reset_num_timesteps=False,
            callback=ModelMetricsCallback()
        )

        final_path = "./models/bmw_obstacles_final"
        model.save(final_path)
        print(f"\n✅ Финальная модель сохранена: {final_path}")

    except KeyboardInterrupt:
        print("\n\n🛑 Обучение прервано пользователем")
        interrupt_path = "./models/bmw_obstacles_interrupted"
        model.save(interrupt_path)
        print(f"✅ Модель сохранена: {interrupt_path}")

    print("\n" + "=" * 70)
    print("ФИНАЛЬНАЯ СТАТИСТИКА")
    print("=" * 70)
    print(f"Всего эпизодов: {env.episode_count}")
    print(f"Успехов: {env.success_count}")
    print(f"Столкновений: {env.collision_count}")
    if env.episode_count > 0:
        success_rate = (env.success_count / env.episode_count) * 100
        print(f"Успешность: {success_rate:.1f}%")

    return model


def test(model_path="./models/bmw_obstacles_final.zip"):
    print("\n" + "=" * 70)
    print("ТЕСТИРОВАНИЕ МОДЕЛИ")
    print("=" * 70)

    env = BMWRLEnvironment()

    if os.path.exists(model_path):
        model = PPO.load(model_path)
        print(f"✅ Модель загружена: {model_path}")
    else:
        print(f"❌ Модель не найдена: {model_path}")
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


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            model_path = sys.argv[2] if len(sys.argv) > 2 else "./models/bmw_obstacles_final.zip"
            test(model_path)
        else:
            print("Использование:")
            print("  python rl_controller.py           # обучение")
            print("  python rl_controller.py --test    # тестирование")
    else:
        train()