#!/usr/bin/env python3
"""
RL КОНТРОЛЛЕР ДЛЯ BMW X5 В WEBOTS
===================================
Обучение на карте Map_Snow с деревьями и камнями.
Цель генерируется с изменением X и Y (высота), Z фиксирован.
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
    """Среда для обучения на карте Map_Snow"""

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode="human"):
        super().__init__()

        print("=" * 70)
        print("BMW X5 — ОБУЧЕНИЕ НА КАРТЕ Map_Snow")
        print("=" * 70)

        # =====================================================================
        # ИНИЦИАЛИЗАЦИЯ WEBOTS
        # =====================================================================
        self.car = Car()
        self.timestep = int(self.car.getBasicTimeStep())

        # =====================================================================
        # ПОИСК НОД
        # =====================================================================
        print("\nПоиск нод в мире:")

        self.car_node = self.car.getFromDef("CAR")
        if self.car_node:
            print("  ✅ CAR: найдена")
        else:
            print("  ⚠️ CAR: не найдена, использую getSelf()")
            self.car_node = self.car.getSelf()

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
            self.virtual_target_pos = [37.78, 0.5, 0.8936]

        print()

        # =====================================================================
        # СБОР ВСЕХ ПРЕПЯТСТВИЙ (ДЕРЕВЬЯ, КАМНИ)
        # =====================================================================
        print("Поиск препятствий (деревья, камни)...")

        self.obstacle_nodes = []
        self.obstacle_positions = []  # [x, z] для горизонтальной проверки

        def find_obstacles(node, depth=0):
            if depth > 20:
                return

            try:
                node_type = ""
                node_def = ""

                try:
                    node_type = node.getTypeName() if hasattr(node, 'getTypeName') else ""
                except:
                    pass

                try:
                    node_def = node.getDef() if hasattr(node, 'getDef') else ""
                except:
                    pass

                obstacle_types = ["Tree1", "Tree2", "Stone1", "Stones", "tree", "stone"]

                if (node_def not in ["CAR", "TARGET"] and node_type in obstacle_types):
                    try:
                        pos = node.getPosition()
                        if pos and len(pos) >= 3:
                            if -2.0 < pos[1] < 5.0:
                                self.obstacle_nodes.append(node)
                                self.obstacle_positions.append([pos[0], pos[2]])
                                name = node_def if node_def else node_type
                                print(f"     {name}: X={pos[0]:6.1f}, Z={pos[2]:6.1f}")
                    except:
                        pass
            except:
                pass

            try:
                if hasattr(node, 'getField'):
                    fields = node.getField("children")
                    if fields:
                        for i in range(fields.getCount()):
                            child = fields.getMFNode(i)
                            if child:
                                find_obstacles(child, depth + 1)
            except:
                pass

        root = self.car.getRoot()
        if root:
            find_obstacles(root)

        print(f"\n  ✅ Найдено препятствий: {len(self.obstacle_nodes)}")

        # =====================================================================
        # ПАРАМЕТРЫ СРЕДЫ
        # =====================================================================
        # Начальная позиция автомобиля (X, Y, Z)
        self.START_POS = [-33.1187, 0.5, 0.585059]
        self.START_ROT = [0, 1, 0, 0]

        self.GOAL_DISTANCE = 4.0
        self.COLLISION_DISTANCE = 0.8
        self.MAX_STEPS = 6000

        # =====================================================================
        # ПАРАМЕТРЫ РАНДОМИЗАЦИИ ЦЕЛИ
        # =====================================================================
        self.randomize_target = True

        # Диапазоны для рандомизации цели
        # X: вправо-влево (как на скринах: 37.78 и 29.50)
        self.TARGET_X_RANGE = (-35, 45)

        # Y: высота (как на скринах: 0.5 и 12.19)
        self.TARGET_Y_RANGE = (0.3, 15.0)

        # Z: ФИКСИРОВАН (как на скринах: 0.8936)
        self.TARGET_Z = 0.8936

        # Минимальные расстояния
        self.MIN_DIST_FROM_START = 8.0
        self.MIN_DIST_FROM_OBSTACLE = 2.5

        # =====================================================================
        # ПАРАМЕТРЫ CURRICULUM LEARNING
        # =====================================================================
        self.curriculum_learning = True
        self.num_obstacles = 1

        self.difficulty_thresholds = [500, 700, 900]
        self.difficulty_levels = ["ЛЁГКАЯ (1 преп.)",
                                  "СРЕДНЯЯ (2 преп.)",
                                  "СЛОЖНАЯ (3 преп.)",
                                  "ЭКСПЕРТНАЯ (4 преп.)"]
        self.current_difficulty_index = 0

        # =====================================================================
        # ПАРАМЕТРЫ ДЛЯ НЕСКОЛЬКИХ ПРЕПЯТСТВИЙ
        # =====================================================================
        self.obstacle_list = []
        self.obstacle_sizes = []
        self.OBSTACLE_SIZE_RANGE = (0.8, 2.0)

        # Диапазоны для препятствий (на земле)
        self.OBSTACLE_X_RANGE = (-8, 8)
        self.OBSTACLE_Z_RANGE = (-8, 8)
        self.OBSTACLE_Y = 0.5

        # =====================================================================
        # СТАТИСТИКА
        # =====================================================================
        self.scenario_stats = {'easy': 0, 'medium': 0, 'hard': 0}
        self.MIN_DIST_BETWEEN_OBJ = 3.0

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
        print(f"  Goal distance: {self.GOAL_DISTANCE} м")
        print(f"  Collision distance: {self.COLLISION_DISTANCE} м")
        print(f"  Max steps: {self.MAX_STEPS}")
        print(f"  Target X range: {self.TARGET_X_RANGE}")
        print(f"  Target Y range: {self.TARGET_Y_RANGE}")
        print(f"  Target Z (fixed): {self.TARGET_Z} м")
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
                print(f"  {device.getName()}")

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
        return [37.78, 0.5, 0.8936]

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
    # МЕТОДЫ ДЛЯ ГЕНЕРАЦИИ ПРЕПЯТСТВИЙ
    # =========================================================================
    def _create_obstacle(self, position, size=1.0, index=0):
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
        if hasattr(self, 'obstacle_list'):
            for obs_node in self.obstacle_list:
                if obs_node:
                    try:
                        obs_node.remove()
                    except:
                        pass
        self.obstacle_list = []
        self.obstacle_sizes = []

    def _generate_random_position(self, for_target=True, exclude_positions=None):
        """Генерация случайной позиции"""
        max_attempts = 100
        attempt = 0

        if exclude_positions is None:
            exclude_positions = []

        while attempt < max_attempts:
            if for_target:
                x = random.uniform(*self.TARGET_X_RANGE)
                y = random.uniform(*self.TARGET_Y_RANGE)  # Y - высота!
                z = self.TARGET_Z  # Z - ФИКСИРОВАН!
            else:
                x = random.uniform(*self.OBSTACLE_X_RANGE)
                y = self.OBSTACLE_Y  # Препятствия на земле
                z = random.uniform(*self.OBSTACLE_Z_RANGE)

            # Проверка расстояния от старта (горизонтальное)
            dist_from_start = math.sqrt(
                (x - self.START_POS[0]) ** 2 +
                (z - self.START_POS[2]) ** 2
            )

            if dist_from_start < self.MIN_DIST_FROM_START:
                attempt += 1
                continue

            # Проверка расстояния от других объектов (горизонтальное)
            valid = True
            for pos in exclude_positions:
                if pos is not None and len(pos) >= 2:
                    dist = math.sqrt(
                        (x - pos[0]) ** 2 +
                        (z - pos[1]) ** 2
                    )
                    if dist < self.MIN_DIST_BETWEEN_OBJ:
                        valid = False
                        break

            if valid:
                return [x, y, z]

            attempt += 1

        print(f"⚠️ Не удалось найти позицию, использую значение по умолчанию")
        if for_target:
            return [37.78, 0.5, 0.8936]
        else:
            return [0, 0.5, 0]

    # =========================================================================
    # МЕТОДЫ ДЛЯ БЕЗОПАСНОЙ ГЕНЕРАЦИИ ЦЕЛИ
    # =========================================================================
    def _is_position_safe(self, x, z):
        """Проверка безопасной позиции для цели (горизонтальная)"""
        # Расстояние от старта
        dist_from_start = math.sqrt(
            (x - self.START_POS[0]) ** 2 +
            (z - self.START_POS[2]) ** 2
        )
        if dist_from_start < self.MIN_DIST_FROM_START:
            return False

        # Расстояние от всех препятствий
        for pos in self.obstacle_positions:
            dist = math.sqrt((x - pos[0]) ** 2 + (z - pos[1]) ** 2)
            if dist < self.MIN_DIST_FROM_OBSTACLE:
                return False

        return True

    def _generate_random_target_position(self):
        """Генерация случайной безопасной позиции для цели"""
        max_attempts = 200
        attempt = 0

        while attempt < max_attempts:
            x = random.uniform(*self.TARGET_X_RANGE)
            y = random.uniform(*self.TARGET_Y_RANGE)  # Y - высота!
            z = self.TARGET_Z  # Z - ФИКСИРОВАН!

            if self._is_position_safe(x, z):
                return [x, y, z]

            attempt += 1

        print(f"  ⚠️ Не удалось найти безопасную позицию после {max_attempts} попыток")
        return [37.78, 0.5, 0.8936]

    def _randomize_target(self):
        """Случайное перемещение цели"""
        if not self.randomize_target:
            return

        print("\n  🎯 Генерация новой позиции цели...")

        target_pos = self._generate_random_target_position()

        if self.target_node:
            try:
                self.target_node.getField("translation").setSFVec3f(target_pos)
                print(f"     Цель: X={target_pos[0]:6.1f}, Y={target_pos[1]:5.2f}, Z={target_pos[2]:6.2f}")

                is_safe = self._is_position_safe(target_pos[0], target_pos[2])
                if is_safe:
                    print(f"     ✅ Позиция безопасна (дист. от препятствий > {self.MIN_DIST_FROM_OBSTACLE}м)")
            except Exception as e:
                print(f"     Ошибка перемещения цели: {e}")
        elif self.use_virtual_target:
            self.virtual_target_pos = target_pos

    # =========================================================================
    # МЕТОДЫ ДЛЯ УПРАВЛЕНИЯ СЛОЖНОСТЬЮ
    # =========================================================================
    def _update_difficulty(self):
        if not self.curriculum_learning:
            return

        old_num = self.num_obstacles

        if self.episode_count < self.difficulty_thresholds[0]:
            self.num_obstacles = 1
            self.current_difficulty_index = 0
        elif self.episode_count < self.difficulty_thresholds[1]:
            self.num_obstacles = 2
            self.current_difficulty_index = 1
        elif self.episode_count < self.difficulty_thresholds[2]:
            self.num_obstacles = 3
            self.current_difficulty_index = 2
        else:
            self.num_obstacles = 4
            self.current_difficulty_index = 3

        if old_num != self.num_obstacles:
            print(f"\n{'=' * 70}")
            print(f"📈 ПОВЫШЕНИЕ СЛОЖНОСТИ")
            print(f"  Было: {old_num} препятствие(й)")
            print(f"  Стало: {self.num_obstacles} препятствие(й)")
            print(f"{'=' * 70}\n")

    def _randomize_scenario(self):
        if not self.randomize_scenarios:
            return

        self._update_difficulty()

        print(f"\n  🎲 Генерация нового сценария [{self.difficulty_levels[self.current_difficulty_index]}]...")

        self._remove_all_obstacles()

        all_obstacle_positions = []

        for i in range(self.num_obstacles):
            size = random.uniform(*self.OBSTACLE_SIZE_RANGE)

            exclude = all_obstacle_positions + [[self.START_POS[0], self.START_POS[2]]]
            obs_pos = self._generate_random_position(
                for_target=False,
                exclude_positions=exclude
            )

            obs_node = self._create_obstacle(obs_pos, size, i)

            if obs_node:
                self.obstacle_list.append(obs_node)
                self.obstacle_sizes.append(size)
                all_obstacle_positions.append([obs_pos[0], obs_pos[2]])
                print(f"     Препятствие {i + 1}: X={obs_pos[0]:5.1f}, Z={obs_pos[2]:5.1f}, размер={size:.1f}м")

        # Генерация цели
        target_pos = self._generate_random_target_position()

        if self.target_node:
            try:
                self.target_node.getField("translation").setSFVec3f(target_pos)
                print(f"     Цель: X={target_pos[0]:5.1f}, Y={target_pos[1]:5.1f}, Z={target_pos[2]:5.1f}")
            except Exception as e:
                print(f"     Ошибка перемещения цели: {e}")

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

        # =====================================================================
        # 1. УМЕРЕННЫЙ ШТРАФ ЗА ВРЕМЯ
        # =====================================================================
        reward -= 0.2  # Уменьшен с 0.5

        # =====================================================================
        # 2. МЯГКИЙ ШТРАФ ЗА ОСТАНОВКУ
        # =====================================================================
        if abs(current_speed) < 0.3:
            reward -= 0.5  # Уменьшен с 2.0
        elif abs(current_speed) > 1.0:
            reward += 0.3  # Небольшой бонус за движение

        # =====================================================================
        # 3. ПРОГРЕСС К ЦЕЛИ (главная награда)
        # =====================================================================
        if hasattr(self, 'prev_dist_to_target'):
            progress = self.prev_dist_to_target - dist_to_target

            if progress > 0.01:
                # Приближение к цели - ХОРОШО
                reward += progress * 100.0  # Увеличено с 50
                self.steps_without_progress = 0
            elif progress < -0.01:
                # Удаление от цели - ПЛОХО
                reward += progress * 50.0  # Увеличено с 20
                self.steps_without_progress += 1
            else:
                self.steps_without_progress += 1

        # =====================================================================
        # 4. УМЕРЕННЫЙ ШТРАФ ЗА ОТСУТСТВИЕ ПРОГРЕССА
        # =====================================================================
        if self.steps_without_progress > 100:
            reward -= 2.0  # Уменьшен с 10
        elif self.steps_without_progress > 50:
            reward -= 1.0  # Уменьшен с 5
        elif self.steps_without_progress > 25:
            reward -= 0.5  # Уменьшен с 2

        # =====================================================================
        # 5. НАГРАДА ЗА НАПРАВЛЕНИЕ (только если едем)
        # =====================================================================
        if abs(current_speed) > 0.5:
            angle_to_target = abs(self._get_angle_to_target())
            angle_score = 1.0 - min(angle_to_target / math.pi, 1.0)
            reward += angle_score * 1.0  # Уменьшен с 2.0

        # =====================================================================
        # 6. ШТРАФ ЗА БЛИЗОСТЬ К ПРЕПЯТСТВИЮ
        # =====================================================================
        if min_lidar < 1.5:
            danger = (1.5 - min_lidar) / 1.5
            reward -= danger * 8.0

        # =====================================================================
        # 7. БОНУС ЗА НОВЫЙ РЕКОРД
        # =====================================================================
        if dist_to_target < self.best_distance:
            improvement = self.best_distance - dist_to_target
            self.best_distance = dist_to_target
            reward += improvement * 40.0  # Увеличено с 30

        self.prev_dist_to_target = dist_to_target
        self.prev_speed = current_speed

        if math.isnan(reward) or math.isinf(reward):
            reward = 0.0

        return reward, is_goal, is_collision, collision_info, current_speed

    # =========================================================================
    # ОСНОВНЫЕ МЕТОДЫ
    # =========================================================================
    def _stabilize_car(self):
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

        self.dist_at_start = self._check_goal_3d()[1]

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

        if is_goal:
            # =====================================================================
            # БОЛЬШАЯ НАГРАДА ЗА ЦЕЛЬ (должна перевешивать штрафы)
            # =====================================================================
            # Базовая награда + бонус за быстроту
            speed_bonus = max(0, (self.MAX_STEPS - self.step_count) * 0.3)  # Увеличено
            reward += 2000 + speed_bonus  # Увеличено с 500

            # Бонус за эффективность (чем меньше шагов, тем лучше)
            efficiency_bonus = max(0, (self.MAX_STEPS - self.step_count) * 0.5)
            reward += efficiency_bonus

            # Если цель была близко к старту, дополнительный бонус
            dist_at_start = self.prev_dist_to_target_at_reset
            if hasattr(self, 'dist_at_start'):
                distance_covered = self.dist_at_start - self.prev_dist_to_target
                reward += distance_covered * 10.0

            terminated = True
            self.success_count += 1
            info['reason'] = 'goal'
            direction = "вперёд" if current_speed >= 0 else "назад"
            print(f"\n🎯 GOAL! Шагов: {self.step_count}, скорость={current_speed:.1f}, "
                  f"бонус={speed_bonus:.1f}, всего={self.episode_reward + reward:.1f}")

        elif is_collision:
            reward -= 200
            terminated = True
            self.collision_count += 1
            info['reason'] = 'collision'
            print(f"\n💥 COLLISION! {collision_info}, reward={self.episode_reward + reward:.1f}")

        elif self.step_count >= self.MAX_STEPS:
            reward -= 50
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
            # Добавить проверку на кружение
            if self.step_count > 50:
                # Проверяем, изменилась ли позиция за последние 50 шагов
                if hasattr(self, 'prev_position_history'):
                    self.position_history.append(self._get_position())
                    if len(self.position_history) > 50:
                        self.position_history.popleft()
                        # Вычисляем среднее перемещение
                        start_pos = self.position_history[0]
                        end_pos = self.position_history[-1]
                        movement = math.sqrt(
                            (end_pos[0] - start_pos[0]) ** 2 +
                            (end_pos[2] - start_pos[2]) ** 2
                        )
                        if movement < 1.0 and self.step_count > 100:
                            # Штраф за кружение/застревание
                            reward -= 5.0
                            print(f"     ⚠️ Штраф за кружение! Перемещение={movement:.2f}м")
                else:
                    from collections import deque
                    self.position_history = deque(maxlen=50)

        return observation, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        self._remove_all_obstacles()
        print("\n👋 Закрытие среды")


# =============================================================================
# КОЛБЭК
# =============================================================================
class ModelMetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_success = []
        self.episode_collisions = []
        self.episode_timeouts = []

        # Для отслеживания прогресса
        self.steps_without_progress = 0
        self.prev_steering = 0.0
        self.position_history = None  # Будет создан в step при необходимости

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
# ФУНКЦИИ ОБУЧЕНИЯ И ТЕСТИРОВАНИЯ
# =============================================================================
def train():
    print("\n" + "=" * 70)
    print("ЗАПУСК ОБУЧЕНИЯ НА КАРТЕ Map_Snow")
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

    print("\nНачинаем обучение на 3,000,000 шагов...")
    print("✅ Карта: Map_Snow (деревья, камни)")
    print("✅ Цель: меняются X и Y (высота), Z фиксирован")
    print(f"   X: {env.TARGET_X_RANGE}")
    print(f"   Y: {env.TARGET_Y_RANGE}")
    print(f"   Z (fixed): {env.TARGET_Z}")
    print("✅ Curriculum Learning: 1→4 препятствия")
    print("✅ Задний ход: ВКЛ (скорость от -5 до 20)")
    print(f"✅ Обнаружено препятствий: {len(env.obstacle_nodes)}")
    print("\n⏱️ Ожидаемое время: ~6-8 часов")
    print("Нажмите Ctrl+C для остановки\n")
    print("💡 Запустите TensorBoard: tensorboard --logdir=./logs\n")

    try:
        model.learn(
            total_timesteps=3_000_000,
            tb_log_name="BMW_X5_SNOW",
            reset_num_timesteps=False,
            callback=ModelMetricsCallback()
        )

        final_path = "./models/bmw_snow_final"
        model.save(final_path)
        print(f"\n✅ Финальная модель сохранена: {final_path}")

    except KeyboardInterrupt:
        print("\n\n🛑 Обучение прервано пользователем")
        interrupt_path = "./models/bmw_snow_interrupted"
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


def test(model_path="./models/bmw_snow_final.zip"):
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
            model_path = sys.argv[2] if len(sys.argv) > 2 else "./models/bmw_snow_final.zip"
            test(model_path)
        else:
            print("Использование:")
            print("  python rl_controller.py           # обучение")
            print("  python rl_controller.py --test    # тестирование")
    else:
        train()