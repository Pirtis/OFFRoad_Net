#!/usr/bin/env python3


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
    RL среда с наградами, рандомизацией препятствий и задним ходом
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode="human"):
        super().__init__()

        print("=" * 60)
        print("BMW X5 — RL СРЕДА с РАНДОМИЗАЦИЕЙ и ЗАДНИМ ХОДОМ (v7.0)")
        print("=" * 60)

        self.car = Car()
        self.timestep = int(self.car.getBasicTimeStep())

        # Поиск нод
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

        # Параметры среды
        self.START_POS = [-11.5, 0.5, 0.5]
        self.START_ROT = [0, 1, 0, 0]

        self.GOAL_DISTANCE = 4.0
        self.COLLISION_DISTANCE = 0.5
        self.MAX_STEPS = 2500

        # === Параметры рандомизации ===
        self.randomize_scenarios = True

        # Диапазоны для рандомизации
        self.OBSTACLE_X_RANGE = (-8, 8)
        self.OBSTACLE_Y_RANGE = (-8, 8)
        self.OBSTACLE_Z = 1.0

        self.TARGET_X_RANGE = (-12, 12)
        self.TARGET_Y_RANGE = (-10, 10)
        self.TARGET_Z = 0.5

        # Минимальные расстояния
        self.MIN_DIST_FROM_START = 6.0
        self.MIN_DIST_BETWEEN_OBJ = 3.0

        # Счётчики для статистики
        self.scenario_stats = {
            'easy': 0,
            'medium': 0,
            'hard': 0
        }

        self._setup_sensors()

        # === Пространство наблюдений ===
        self.observation_space = spaces.Box(
            low=np.array([-30, -10, -30, -np.pi, -np.pi,
                          0, 0, 0, 0, 0, 0, -0.8], dtype=np.float32),
            high=np.array([30, 10, 30, np.pi, np.pi,
                           25, 25, 25, 25, 30, 1000, 0.8], dtype=np.float32),
            dtype=np.float32
        )

        # === Пространство действий с задним ходом ===
        self.action_space = spaces.Box(
            low=np.array([-0.8, -5.0], dtype=np.float32),
            high=np.array([0.8, 20.0], dtype=np.float32),
            dtype=np.float32
        )

        self.step_count = 0
        self.episode_reward = 0
        self.episode_count = 0
        self.success_count = 0
        self.collision_count = 0
        self.prev_action = np.array([0.0, 0.0], dtype=np.float32)
        self.prev_speed = 0.0  # НОВОЕ: для отслеживания скорости

        self.best_distance = float('inf')

        self.render_mode = render_mode

        print(f"✅ Goal distance: {self.GOAL_DISTANCE}m")
        print(f"✅ Collision distance: {self.COLLISION_DISTANCE}m")
        print(f"✅ Max steps: {self.MAX_STEPS}")
        print(f"✅ Randomization: {'ON' if self.randomize_scenarios else 'OFF'}")
        print(f"✅ Reverse gear: ENABLED (speed range: -5 to 20)")
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

    def _generate_random_position(self, for_target=True, exclude_positions=None):
        """
        Генерация случайной позиции для цели или препятствия
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

        print("\n  🎲 Генерация нового сценария...")

        obs_pos = self._generate_random_position(for_target=False, exclude_positions=None)
        target_pos = self._generate_random_position(for_target=True, exclude_positions=[obs_pos])

        if self.obstacle_node:
            try:
                self.obstacle_node.getField("translation").setSFVec3f(obs_pos)
                print(f"     Препятствие: ({obs_pos[0]:.1f}, {obs_pos[2]:.1f})")
            except Exception as e:
                print(f"     Ошибка перемещения препятствия: {e}")

        if self.target_node:
            try:
                self.target_node.getField("translation").setSFVec3f(target_pos)
                print(f"     Цель: ({target_pos[0]:.1f}, {target_pos[2]:.1f})")
            except Exception as e:
                print(f"     Ошибка перемещения цели: {e}")
        elif self.use_virtual_target:
            self.virtual_target_pos = target_pos
            print(f"     Виртуальная цель: ({target_pos[0]:.1f}, {target_pos[2]:.1f})")

        dist_obs_target = math.sqrt(
            (obs_pos[0] - target_pos[0]) ** 2 +
            (obs_pos[2] - target_pos[2]) ** 2
        )

        if dist_obs_target > 10:
            difficulty = "easy"
        elif dist_obs_target > 5:
            difficulty = "medium"
        else:
            difficulty = "hard"

        self.scenario_stats[difficulty] += 1

        if self.episode_count % 100 == 0 and self.episode_count > 0:
            total = sum(self.scenario_stats.values())
            print(f"\n  📊 Статистика сценариев:")
            for diff, count in self.scenario_stats.items():
                percentage = (count / total) * 100 if total > 0 else 0
                print(f"     {diff}: {count} ({percentage:.1f}%)")

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
        return [10, 0.5, 0.4]

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
                    # Знак скорости определяем по направлению движения
                    forward_vel = vel[2]  # Z - вперёд/назад
                    current_speed = forward_vel  # Положительная = вперёд, отрицательная = назад
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

    def _calculate_reward(self, action):
        is_goal, dist_to_target = self._check_goal_3d()
        is_collision, collision_info, min_lidar = self._check_collision_lidar()

        # Текущая скорость (с учётом направления)
        current_speed = 0.0
        if self.car_node:
            try:
                vel = self.car_node.getVelocity()
                if vel:
                    current_speed = vel[2]  # Z - вперёд/назад
            except:
                pass

        reward = 0.0

        # Награды с учётом заднего хода

        # Базовый штраф за время
        reward -= 0.1

        # Награда за любое движение (вперёд или назад)
        if abs(current_speed) > 0.5:
            reward += 1.0
        else:
            reward -= 1.0  # Штраф за остановку

        # Прогресс к цели
        if hasattr(self, 'prev_dist_to_target'):
            progress = self.prev_dist_to_target - dist_to_target
            if progress > 0.01:  # Приближаемся
                reward += progress * 50.0
            elif progress < -0.01:  # Удаляемся
                reward += progress * 20.0

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

        # Штраф за резкую смену направления
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

        # Бонус за новый рекорд
        if dist_to_target < self.best_distance:
            self.best_distance = dist_to_target
            reward += 10.0

        self.prev_dist_to_target = dist_to_target
        self.prev_speed = current_speed

        if math.isnan(reward) or math.isinf(reward):
            reward = 0.0

        return reward, is_goal, is_collision, collision_info, current_speed

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.randomize_scenarios:
            self._randomize_scenario()

        self.episode_count += 1

        if self.episode_count > 1:
            success_rate = (self.success_count / (self.episode_count - 1)) * 100 if self.episode_count > 1 else 0
            avg_reward = self.episode_reward / max(self.step_count, 1)
            print(f"\nЭпизод {self.episode_count - 1}: награда={self.episode_reward:.1f}, "
                  f"средняя={avg_reward:.2f}/шаг, успех={success_rate:.1f}%")

        print(f"\nСброс эпизода {self.episode_count}...")

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

        observation = self._get_observation()
        return observation, {}

    def step(self, action):
        action = np.array(action, dtype=np.float32)
        action = np.nan_to_num(action, nan=0.0, posinf=0.8, neginf=-0.8)

        # Исследование только в первых 20 эпизодах
        if self.episode_count < 20:
            action[0] += np.random.normal(0, 0.2)
            action[1] += np.random.normal(0, 2)

        # === ИЗМЕНЕНО: Плавное управление с учётом заднего хода ===
        smoothed_steering = 0.7 * self.prev_action[0] + 0.3 * action[0]
        steering = float(np.clip(smoothed_steering, -0.8, 0.8))

        # Скорость теперь может быть отрицательной
        smoothed_speed = 0.7 * self.prev_speed + 0.3 * action[1]
        speed = float(np.clip(smoothed_speed, -5.0, 20.0))

        self.prev_action = np.array([steering, speed], dtype=np.float32)

        # Применяем действие (Webots сам поймёт, что отрицательная скорость = задний ход)
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
            reward += 500
            terminated = True
            self.success_count += 1
            info['reason'] = 'goal'
            direction = "вперёд" if current_speed >= 0 else "назад"
            print(
                f"\nGOAL! dist={self.prev_dist_to_target:.2f}m, направление={direction}, reward={self.episode_reward + reward:.1f}")

        elif is_collision:
            reward -= 200
            terminated = True
            self.collision_count += 1
            info['reason'] = 'collision'
            print(f"\nCOLLISION! {collision_info}, reward={self.episode_reward + reward:.1f}")

        elif self.step_count >= self.MAX_STEPS:
            reward -= 50
            truncated = True
            info['reason'] = 'timeout'
            print(f"\nTIMEOUT! dist={self.prev_dist_to_target:.1f}m, reward={self.episode_reward + reward:.1f}")

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
        print("Закрытие среды")
        if self.randomize_scenarios:
            total = sum(self.scenario_stats.values())
            print("\n📊 ИТОГОВАЯ СТАТИСТИКА СЦЕНАРИЕВ:")
            for diff, count in self.scenario_stats.items():
                percentage = (count / total) * 100 if total > 0 else 0
                print(f"   {diff}: {count} ({percentage:.1f}%)")


class ModelMetricsCallback(BaseCallback):
    """
    Callback для логирования метрик модели в TensorBoard
    """

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

                    if "final_distance" in info and info["final_distance"] is not None:
                        self.logger.record("environment/final_distance", info["final_distance"])

        return True


def train():
    print("\n" + "=" * 60)
    print("ЗАПУСК ОБУЧЕНИЯ (С РАНДОМИЗАЦИЕЙ И ЗАДНИМ ХОДОМ)")
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
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log="./logs/",
        policy_kwargs=policy_kwargs,
        device="cpu"
    )

    os.makedirs("./models", exist_ok=True)

    print("\nНачинаем обучение на 500,000 шагов...")
    print("✅ Рандомизация сценариев ВКЛЮЧЕНА")
    print("✅ Задний ход ВКЛЮЧЕН (скорость от -5 до 20)")
    print("🎲 Каждый эпизод - новая позиция препятствия и цели")
    print("📊 Все метрики модели логируются в TensorBoard")
    print("Нажмите Ctrl+C для остановки\n")
    print("💡 Запустите TensorBoard: tensorboard --logdir=./logs\n")

    try:
        model.learn(
            total_timesteps=1_500_000,
            tb_log_name="BMW_X5_RL_REVERSE",
            reset_num_timesteps=False,
            callback=ModelMetricsCallback()
        )

        final_path = "./models/bmw_rl_reverse_final"
        model.save(final_path)
        print(f"\nФинальная модель сохранена: {final_path}")

    except KeyboardInterrupt:
        print("\n\nОбучение прервано пользователем")
        interrupt_path = "./models/bmw_rl_reverse_interrupted"
        model.save(interrupt_path)
        print(f"Модель сохранена: {interrupt_path}")

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


def test(model_path="./models/bmw_rl_reverse_final.zip"):
    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ МОДЕЛИ")
    print("=" * 60)

    env = BMWRLEnvironment()

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
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        if step % 50 == 0:
            _, dist_target = env._check_goal_3d()
            _, _, min_lidar = env._check_collision_lidar()
            print(f"Step {step}: reward={reward:.2f}, total={total_reward:.1f}, "
                  f"goal={dist_target:.1f}m, lidar={min_lidar:.1f}m")

        if terminated or truncated:
            print(f"\n📊 Эпизод завершен! Шагов: {step}, награда: {total_reward:.1f}")
            print(f"Причина: {info.get('reason', 'unknown')}")
            break


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            model_path = sys.argv[2] if len(sys.argv) > 2 else "./models/bmw_rl_reverse(рандом препятствий, задний ход).zip"
            test(model_path)
        else:
            print("Использование:")
            print("  python rl_controller.py           # обучение")
            print("  python rl_controller.py --test    # тестирование")
    else:
        train()