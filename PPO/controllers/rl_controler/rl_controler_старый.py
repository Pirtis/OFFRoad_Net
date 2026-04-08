#!/usr/bin/env python3
"""
RL Controller — улучшенная версия с интеллектуальным объездом препятствий
"""

from vehicle import Car
import numpy as np
import math
import random


class BMWX5Controller:
    def __init__(self):
        print("=" * 60)
        print("🚗 BMW X5 — ИНТЕЛЛЕКТУАЛЬНОЕ УПРАВЛЕНИЕ")
        print("=" * 60)

        self.car = Car()
        self.timestep = int(self.car.getBasicTimeStep())

        self.car_node = self.car.getFromDef("CAR")
        self.obstacle_node = self.car.getFromDef("obstacle")
        self.target_node = self.car.getFromDef("target")

        self._setup_sensors()

        self.START_POS = [-11.5, 0.5, 0]
        self.START_ROT = [0, 1, 0, 0]
        self.COLLISION_DISTANCE = 3.5
        self.COLLISION_HEIGHT_TOLERANCE = 1.5

        # Параметры для объезда препятствий
        self.OBSTACLE_AVOID_DISTANCE = 8.0  # Дистанция начала объезда
        self.OBSTACLE_SAFE_DISTANCE = 4.5  # Безопасная дистанция
        self.AVOIDANCE_ANGLE = 0.6  # Угол поворота при объезде

        # Цель достигается только если машина близко по ВСЕМ трем осям
        self.GOAL_DISTANCE_X = 3.0
        self.GOAL_DISTANCE_Y = 1.5
        self.GOAL_DISTANCE_Z = 1.0

        self.max_steps = 2000
        self.episode_count = 0
        self.success_count = 0
        self.collision_count = 0
        self.step_count = 0
        self.episode_reward = 0
        self.is_resetting = False

        # Состояния для объезда препятствия
        self.avoidance_state = "NONE"  # NONE, LEFT, RIGHT
        self.avoidance_timer = 0
        self.avoidance_duration = 150  # Длительность маневра объезда

        print("\n✅ Готов!\n")
        print(f"🎯 Цель: X<{self.GOAL_DISTANCE_X}м, Y<{self.GOAL_DISTANCE_Y}м, Z<{self.GOAL_DISTANCE_Z}м")
        print(f"💥 Столкновение: <{self.COLLISION_DISTANCE}м")

    def _setup_sensors(self):
        self.lidar_front = self.car.getDevice("lidar_front")
        self.lidar_left = self.car.getDevice("lidar_left")
        self.lidar_right = self.car.getDevice("lidar_right")
        self.gps = self.car.getDevice("gps")
        self.imu = self.car.getDevice("imu")

        for device in [self.lidar_front, self.lidar_left, self.lidar_right, self.gps, self.imu]:
            if device:
                device.enable(self.timestep)

    def _get_lidar_data(self, lidar):
        """Получение и фильтрация данных лидара"""
        if not lidar:
            return None
        data = lidar.getRangeImage()
        if data:
            # Фильтруем некорректные значения
            return [x if 0.1 < x < 100 and not math.isinf(x) else 20.0 for x in data]
        return None

    def _get_lidar_min(self, lidar):
        """Минимальное расстояние до препятствия"""
        data = self._get_lidar_data(lidar)
        if data:
            return min(data)
        return 20.0

    def _get_lidar_sectors(self, data, num_sectors=3):
        """Разделение данных лидара на сектора для лучшего анализа"""
        if not data:
            return [20.0] * num_sectors

        sector_size = len(data) // num_sectors
        sectors = []

        for i in range(num_sectors):
            start = i * sector_size
            end = start + sector_size if i < num_sectors - 1 else len(data)
            sector_data = data[start:end]
            sectors.append(min(sector_data))

        return sectors

    def _get_position(self):
        if self.gps:
            return self.gps.getValues()
        if self.car_node:
            return self.car_node.getPosition()
        return [0, 0, 0]

    def _get_obstacle_relative_position(self):
        """Получение относительной позиции препятствия"""
        if not self.obstacle_node:
            return None, None

        car_pos = self._get_position()
        car_rot = self.imu.getRollPitchYaw()[2] if self.imu else 0
        obs_pos = self.obstacle_node.getPosition()

        # Вектор от машины к препятствию
        dx = obs_pos[0] - car_pos[0]
        dz = obs_pos[2] - car_pos[2]

        # Расстояние
        distance = math.sqrt(dx * dx + dz * dz)

        # Угол до препятствия относительно направления машины
        angle_to_obs = math.atan2(dz, dx)
        relative_angle = angle_to_obs - car_rot

        # Нормализуем угол
        while relative_angle > math.pi:
            relative_angle -= 2 * math.pi
        while relative_angle < -math.pi:
            relative_angle += 2 * math.pi

        return distance, relative_angle

    def _check_goal_reached(self):
        """Проверка достижения цели по ВСЕМ трем координатам"""
        if not self.target_node:
            return False

        car_pos = self._get_position()
        target_pos = self.target_node.getPosition()

        dx = abs(car_pos[0] - target_pos[0])
        dy = abs(car_pos[1] - target_pos[1])
        dz = abs(car_pos[2] - target_pos[2])

        return (dx < self.GOAL_DISTANCE_X and
                dy < self.GOAL_DISTANCE_Y and
                dz < self.GOAL_DISTANCE_Z)

    def _check_collision(self):
        """Проверка столкновения только с препятствием"""
        if not self.obstacle_node:
            return False

        car_pos = self._get_position()
        obs_pos = self.obstacle_node.getPosition()

        # Проверяем по вертикали
        dy = abs(car_pos[1] - obs_pos[1])
        if dy > self.COLLISION_HEIGHT_TOLERANCE:
            return False

        # Проверяем по горизонтали
        dx = abs(car_pos[0] - obs_pos[0])
        dz = abs(car_pos[2] - obs_pos[2])

        return dx < self.COLLISION_DISTANCE and dz < 2.0

    def _decide_avoidance_direction(self, obs_distance, obs_angle):
        """Принятие решения о направлении объезда препятствия"""
        if obs_distance > self.OBSTACLE_AVOID_DISTANCE:
            return "NONE"

        # Получаем данные с боковых лидаров
        left_dist = self._get_lidar_min(self.lidar_left)
        right_dist = self._get_lidar_min(self.lidar_right)

        print(f"  📡 Лидары: левый={left_dist:.1f}м, правый={right_dist:.1f}м")

        # Анализируем свободное пространство
        if left_dist > right_dist + 1.0 and left_dist > 3.0:
            return "LEFT"
        elif right_dist > left_dist + 1.0 and right_dist > 3.0:
            return "RIGHT"
        else:
            # Если оба направления примерно равны, выбираем то,
            # которое ближе к направлению на цель
            angle_to_target = self._get_angle_to_target()
            return "LEFT" if angle_to_target > 0 else "RIGHT"

    def _get_angle_to_target(self):
        """Угол до цели для поворота"""
        if not self.target_node or not self.imu:
            return 0

        car_pos = self._get_position()
        target_pos = self.target_node.getPosition()

        # Вектор к цели в плоскости XZ
        dx = target_pos[0] - car_pos[0]
        dz = target_pos[2] - car_pos[2]

        # Глобальный угол до цели
        target_angle = math.atan2(dz, dx)

        # Текущий поворот машины
        yaw = self.imu.getRollPitchYaw()[2]

        # Относительный угол
        angle = target_angle - yaw
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi

        return angle

    def _calculate_avoidance_path(self, obs_distance, obs_angle):
        """Расчет траектории объезда препятствия"""

        # Если препятствие близко, начинаем объезд
        if obs_distance < self.OBSTACLE_AVOID_DISTANCE and abs(obs_angle) < 1.0:
            if self.avoidance_state == "NONE":
                self.avoidance_state = self._decide_avoidance_direction(obs_distance, obs_angle)
                self.avoidance_timer = self.avoidance_duration
                print(f"  🚧 Начинаем объезд {self.avoidance_state}")

        # Если объезд затянулся, сбрасываем состояние
        if self.avoidance_timer <= 0:
            self.avoidance_state = "NONE"

        self.avoidance_timer -= 1

        # Расчет угла поворота в зависимости от состояния
        if self.avoidance_state == "LEFT":
            return -self.AVOIDANCE_ANGLE
        elif self.avoidance_state == "RIGHT":
            return self.AVOIDANCE_ANGLE
        else:
            return None

    def _reset_episode(self, reason):
        self.episode_count += 1
        if reason == "goal":
            self.success_count += 1
        elif reason == "collision":
            self.collision_count += 1

        print(f"\n{'=' * 50}")
        print(f"📊 ЭПИЗОД {self.episode_count}: {reason.upper()}")
        print(f"   Награда: {self.episode_reward:.1f}")
        print(f"{'=' * 50}")

        # Остановка
        for _ in range(20):
            self.car.setCruisingSpeed(0)
            self.car.setSteeringAngle(0)
            self.car.step()

        # Сброс физики
        self.car.simulationResetPhysics()
        for _ in range(5):
            self.car.step()

        # Телепортация
        if self.car_node:
            self.car_node.getField("rotation").setSFRotation(self.START_ROT)
            for _ in range(3):
                self.car.step()
            self.car_node.getField("translation").setSFVec3f(self.START_POS)
            for _ in range(5):
                self.car.step()

        # Стабилизация
        for _ in range(50):
            self.car.setCruisingSpeed(0)
            self.car.setSteeringAngle(0)
            self.car.step()

        # Сброс состояния объезда
        self.avoidance_state = "NONE"
        self.avoidance_timer = 0

        self.episode_reward = 0
        self.step_count = 0
        self.is_resetting = False

    def run(self):
        print("\n🚗 СТАРТ\n")

        # Первичная стабилизация
        for _ in range(30):
            self.car.setCruisingSpeed(0)
            self.car.setSteeringAngle(0)
            self.car.step()

        episode_active = False

        while self.car.step() != -1:
            if self.is_resetting:
                continue

            if not episode_active:
                print(f"\n🎬 ЭПИЗОД {self.episode_count + 1}")
                episode_active = True

            self.step_count += 1

            # === СБОР ДАННЫХ ===
            current_pos = self._get_position()
            goal_reached = self._check_goal_reached()

            # Угол до цели
            angle_to_target = self._get_angle_to_target()
            angle_deg = math.degrees(angle_to_target)

            # Информация о препятствии
            obs_distance, obs_angle = self._get_obstacle_relative_position()

            # Данные с лидаров
            front_dist = self._get_lidar_min(self.lidar_front)

            # === ПРОВЕРКА ===
            collision = False
            if not goal_reached:
                collision = self._check_collision()

            boundary = abs(current_pos[0]) > 14 or abs(current_pos[2]) > 14

            # === НАГРАДА ===
            reward = -0.1

            # Поощрение за движение к цели
            if not collision and not goal_reached:
                # Бонус за приближение к цели
                target_pos = self.target_node.getPosition()
                dist_to_target = math.sqrt(
                    (current_pos[0] - target_pos[0]) ** 2 +
                    (current_pos[2] - target_pos[2]) ** 2
                )
                reward += 0.01 * (20 - min(dist_to_target, 20))

                # Бонус за безопасное вождение
                if front_dist > 5.0:
                    reward += 0.05

            if goal_reached:
                reward += 100
                print(f"\n🎯 ЦЕЛЬ ДОСТИГНУТА!\n")
            elif collision:
                reward += -50
                print(f"\n💥 СТОЛКНОВЕНИЕ!\n")

            self.episode_reward += reward

            # === УПРАВЛЕНИЕ ===
            if goal_reached:
                self.car.setCruisingSpeed(0)
                self.car.setSteeringAngle(0)
                mode = "🎯 СТОП"
            else:
                # Проверяем, нужно ли объезжать препятствие
                if obs_distance and obs_distance < self.OBSTACLE_AVOID_DISTANCE and abs(obs_angle) < 1.2:
                    # Рассчитываем угол для объезда
                    avoid_steer = self._calculate_avoidance_path(obs_distance, obs_angle)

                    if avoid_steer is not None:
                        steer = avoid_steer
                        speed = 15  # Немного снижаем скорость при объезде
                        mode = f"🚧 ОБЪЕЗД {self.avoidance_state} ({obs_distance:.1f}м)"
                    else:
                        # Если не в режиме объезда, но препятствие близко - готовимся
                        steer = np.clip(angle_to_target * 0.3, -0.3, 0.3)
                        speed = 20
                        mode = f"⚠️ ПРИБЛИЖЕНИЕ {obs_distance:.1f}м"
                else:
                    # Нет препятствий - едем к цели
                    steer = np.clip(angle_to_target * 0.5, -0.5, 0.5)
                    speed = 20
                    mode = f"➡️ К ЦЕЛИ {angle_deg:.0f}°"

                    # Сбрасываем состояние объезда
                    self.avoidance_state = "NONE"

                self.car.setSteeringAngle(steer)
                self.car.setCruisingSpeed(speed)

            # === ЛОГ ===
            if self.step_count % 50 == 0 or collision or goal_reached:
                print(f"[{self.step_count:4d}] {mode} | 🎯 {angle_deg:6.1f}° | 📏 {front_dist:.1f}м")

            # === ЗАВЕРШЕНИЕ ===
            reason = None
            if goal_reached:
                reason = "goal"
            elif collision:
                reason = "collision"
            elif boundary:
                reason = "boundary"
                print(f"\n🚧 ГРАНИЦА!\n")
            elif self.step_count >= self.max_steps:
                reason = "max_steps"
                print(f"\n⏱️ ТАЙМАУТ!\n")

            if reason:
                self.is_resetting = True
                self._reset_episode(reason)
                episode_active = False


def main():
    controller = BMWX5Controller()
    try:
        controller.run()
    except KeyboardInterrupt:
        print("\n🛑 Стоп")
    finally:
        print(f"\n📊 Успех: {controller.success_count}/{controller.episode_count}")
        if controller.episode_count > 0:
            success_rate = (controller.success_count / controller.episode_count) * 100
            print(f"📈 Успешность: {success_rate:.1f}%")


if __name__ == "__main__":
    main()