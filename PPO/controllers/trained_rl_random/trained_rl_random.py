#!/usr/bin/env python3
"""
Trained RL Controller для Webots с рандомной генерацией
Запускает обученную модель с циклическим респауном и случайными позициями
"""

from vehicle import Car
import numpy as np
import math
import os
import sys
import random  # Добавляем random для генерации
from collections import deque

# Stable-Baselines3
from stable_baselines3 import PPO


class TrainedBMWController:
    """
    Контроллер для запуска обученной модели с рандомными сценариями
    """

    def __init__(self):
        print("=" * 60)
        print("🚗 BMW X5 — ЗАПУСК ОБУЧЕННОЙ МОДЕЛИ (РАНДОМНЫЕ СЦЕНАРИИ)")
        print("=" * 60)

        # Инициализация Webots
        self.car = Car()
        self.timestep = int(self.car.getBasicTimeStep())

        # Поиск нод
        print("\n🔍 Поиск нод в мире:")
        self.car_node = self.car.getFromDef("CAR")
        if self.car_node:
            print(f"   CAR: найдена")
        else:
            print(f"  ️ CAR: не найдена, использую getSelf()")
            self.car_node = self.car.getSelf()

        self.obstacle_node = self.car.getFromDef("OBSTACLE")
        if self.obstacle_node:
            print(f"   OBSTACLE: найдена")
        else:
            print(f"  ️ OBSTACLE: не найдена")

        self.target_node = self.car.getFromDef("TARGET")
        self.use_virtual_target = False

        if self.target_node:
            target_pos = self.target_node.getPosition()
            print(f"   TARGET: найдена на ({target_pos[0]:.1f}, {target_pos[1]:.1f}, {target_pos[2]:.1f})")
            # Делаем цель призрачной
            try:
                self.target_node.removeBoundingObject()
                self.target_node.enableContactPointsTracking(False)
                self.target_node.resetPhysics()
                print("      Физика цели отключена")
            except Exception as e:
                print(f"     ️ Не удалось отключить физику: {e}")
        else:
            print(f"  ️ TARGET: не найдена, использую виртуальную цель")
            self.use_virtual_target = True
            self.virtual_target_pos = [10, 0.5, 0.4]

        print()

        # Параметры среды
        self.START_POS = [-11.5, 0.5, 0]
        self.START_ROT = [0, 1, 0, 0]
        self.GOAL_DISTANCE = 4.0
        self.COLLISION_DISTANCE = 0.5

        # === Параметры рандомизации ===
        self.randomize_scenarios = True

        # Диапазоны для рандомизации
        self.OBSTACLE_X_RANGE = (-8, 8)  # X: вправо-влево
        self.OBSTACLE_Y_RANGE = (-8, 8)  # Y: вперёд-назад
        self.OBSTACLE_Z = 1.0  # Z: высота (фикс)

        self.TARGET_X_RANGE = (-12, 12)  # X: вправо-влево
        self.TARGET_Y_RANGE = (-10, 10)  # Y: вперёд-назад
        self.TARGET_Z = 0.5  # Z: высота (фикс)

        # Минимальные расстояния
        self.MIN_DIST_FROM_START = 3.0  # Минимум от старта
        self.MIN_DIST_BETWEEN_OBJ = 3.0  # Минимум между целью и препятствием

        # Статистика сценариев
        self.scenario_stats = {
            'easy': 0,
            'medium': 0,
            'hard': 0
        }

        # Инициализация сенсоров
        self._setup_sensors()

        # Загрузка модели
        self.model = self._load_model()

        # Статистика
        self.episode_count = 0
        self.success_count = 0
        self.collision_count = 0
        self.total_steps = 0

        # Для observation
        self.prev_action = np.array([0.0, 0.0], dtype=np.float32)
        self.prev_dist_to_target = float('inf')

        print(f"\n Goal distance: {self.GOAL_DISTANCE}m")
        print(f" Collision distance: {self.COLLISION_DISTANCE}m")
        print(f" Randomization: {'ON' if self.randomize_scenarios else 'OFF'}")
        if self.randomize_scenarios:
            print(f"   Obstacle X: {self.OBSTACLE_X_RANGE}")
            print(f"   Obstacle Y: {self.OBSTACLE_Y_RANGE}")
            print(f"   Obstacle Z: {self.OBSTACLE_Z}")
            print(f"   Target X: {self.TARGET_X_RANGE}")
            print(f"   Target Y: {self.TARGET_Y_RANGE}")
            print(f"   Target Z: {self.TARGET_Z}")
        print("=" * 60)
        print("\n Модель готова! Запускаем циклическое движение с рандомными сценариями...")
        print("(Ctrl+C для остановки)\n")

    def _load_model(self):
        """Загрузка обученной модели"""
        model_paths = [
            "./models/bmw_rl_reverse_final.zip",  # Модель с задним ходом
            "./models/bmw_rl_randomized_final.zip",  # Рандомная модель
            "./models/bmw_rl_balanced_final.zip",  # Балансная модель
            "./models/bmw_rl_reverse_long_final.zip",  # Долгая модель
            "./models/bmw_rl_balanced_interrupted.zip",
            "./bmw_rl_balanced_final.zip",
            "./bmw_rl_reverse_final.zip"
        ]

        for path in model_paths:
            if os.path.exists(path):
                print(f" Загрузка модели: {path}")
                try:
                    model = PPO.load(path)
                    print(" Модель успешно загружена!")
                    return model
                except Exception as e:
                    print(f"   Ошибка загрузки: {e}")

        print(" Модель не найдена! Создайте папку models и поместите туда обученную модель.")
        print("   Ожидаемые файлы:")
        for p in model_paths:
            print(f"     - {p}")
        sys.exit(1)

    def _setup_sensors(self):
        """Инициализация сенсоров"""
        self.lidar_front = self.car.getDevice("lidar_front")
        self.lidar_left = self.car.getDevice("lidar_left")
        self.lidar_right = self.car.getDevice("lidar_right")
        self.gps = self.car.getDevice("gps")
        self.imu = self.car.getDevice("imu")

        for device in [self.lidar_front, self.lidar_left, self.lidar_right, self.gps, self.imu]:
            if device:
                device.enable(self.timestep)
                print(f"  ✅ {device.getName()}")

    # === НОВЫЙ МЕТОД: Генерация случайной позиции ===
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

        # Запасной вариант
        print(f" Не удалось найти позицию, использую значение по умолчанию")
        if for_target:
            return [10, 0.5, 0.4]
        else:
            return [0, 0, 1]

    # === Рандомизация сценария ===
    def _randomize_scenario(self):
        """Создание нового случайного сценария"""
        if not self.randomize_scenarios:
            return

        print("\n  🎲 Генерация нового сценария...")

        # Генерируем позицию препятствия
        obs_pos = self._generate_random_position(for_target=False, exclude_positions=None)

        # Генерируем позицию цели (избегаем близости к препятствию)
        target_pos = self._generate_random_position(for_target=True, exclude_positions=[obs_pos])

        # Перемещаем препятствие
        if self.obstacle_node:
            try:
                self.obstacle_node.getField("translation").setSFVec3f(obs_pos)
                print(f"     Препятствие: X={obs_pos[0]:5.1f}, Y={obs_pos[1]:5.1f}, Z={obs_pos[2]:.1f}")
            except Exception as e:
                print(f"     Ошибка перемещения препятствия: {e}")

        # Перемещаем цель
        if self.target_node:
            try:
                self.target_node.getField("translation").setSFVec3f(target_pos)
                print(f"     Цель:        X={target_pos[0]:5.1f}, Y={target_pos[1]:5.1f}, Z={target_pos[2]:.1f}")
            except Exception as e:
                print(f"     Ошибка перемещения цели: {e}")
        elif self.use_virtual_target:
            self.virtual_target_pos = target_pos
            print(f"     Виртуальная цель: ({target_pos[0]:.1f}, {target_pos[1]:.1f})")

        # Определяем сложность сценария
        dist_obs_target = math.sqrt(
            (obs_pos[0] - target_pos[0]) ** 2 +
            (obs_pos[1] - target_pos[1]) ** 2
        )

        if dist_obs_target > 10:
            difficulty = "easy"
            difficulty_emoji = "🟢"
        elif dist_obs_target > 5:
            difficulty = "medium"
            difficulty_emoji = "🟡"
        else:
            difficulty = "hard"
            difficulty_emoji = "🔴"

        self.scenario_stats[difficulty] += 1
        print(
            f"     Сложность: {difficulty_emoji} {difficulty.upper()} (дист. цель-препятствие: {dist_obs_target:.1f}м)")

    def _sanitize_value(self, value, default=0.0, min_val=-1e6, max_val=1e6):
        """Защита от NaN и inf"""
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
        """Получение позиции"""
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
        """Получение угла поворота"""
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
        """Угол до цели"""
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
        """Проверка столкновения по лидарам"""
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
        """Формирование наблюдения для модели"""
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
                    current_speed = math.sqrt(vel[0] ** 2 + vel[2] ** 2)
                    current_speed = self._sanitize_value(current_speed, 0, 0, 100)
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

    def reset_car(self):
        """Сброс машины на исходную позицию"""
        print(f"\n Сброс позиции...")

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
                print(f"  Ошибка телепортации: {e}")

        for _ in range(20):
            self.car.setCruisingSpeed(0)
            self.car.setSteeringAngle(0)
            self.car.step()

        # === Рандомизация сценария при сбросе ===
        if self.randomize_scenarios:
            self._randomize_scenario()

        self.prev_action = np.array([0.0, 0.0], dtype=np.float32)
        self.prev_dist_to_target = self._check_goal_3d()[1]

        print("   ✅ Готово к следующему заезду!\n")

    def run_episode(self):
        """Один эпизод движения"""
        self.episode_count += 1
        episode_steps = 0
        episode_reward = 0.0

        # Определяем эмодзи сложности
        if hasattr(self, 'scenario_stats'):
            total = sum(self.scenario_stats.values())
            if total > 0:
                easy_pct = (self.scenario_stats['easy'] / total) * 100
                medium_pct = (self.scenario_stats['medium'] / total) * 100
                hard_pct = (self.scenario_stats['hard'] / total) * 100
                print(f"\nСтатистика сложности: 🟢{easy_pct:.0f}% 🟡{medium_pct:.0f}% 🔴{hard_pct:.0f}%")

        print(f"Эпизод {self.episode_count} начался!")

        # Начальное наблюдение
        obs = self._get_observation()

        while self.car.step() != -1:
            # Получаем действие от модели
            action, _ = self.model.predict(obs, deterministic=True)

            # Применяем действие
            steering = float(np.clip(action[0], -0.8, 0.8))
            speed = float(np.clip(action[1], -5.0, 20.0))  # Отрицательная скорость = задний ход

            self.prev_action = np.array([steering, speed], dtype=np.float32)
            self.car.setSteeringAngle(steering)
            self.car.setCruisingSpeed(speed)

            # Новое наблюдение
            obs = self._get_observation()
            episode_steps += 1
            self.total_steps += 1

            # Проверка условий
            is_goal, dist_to_target = self._check_goal_3d()
            is_collision, collision_info, min_lidar = self._check_collision_lidar()

            # Логирование каждые 100 шагов
            if episode_steps % 100 == 0:
                direction = "ВПЕРЁД" if speed >= 0 else "НАЗАД"
                print(f"   Step {episode_steps:4d}: goal={dist_to_target:5.2f}m, lidar={min_lidar:4.2f}m, {direction}")

            # Проверка цели
            if is_goal:
                self.success_count += 1
                print(f"\nGOAL! Цель достигнута на {episode_steps} шаге!")
                print(f"   Расстояние: {dist_to_target:.2f}m")
                return True

            # Проверка столкновения
            if is_collision:
                self.collision_count += 1
                print(f"\nCOLLISION! {collision_info}")
                return False

            # Таймаут
            if episode_steps >= 2500:  # Увеличено для более сложных сценариев
                print(f"\nTIMEOUT! Слишком долго...")
                return False

        return False

    def print_stats(self):
        """Вывод статистики"""
        print(f"\n{'=' * 60}")
        print("СТАТИСТИКА")
        print(f"{'=' * 60}")
        print(f"Всего эпизодов: {self.episode_count}")
        print(f"Успехов (цель): {self.success_count}")
        print(f"Столкновений:   {self.collision_count}")
        print(f"Всего шагов:    {self.total_steps}")
        if self.episode_count > 0:
            success_rate = (self.success_count / self.episode_count) * 100
            print(f"Успешность:     {success_rate:.1f}%")

        # Статистика сценариев
        if self.randomize_scenarios:
            total = sum(self.scenario_stats.values())
            print(f"\nСТАТИСТИКА СЦЕНАРИЕВ:")
            for diff, count in self.scenario_stats.items():
                percentage = (count / total) * 100 if total > 0 else 0
                emoji = "🟢" if diff == "easy" else "🟡" if diff == "medium" else "🔴"
                print(f"  {emoji} {diff.upper()}: {count} ({percentage:.1f}%)")
        print(f"{'=' * 60}\n")

    def run(self):
        """Главный цикл"""
        try:
            while True:
                # Запускаем эпизод
                success = self.run_episode()

                # Выводим результат
                if success:
                    print("Отличная работа модели!")
                else:
                    print("Пробуем снова...")

                # Статистика
                self.print_stats()

                # Респаун с новым случайным сценарием
                self.reset_car()

        except KeyboardInterrupt:
            print("\n\nОстановлено пользователем")
            self.print_stats()
            print("До свидания!")


def main():
    controller = TrainedBMWController()
    controller.run()


if __name__ == "__main__":
    main()