#!/usr/bin/env python3
"""
TRAINED RL CONTROLLER ДЛЯ BMW X5 В WEBOTS
==========================================
Данный файл запускает обученную модель для тестирования в среде
с рандомной генерацией нескольких препятствий и цели.

Особенности:
- Автоматическая загрузка лучшей обученной модели
- Рандомная генерация нескольких препятствий в каждом эпизоде
- Поддержка заднего хода (отрицательные скорости)
- Детальная статистика производительности
- Циклический респаун для непрерывного тестирования

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

from stable_baselines3 import PPO


class TrainedBMWController:
    """
    Контроллер для запуска обученной модели с рандомными сценариями.

    Этот класс загружает предварительно обученную модель PPO и использует её
    для управления автомобилем в среде с рандомно генерируемыми препятствиями.
    После каждого эпизода (успех, столкновение или таймаут) автомобиль
    возвращается на старт, и генерируется новый сценарий.
    """

    def __init__(self):
        """
        Инициализация контроллера: подключение к Webots, поиск нод,
        настройка сенсоров, загрузка модели и инициализация параметров.
        """
        print("=" * 60)
        print("ЗАПУСК ОБУЧЕННОЙ МОДЕЛИ (НЕСКОЛЬКО ПРЕПЯТСТВИЙ)")
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

        # Поиск автомобиля
        self.car_node = self.car.getFromDef("CAR")
        if self.car_node:
            print(f"  ✅ CAR: найдена")
        else:
            print(f"  ⚠️ CAR: не найдена, использую getSelf()")
            self.car_node = self.car.getSelf()

        # Поиск препятствия (опционально, может не существовать)
        self.obstacle_node = self.car.getFromDef("OBSTACLE")
        if self.obstacle_node:
            print(f"  ✅ OBSTACLE: найдена")
        else:
            print(f"  ⚠️ OBSTACLE: не найдена (будут созданы программно)")

        # Поиск цели
        self.target_node = self.car.getFromDef("TARGET")
        self.use_virtual_target = False

        if self.target_node:
            target_pos = self.target_node.getPosition()
            print(f"  ✅ TARGET: найдена на ({target_pos[0]:.1f}, {target_pos[1]:.1f}, {target_pos[2]:.1f})")
            # Делаем цель призрачной (без физики)
            try:
                self.target_node.removeBoundingObject()
                self.target_node.enableContactPointsTracking(False)
                self.target_node.resetPhysics()
                print("     👻 Физика цели отключена")
            except Exception as e:
                print(f"     ⚠️ Не удалось отключить физику: {e}")
        else:
            print(f"  ⚠️ TARGET: не найдена, использую виртуальную цель")
            self.use_virtual_target = True
            self.virtual_target_pos = [10, 0.5, 0.4]

        print()

        # =====================================================================
        # ПАРАМЕТРЫ СРЕДЫ
        # =====================================================================
        # Начальная позиция автомобиля
        self.START_POS = [-11.5, 0.5, 0.5]

        # Начальная ориентация
        self.START_ROT = [0, 1, 0, 0]

        # Расстояние до цели, считающееся успехом
        self.GOAL_DISTANCE = 4.0

        # Расстояние до препятствия, считающееся столкновением
        self.COLLISION_DISTANCE = 0.5

        # Максимальное количество шагов в эпизоде (таймаут)
        self.MAX_STEPS = 2500

        # =====================================================================
        # ПАРАМЕТРЫ РАНДОМИЗАЦИИ
        # =====================================================================
        # Включение/выключение рандомизации сценариев
        self.randomize_scenarios = True

        # Количество препятствий для генерации
        self.num_obstacles = 3

        # Диапазоны для рандомизации положения препятствий
        self.OBSTACLE_X_RANGE = (-8, 8)  # По оси X (вправо-влево)
        self.OBSTACLE_Y_RANGE = (-8, 8)  # По оси Y (вперёд-назад)
        self.OBSTACLE_Z = 1.0  # Высота препятствий (фиксирована)

        # Диапазон размеров препятствий (для разнообразия)
        self.OBSTACLE_SIZE_RANGE = (0.8, 2.0)

        # Диапазоны для рандомизации положения цели
        self.TARGET_X_RANGE = (-12, 12)  # По оси X
        self.TARGET_Y_RANGE = (-10, 10)  # По оси Y
        self.TARGET_Z = 0.5  # Высота цели

        # Минимальные расстояния для валидной генерации
        self.MIN_DIST_FROM_START = 6.0  # От старта до объектов
        self.MIN_DIST_BETWEEN_OBJ = 3.0  # Между объектами

        # =====================================================================
        # ХРАНЕНИЕ СОЗДАННЫХ ПРЕПЯТСТВИЙ
        # =====================================================================
        self.obstacle_nodes = []  # Список созданных препятствий
        self.obstacle_sizes = []  # Список их размеров

        # =====================================================================
        # СТАТИСТИКА СЦЕНАРИЕВ
        # =====================================================================
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
        # ЗАГРУЗКА МОДЕЛИ
        # =====================================================================
        self.model = self._load_model()

        # =====================================================================
        # СТАТИСТИКА ТЕСТИРОВАНИЯ
        # =====================================================================
        self.episode_count = 0
        self.success_count = 0
        self.collision_count = 0
        self.total_steps = 0

        # Для формирования наблюдений
        self.prev_action = np.array([0.0, 0.0], dtype=np.float32)
        self.prev_dist_to_target = float('inf')

        # Для отслеживания прогресса (опционально)
        self.steps_without_progress = 0
        self.action_history = deque(maxlen=50)

        # =====================================================================
        # ВЫВОД ИНФОРМАЦИИ
        # =====================================================================
        print(f"\n✅ Goal distance: {self.GOAL_DISTANCE}m")
        print(f"✅ Collision distance: {self.COLLISION_DISTANCE}m")
        print(f"✅ Max steps: {self.MAX_STEPS}")
        print(f"✅ Randomization: {'ON' if self.randomize_scenarios else 'OFF'}")
        print(f"✅ Number of obstacles: {self.num_obstacles}")
        print("=" * 60)
        print("\n🎮 Модель готова! Запускаем тестирование...")
        print("   Каждый эпизод - новая случайная конфигурация препятствий")
        print("   (Ctrl+C для остановки)\n")

    # =========================================================================
    # МЕТОДЫ ДЛЯ РАБОТЫ С МОДЕЛЬЮ
    # =========================================================================

    def _load_model(self):
        """
        Загрузка обученной модели из файла.

        Пытается загрузить модель по нескольким возможным путям.
        Приоритет отдаётся моделям с задним ходом и несколькими препятствиями.

        Returns:
            model: загруженная модель PPO

        Raises:
            SystemExit: если модель не найдена
        """
        # Список возможных путей к моделям (в порядке приоритета)
        model_paths = [
            "bmw_curriculum_final.zip"
        ]

        for path in model_paths:
            if os.path.exists(path):
                print(f"📦 Загрузка модели: {path}")
                try:
                    model = PPO.load(path)
                    print("   ✅ Модель успешно загружена!")
                    return model
                except Exception as e:
                    print(f"   ⚠️ Ошибка загрузки: {e}")

        print("❌ Модель не найдена! Создайте папку models и поместите туда обученную модель.")
        print("   Ожидаемые файлы:")
        for p in model_paths:
            print(f"     - {p}")
        sys.exit(1)

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
                print(f"  ✅ {device.getName()}")

    def _sanitize_value(self, value, default=0.0, min_val=-1e6, max_val=1e6):
        """
        Очистка значения от NaN и бесконечности.

        Параметры:
            value: исходное значение
            default: значение по умолчанию при ошибке
            min_val: минимальное допустимое значение
            max_val: максимальное допустимое значение

        Returns:
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

        Параметры:
            lidar: объект лидара из Webots

        Returns:
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

        Returns:
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

        Yaw - поворот вокруг вертикальной оси:
        0 = направление вдоль оси X
        π/2 = направление вдоль оси Y
        π = направление назад

        Returns:
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

        Returns:
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

        Угол 0 = автомобиль смотрит прямо на цель.
        Положительный угол = цель справа.
        Отрицательный угол = цель слева.

        Returns:
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

        Returns:
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

        Returns:
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
        Формирование вектора наблюдения для модели.

        Наблюдение содержит 12 параметров, совместимых с обученной моделью:
        [dx, dy, dz, yaw, angle_to_target, front, left, right, min_lidar,
         current_speed, dist_to_target, prev_steering]

        Returns:
            np.array: вектор наблюдения
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

        # Текущая скорость с учётом направления
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

        Returns:
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
            print(f"     Ошибка создания препятствия: {e}")
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

        Returns:
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
        print(f"  ⚠️ Не удалось найти позицию после {max_attempts} попыток")
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

        print("\n  🎲 Генерация нового сценария...")

        # Удаляем старые препятствия
        self._remove_all_obstacles()

        # Список для хранения позиций всех препятствий
        all_obstacle_positions = []

        # Генерируем несколько препятствий
        for i in range(self.num_obstacles):
            # Случайный размер
            size = random.uniform(*self.OBSTACLE_SIZE_RANGE)

            # Генерируем позицию с учётом уже созданных препятствий
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
                print(f"     Цель:        X={target_pos[0]:5.1f}, Y={target_pos[1]:5.1f}, Z={target_pos[2]:.1f}")
            except Exception as e:
                print(f"     Ошибка перемещения цели: {e}")
        elif self.use_virtual_target:
            self.virtual_target_pos = target_pos
            print(f"     Виртуальная цель: ({target_pos[0]:.1f}, {target_pos[1]:.1f})")

        # Определяем сложность сценария (минимальное расстояние до цели)
        min_dist_to_target = float('inf')
        for obs_pos in all_obstacle_positions:
            dist = math.sqrt(
                (obs_pos[0] - target_pos[0]) ** 2 +
                (obs_pos[1] - target_pos[1]) ** 2
            )
            min_dist_to_target = min(min_dist_to_target, dist)

        if min_dist_to_target > 8:
            difficulty = "easy"
            emoji = "🟢"
        elif min_dist_to_target > 4:
            difficulty = "medium"
            emoji = "🟡"
        else:
            difficulty = "hard"
            emoji = "🔴"

        self.scenario_stats[difficulty] += 1
        print(f"     Сложность: {emoji} {difficulty.upper()} (мин. дист. до цели: {min_dist_to_target:.1f}м)")

    # =========================================================================
    # ОСНОВНЫЕ МЕТОДЫ УПРАВЛЕНИЯ
    # =========================================================================

    def reset_car(self):
        """
        Сброс автомобиля на стартовую позицию и генерация нового сценария.

        Телепортирует машину на старт, останавливает её,
        генерирует новые препятствия и цель.
        """
        print(f"\n🔄 Сброс позиции...")

        # Останавливаем машину
        self.car.setCruisingSpeed(0)
        self.car.setSteeringAngle(0)

        # Сброс физики
        self.car.simulationResetPhysics()
        for _ in range(5):
            self.car.step()

        # Телепортация на старт
        if self.car_node:
            try:
                # Сначала поворот
                self.car_node.getField("rotation").setSFRotation(self.START_ROT)
                for _ in range(3):
                    self.car.step()
                # Затем позиция
                self.car_node.getField("translation").setSFVec3f(self.START_POS)
                for _ in range(10):
                    self.car.step()
            except Exception as e:
                print(f"  ⚠️ Ошибка телепортации: {e}")

        # Стабилизация после телепортации
        for _ in range(20):
            self.car.setCruisingSpeed(0)
            self.car.setSteeringAngle(0)
            self.car.step()

        # Генерация нового случайного сценария
        if self.randomize_scenarios:
            self._randomize_scenario()

        # Сброс внутренних переменных
        self.prev_action = np.array([0.0, 0.0], dtype=np.float32)
        self.prev_dist_to_target = self._check_goal_3d()[1]
        self.steps_without_progress = 0
        self.action_history.clear()

        print("   ✅ Готово к следующему заезду!\n")

    def run_episode(self):
        """
        Запуск одного эпизода тестирования.

        Агент управляет автомобилем, пока не будет достигнута цель,
        не произойдёт столкновение или не истечёт таймаут.

        Returns:
            bool: True если цель достигнута, False в противном случае
        """
        self.episode_count += 1
        episode_steps = 0
        episode_reward = 0.0  # Для информации, модель не обучается

        # Статистика сложности
        if hasattr(self, 'scenario_stats'):
            total = sum(self.scenario_stats.values())
            if total > 0:
                easy_pct = (self.scenario_stats['easy'] / total) * 100
                medium_pct = (self.scenario_stats['medium'] / total) * 100
                hard_pct = (self.scenario_stats['hard'] / total) * 100
                print(f"\n📊 Статистика сложности: 🟢{easy_pct:.0f}% 🟡{medium_pct:.0f}% 🔴{hard_pct:.0f}%")

        print(f"🏁 Эпизод {self.episode_count} начался!")

        # Начальное наблюдение
        obs = self._get_observation()

        while self.car.step() != -1:
            # Получаем действие от модели (детерминированное для тестирования)
            action, _ = self.model.predict(obs, deterministic=True)

            # Применяем действие
            steering = float(np.clip(action[0], -0.8, 0.8))
            speed = float(np.clip(action[1], -5.0, 20.0))  # Поддержка заднего хода

            self.prev_action = np.array([steering, speed], dtype=np.float32)
            self.car.setSteeringAngle(steering)
            self.car.setCruisingSpeed(speed)

            # Шаг симуляции
            self.car.step()

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
                print(f"\n🎯 GOAL! Цель достигнута на {episode_steps} шаге!")
                print(f"   Расстояние: {dist_to_target:.2f}m")
                return True

            # Проверка столкновения
            if is_collision:
                self.collision_count += 1
                print(f"\n💥 COLLISION! {collision_info}")
                return False

            # Проверка таймаута
            if episode_steps >= self.MAX_STEPS:
                print(f"\n⏱️ TIMEOUT! Слишком долго...")
                return False

        return False

    def print_stats(self):
        """
        Вывод подробной статистики тестирования.
        """
        print(f"\n{'=' * 60}")
        print("📊 СТАТИСТИКА ТЕСТИРОВАНИЯ")
        print(f"{'=' * 60}")
        print(f"Всего эпизодов:    {self.episode_count}")
        print(f"Успехов (цель):    {self.success_count}")
        print(f"Столкновений:      {self.collision_count}")
        print(f"Всего шагов:       {self.total_steps}")

        if self.episode_count > 0:
            success_rate = (self.success_count / self.episode_count) * 100
            collision_rate = (self.collision_count / self.episode_count) * 100
            timeout_rate = 100 - success_rate - collision_rate
            print(f"\n📈 Процент успехов:     {success_rate:.1f}%")
            print(f"💥 Процент столкновений: {collision_rate:.1f}%")
            print(f"⏱️ Процент таймаутов:    {timeout_rate:.1f}%")

        # Статистика сценариев
        if self.randomize_scenarios:
            total = sum(self.scenario_stats.values())
            print(f"\n🎲 СТАТИСТИКА СЦЕНАРИЕВ:")
            for diff, count in self.scenario_stats.items():
                percentage = (count / total) * 100 if total > 0 else 0
                emoji = "🟢" if diff == "easy" else "🟡" if diff == "medium" else "🔴"
                print(f"  {emoji} {diff.upper()}: {count} ({percentage:.1f}%)")
        print(f"{'=' * 60}\n")

    def run(self):
        """
        Главный цикл тестирования.

        Бесконечно запускает эпизоды, пока пользователь не прервёт выполнение.
        """
        try:
            while True:
                # Запуск эпизода
                success = self.run_episode()

                # Вывод результата
                if success:
                    print("✨ Отличная работа модели!")
                else:
                    print("💪 Пробуем снова...")

                # Статистика
                self.print_stats()

                # Сброс для следующего эпизода
                self.reset_car()

        except KeyboardInterrupt:
            print("\n\n🛑 Остановлено пользователем")
            self.print_stats()
            print("👋 До свидания!")


def main():
    """
    Точка входа в программу.
    """
    controller = TrainedBMWController()
    controller.run()


if __name__ == "__main__":
    main()