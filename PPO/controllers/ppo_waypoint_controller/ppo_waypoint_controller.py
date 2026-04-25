"""
Основной контроллер для PPO обучения с глобальным планированием маршрута.
Управляет автомобилем, строит маршрут и обучает нейросеть.
"""

from vehicle import Driver
import math
import numpy as np
import os
import json
import socket
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ppo_agent import PPOAgent
from path_planner import PathPlanner

MAX_SPEED = 15.0
MAX_STEER = 0.5
WAYPOINT_REACHED_DISTANCE = 5.0
FINISH_REACHED_DISTANCE = 7.0
RESET_STEPS = 50

# =============================================================================
# ПАРАМЕТРЫ СНЕГА
# =============================================================================
SNOW_SIZE = 100
KOEF_SNOW = 0.85
KOEF_SNOW_SLIP = 0.8
SNOW_TRACK_STRENGTH = 0.0002
TRACK_BONUS_MAX = 1.0

# UDP сокет для связи с супервизором
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
SUPERVISOR_ADDR = ("127.0.0.1", 6006)
sock.setblocking(False)

driver = Driver()
timestep = int(driver.getBasicTimeStep())

# Инициализация датчиков
gps = driver.getDevice("gps")
gps.enable(timestep)
imu = driver.getDevice("imu")
imu.enable(timestep)
compass = driver.getDevice("compass")
compass.enable(timestep)
lidar_front = driver.getDevice("lidar_front")
lidar_front.enable(timestep)
lidar_left = driver.getDevice("lidar_left")
lidar_left.enable(timestep)
lidar_right = driver.getDevice("lidar_right")
lidar_right.enable(timestep)
lidar_back = driver.getDevice("lidar_back")
lidar_back.enable(timestep)

os.makedirs('logs_ppo', exist_ok=True)
os.makedirs('models', exist_ok=True)


def get_finish_position():
    """
    Получение актуальной позиции финиша из файла.

    Returns:
        tuple: (x, y) координаты финиша или None
    """
    try:
        if os.path.exists("finish_pos.txt"):
            with open("finish_pos.txt", "r") as f:
                content = f.read().strip()
                parts = content.split()
                if len(parts) >= 2:
                    x = float(parts[0])
                    y = float(parts[1])
                    print(f"Finish loaded from file: ({x:.1f}, {y:.1f})")
                    return (x, y)
    except Exception as e:
        print(f"Error reading finish_pos.txt: {e}")

    # Попытка найти файл в директории car_reset
    try:
        controller_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(controller_dir)
        car_reset_path = os.path.join(parent_dir, "car_reset", "finish_pos.txt")
        if os.path.exists(car_reset_path):
            with open(car_reset_path, "r") as f:
                content = f.read().strip()
                parts = content.split()
                if len(parts) >= 2:
                    x = float(parts[0])
                    y = float(parts[1])
                    print(f"Finish loaded from car_reset: ({x:.1f}, {y:.1f})")
                    return (x, y)
    except:
        pass

    return None


def load_snow_positions():
    """Загрузка позиций снежных зон из файла."""
    snow_positions = []
    try:
        paths = ["snow.txt", "../car_reset/snow.txt", "../../controllers/car_reset/snow.txt"]
        for path in paths:
            if os.path.exists(path):
                with open(path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 3:
                            snow_positions.append((
                                float(parts[0]), float(parts[1]), float(parts[2])
                            ))
                print(f"Loaded {len(snow_positions)} snow zones from {path}")
                break
    except Exception as e:
        print(f"Error loading snow: {e}")
    return snow_positions


def get_half_size(z):
    """Расчёт размера снежной зоны в зависимости от глубины."""
    z = max(-1, min(0, z))
    return 50 - (SNOW_SIZE / 2 * (z / -1))


print("=" * 60)
print("PPO Waypoint Navigator with Snow Effect")
print("=" * 60)


class SnowManager:
    """Управление эффектами снега и колеёй."""

    def __init__(self):
        self.snow_positions = []
        self.track_maps = []
        self.speed_multiplier = 1.0
        self.load_snow_data()

    def load_snow_data(self):
        """Загрузка данных о снежных зонах."""
        self.snow_positions = load_snow_positions()
        self.track_maps = []
        for _ in self.snow_positions:
            self.track_maps.append([[0.0 for _ in range(SNOW_SIZE)] for _ in range(SNOW_SIZE)])
        print(f"Snow manager initialized with {len(self.snow_positions)} zones")

    def update_track(self, car_x, car_y, snow_id, speed):
        """Обновление карты колеи."""
        if snow_id >= len(self.track_maps):
            return

        radius = 2
        track_map = self.track_maps[snow_id]
        sx, sy, sz = self.snow_positions[snow_id]
        half_size = get_half_size(sz)

        if half_size < 1:
            return

        local_x = int((car_x - (sx - half_size)) / (half_size * 2 / SNOW_SIZE))
        local_y = int((car_y - (sy - half_size)) / (half_size * 2 / SNOW_SIZE))

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                ni, nj = local_x + dx, local_y + dy
                if 0 <= ni < SNOW_SIZE and 0 <= nj < SNOW_SIZE:
                    if abs(speed) > 1:
                        track_map[ni][nj] += SNOW_TRACK_STRENGTH
                        track_map[ni][nj] = min(TRACK_BONUS_MAX, track_map[ni][nj])
                    else:
                        track_map[ni][nj] -= SNOW_TRACK_STRENGTH / 2
                        track_map[ni][nj] = max(0.0, track_map[ni][nj])

    def get_snow_effect(self, car_x, car_y):
        """
        Получение эффекта снега в текущей позиции.

        Returns:
            tuple: (speed_multiplier, in_snow, snow_id)
        """
        self.speed_multiplier = 1.0
        in_snow = False
        current_snow_id = -1

        for i, (sx, sy, sz) in enumerate(self.snow_positions):
            half_size = get_half_size(sz)
            if half_size < 1:
                continue

            if (sx - half_size <= car_x <= sx + half_size and
                sy - half_size <= car_y <= sy + half_size):

                in_snow = True
                current_snow_id = i

                distance = math.sqrt((car_x - sx)**2 + (car_y - sy)**2)
                max_dist = half_size

                # Базовый коэффициент замедления от снега
                snow_factor = KOEF_SNOW + 0.092 * distance / max_dist
                snow_factor = min(snow_factor, 1.0)

                # Бонус от колеи
                local_x = int((car_x - (sx - half_size)) / (half_size * 2 / SNOW_SIZE))
                local_y = int((car_y - (sy - half_size)) / (half_size * 2 / SNOW_SIZE))

                if 0 <= local_x < SNOW_SIZE and 0 <= local_y < SNOW_SIZE:
                    track_value = self.track_maps[i][local_x][local_y]
                    track_bonus = 1.0 + track_value * 0.5
                else:
                    track_bonus = 1.0

                self.speed_multiplier = snow_factor * track_bonus
                break

        return self.speed_multiplier, in_snow, current_snow_id

    def reset_tracks(self):
        """Сброс карт колеи."""
        self.track_maps = []
        for _ in self.snow_positions:
            self.track_maps.append([[0.0 for _ in range(SNOW_SIZE)] for _ in range(SNOW_SIZE)])
        print("Snow tracks reset")


class PPOWaypointNavigator:
    """Основной класс навигатора с PPO обучением."""

    def __init__(self):
        self.agent = PPOAgent()
        self.path_planner = PathPlanner()
        self.snow_manager = SnowManager()

        self.current_waypoints = []
        self.current_target_idx = 0

        self.waiting_reset = True
        self.reset_steps = RESET_STEPS

        self.prev_distance_to_target = float('inf')

        self.episode = 0
        self.finishes = 0
        self.dtp_count = 0
        self.step_count = 0

        self.speed = 0.0
        self.steering_angle = 0.0

        self.prev_state = None
        self.prev_action = None
        self.episode_reward = 0.0

        self.obstacles = []
        self.load_obstacles()

        self.last_save_episode = 0
        self.last_in_snow = False

        # Загрузка начальной позиции финиша
        finish = get_finish_position()
        if finish:
            self.target_x, self.target_y = finish
        else:
            self.target_x, self.target_y = 155.157, -155.751

        print(f"Initial target: ({self.target_x:.1f}, {self.target_y:.1f})")

    def load_obstacles(self):
        """Загрузка данных о препятствиях из конфигурационного файла."""
        try:
            with open("world_data.json", "r") as f:
                data = json.load(f)
                self.obstacles = data.get("obstacles", [])
            print(f"Loaded obstacles: {len(self.obstacles)}")
        except:
            print("world_data.json not found")
            self.obstacles = []

    def send_waypoints_to_supervisor(self):
        """Отправка путевых точек в супервизор для визуализации."""
        if not self.current_waypoints:
            return

        waypoints_str = ";".join([f"{wp[0]:.2f},{wp[1]:.2f}" for wp in self.current_waypoints])
        message = f"WAYPOINTS:{waypoints_str}"

        try:
            sock.sendto(message.encode(), SUPERVISOR_ADDR)
            print(f"Sent {len(self.current_waypoints)} waypoints to supervisor")
        except Exception as e:
            print(f"Error sending waypoints: {e}")

    def plan_route(self):
        """
        Построение маршрута от текущей позиции до финиша.
        Перед построением обновляется позиция финиша из файла.
        """
        # Обновление позиции финиша
        finish = get_finish_position()
        if finish:
            self.target_x, self.target_y = finish
            print(f"Finish updated: ({self.target_x:.1f}, {self.target_y:.1f})")
        else:
            print(f"Finish not found, using previous: ({self.target_x:.1f}, {self.target_y:.1f})")

        start_pos = self.get_position()
        target_pos = (self.target_x, self.target_y)

        print(f"\nPlanning route from ({start_pos[0]:.1f}, {start_pos[1]:.1f})")
        print(f"   to finish at ({target_pos[0]:.1f}, {target_pos[1]:.1f})")

        self.current_waypoints = self.path_planner.plan_path(
            start_pos, target_pos, self.obstacles
        )
        self.current_target_idx = 0
        self.prev_distance_to_target = float('inf')

        print(f"Route built with {len(self.current_waypoints)} waypoints")

        self.send_waypoints_to_supervisor()

        for i, wp in enumerate(self.current_waypoints[:5]):
            print(f"   Waypoint {i}: ({wp[0]:.1f}, {wp[1]:.1f})")

    def get_position(self):
        """Получение текущей позиции автомобиля из GPS."""
        pos = gps.getValues()
        return (pos[0], pos[1])

    def get_heading(self):
        """Получение текущего курса автомобиля из компаса."""
        north = compass.getValues()
        heading = math.atan2(north[0], north[1])
        return heading

    def get_speed(self):
        """Получение текущей скорости автомобиля."""
        return driver.getCurrentSpeed()

    def get_lidar_state(self):
        """
        Получение и обработка данных с лидаров.

        Returns:
            tuple: (front_state, min_front, min_left, min_right)
        """
        front_data = lidar_front.getRangeImage()
        left_data = lidar_left.getRangeImage()
        right_data = lidar_right.getRangeImage()

        front_data = np.clip(front_data, 0, 10)
        left_data = np.clip(left_data, 0, 5)
        right_data = np.clip(right_data, 0, 5)

        front_state = front_data[::8]
        min_front = min(front_data) if len(front_data) > 0 else 10
        min_left = min(left_data) if len(left_data) > 0 else 5
        min_right = min(right_data) if len(right_data) > 0 else 5

        return front_state, min_front, min_left, min_right

    def get_target_info(self):
        """
        Получение информации о текущей целевой точке.

        Returns:
            tuple: (target, distance, angle_diff) или (None, None, None)
        """
        if not self.current_waypoints or self.current_target_idx >= len(self.current_waypoints):
            return None, None, None

        target = self.current_waypoints[self.current_target_idx]
        pos = self.get_position()
        heading = self.get_heading()

        dx = target[0] - pos[0]
        dy = target[1] - pos[1]
        distance = math.sqrt(dx * dx + dy * dy)

        goal_angle = math.atan2(dy, dx)
        angle_diff = goal_angle - heading
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

        return target, distance, angle_diff

    def check_waypoint_reached(self):
        """
        Проверка достижения текущей путевой точки.

        Returns:
            bool: True если точка достигнута
        """
        _, distance, _ = self.get_target_info()

        if distance is not None and distance < WAYPOINT_REACHED_DISTANCE:
            print(f"Waypoint {self.current_target_idx + 1}/{len(self.current_waypoints)} reached")
            self.current_target_idx += 1

            # Отправка оставшихся точек для обновления визуализации
            remaining_waypoints = self.current_waypoints[self.current_target_idx:]
            if remaining_waypoints:
                waypoints_str = ";".join([f"{wp[0]:.2f},{wp[1]:.2f}" for wp in remaining_waypoints])
                try:
                    sock.sendto(f"WAYPOINTS:{waypoints_str}".encode(), SUPERVISOR_ADDR)
                except:
                    pass
            else:
                try:
                    sock.sendto(b"CLEAR_WAYPOINTS", SUPERVISOR_ADDR)
                except:
                    pass

            self.prev_distance_to_target = float('inf')
            return True
        return False

    def check_finish_reached(self):
        """
        Проверка достижения финиша.

        Returns:
            bool: True если финиш достигнут
        """
        if self.current_waypoints and self.current_target_idx >= len(self.current_waypoints):
            pos = self.get_position()
            dist_to_finish = math.sqrt((pos[0] - self.target_x) ** 2 + (pos[1] - self.target_y) ** 2)
            if dist_to_finish < FINISH_REACHED_DISTANCE:
                print("\nFINISH REACHED")
                self.finishes += 1
                return True
        return False

    def compute_reward(self, distance, angle_diff, min_front, min_left, min_right, speed):
        """
        Вычисление награды на основе текущего состояния.

        Args:
            distance: Расстояние до текущей цели
            angle_diff: Угловое отклонение от цели
            min_front: Минимальное расстояние до препятствия спереди
            min_left: Минимальное расстояние до препятствия слева
            min_right: Минимальное расстояние до препятствия справа
            speed: Текущая скорость

        Returns:
            tuple: (reward, done) награда и флаг завершения эпизода
        """
        reward = 0.0
        done = False

        # Базовое вознаграждение за каждый шаг
        reward += 0.01

        # Вознаграждение за приближение к цели
        if self.prev_distance_to_target != float('inf'):
            progress = self.prev_distance_to_target - distance
            if progress > 0:
                reward += min(progress * 2.0, 5.0)
            else:
                reward -= 0.2

        self.prev_distance_to_target = distance

        # Вознаграждение за правильное направление
        alignment = 1 - min(abs(angle_diff) / math.pi, 1.0)
        reward += alignment * 0.3

        # Вознаграждение за движение
        if speed > 1.0:
            reward += 0.02
        else:
            reward -= 0.05

        # Штраф за близость к препятствиям
        if min_front < 1.5:
            reward -= (1.5 - min_front) * 2.0
        if min_left < 0.8:
            reward -= (0.8 - min_left) * 1.5
        if min_right < 0.8:
            reward -= (0.8 - min_right) * 1.5

        # Штраф за столкновение
        collision = (min_front < 0.5 or min_left < 0.3 or min_right < 0.3)
        if collision:
            reward -= 20.0
            done = True
            self.dtp_count += 1
            print("COLLISION DETECTED")

        reward = np.clip(reward, -30, 30)
        return reward, done

    def make_state(self, front_state, distance, angle_diff, speed, min_front, min_left, min_right, in_snow):
        """
        Формирование вектора состояния для PPO агента с учётом снега.

        Returns:
            numpy.ndarray: Вектор состояния
        """
        norm_distance = min(distance / 100.0, 1.0)
        norm_angle = angle_diff / math.pi
        norm_speed = speed / MAX_SPEED
        norm_min_front = min(min_front / 10.0, 1.0)
        norm_min_left = min(min_left / 5.0, 1.0)
        norm_min_right = min(min_right / 5.0, 1.0)

        progress = self.current_target_idx / max(len(self.current_waypoints), 1)
        in_snow_flag = 1.0 if in_snow else 0.0

        state = np.array(list(front_state) + [
            norm_distance, norm_angle, norm_speed,
            norm_min_front, norm_min_left, norm_min_right,
            progress, in_snow_flag
        ], dtype=np.float32)

        # Замена нечисловых значений
        if np.any(np.isnan(state)) or np.any(np.isinf(state)):
            state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)

        return state

    def apply_action(self, action, speed_multiplier):
        """
        Применение действия PPO агента к автомобилю с учётом снега.

        Args:
            action: Массив [speed, steering] в диапазоне [-1, 1]
            speed_multiplier: Множитель скорости от снега
        """
        target_speed = (action[0] + 1.0) / 2.0 * MAX_SPEED
        target_speed *= speed_multiplier  # Применяем эффект снега

        target_steer = np.clip(action[1], -1.0, 1.0) * MAX_STEER

        # Плавное изменение параметров для стабильности
        self.speed = self.speed * 0.9 + target_speed * 0.1
        self.steering_angle = self.steering_angle * 0.8 + target_steer * 0.2

        self.speed = np.clip(self.speed, 0.0, MAX_SPEED)
        self.steering_angle = np.clip(self.steering_angle, -MAX_STEER, MAX_STEER)

        driver.setCruisingSpeed(self.speed)
        driver.setSteeringAngle(self.steering_angle)

    def reset_episode(self):
        """Сброс текущего эпизода и отправка сигнала на перегенерацию мира."""
        if self.episode > 0:
            print(f"\n{'-' * 50}")
            print(f"Episode {self.episode} completed")
            print(f"   Total reward: {self.episode_reward:.2f}")
            print(f"   Finishes: {self.finishes}")
            print(f"   Collisions: {self.dtp_count}")
            print(f"{'-' * 50}\n")

            # Сохранение лога эпизода
            log_entry = {
                "episode": self.episode,
                "steps": self.step_count,
                "reward": round(self.episode_reward, 2),
                "finishes": self.finishes,
                "collisions": self.dtp_count
            }
            with open(os.path.join("logs_ppo", "episode_log.json"), "a") as f:
                f.write(json.dumps(log_entry) + "\n")

        # Сохранение модели каждые 10 эпизодов
        if self.episode > 0 and self.episode % 10 == 0 and self.episode != self.last_save_episode:
            self.agent.save(f"models/ppo_model_episode_{self.episode}.pth")
            self.last_save_episode = self.episode

        self.speed = 0.0
        self.steering_angle = 0.0
        self.waiting_reset = True
        self.reset_steps = RESET_STEPS
        self.prev_distance_to_target = float('inf')
        self.prev_state = None
        self.prev_action = None
        self.episode_reward = 0.0
        self.episode += 1

        # Сбрасываем карты колеи
        self.snow_manager.reset_tracks()

        try:
            sock.sendto(b"RESET", SUPERVISOR_ADDR)
        except:
            pass

    def step(self):
        """Основной шаг контроллера, вызываемый каждый тик симуляции."""

        # Ожидание стабилизации после сброса
        if self.waiting_reset:
            self.speed = 0.0
            self.steering_angle = 0.0
            driver.setCruisingSpeed(0)
            driver.setSteeringAngle(0)

            self.reset_steps -= 1
            if self.reset_steps <= 0:
                self.waiting_reset = False
                finish = get_finish_position()
                if finish:
                    self.target_x, self.target_y = finish
                    print(f"After reset, finish at: ({self.target_x:.1f}, {self.target_y:.1f})")
                self.plan_route()
            return

        # Получение данных с датчиков
        pos = self.get_position()
        speed = self.get_speed()
        front_state, min_front, min_left, min_right = self.get_lidar_state()

        # =====================================================================
        # ОБРАБОТКА ЭФФЕКТА СНЕГА
        # =====================================================================
        speed_multiplier, in_snow, snow_id = self.snow_manager.get_snow_effect(pos[0], pos[1])

        if in_snow and snow_id >= 0:
            self.snow_manager.update_track(pos[0], pos[1], snow_id, speed)
            if not self.last_in_snow:
                print(f"[SNOW] Entered snow zone - speed reduced to {speed_multiplier:.0%}")
        elif self.last_in_snow:
            print("[ROAD] Exited snow zone - speed restored")

        self.last_in_snow = in_snow

        # Получение информации о цели
        target, distance, angle_diff = self.get_target_info()

        if target is None:
            self.plan_route()
            return

        # Проверка условий
        self.check_waypoint_reached()

        if self.check_finish_reached():
            self.reset_episode()
            return

        if min_front < 0.4 or min_left < 0.25 or min_right < 0.25:
            self.reset_episode()
            return

        # Формирование состояния и получение награды
        state = self.make_state(front_state, distance, angle_diff, speed,
                                min_front, min_left, min_right, in_snow)

        reward, crashed = self.compute_reward(distance, angle_diff, min_front,
                                              min_left, min_right, speed)
        self.episode_reward += reward

        # Обучение агента
        if self.prev_state is not None and not np.isnan(reward) and not np.isinf(reward):
            done = crashed or self.check_finish_reached()
            self.agent.memory.add(self.prev_state, self.prev_action, reward, state, done)

            if len(self.agent.memory) > 64:
                self.agent.train()

        # Выбор и применение действия с учётом снега
        action = self.agent.select_action(state)
        self.apply_action(action, speed_multiplier)

        # Сохранение состояния для следующего шага
        self.prev_state = state
        self.prev_action = action
        self.step_count += 1

        # Логирование
        if self.step_count % 30 == 0:
            progress = f"{self.current_target_idx}/{len(self.current_waypoints)}"
            snow_mark = "[SNOW]" if in_snow else "[ROAD]"
            print(f"Episode:{self.episode:3d} | Reward:{self.episode_reward:6.1f} | "
                  f"WP:{progress} | Dist:{distance:5.1f} | Speed:{self.speed:4.1f} {snow_mark} | "
                  f"Mult:{speed_multiplier:.2f}x")


print("Waiting for simulation to start...\n")

navigator = PPOWaypointNavigator()

while driver.step() != -1:
    try:
        navigator.step()
    except Exception as e:
        print(f"Error in main loop: {e}")