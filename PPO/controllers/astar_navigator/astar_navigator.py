from vehicle import Driver
import math
import numpy as np
import socket
import json
import threading

# =============================================================================
# ПАРАМЕТРЫ
# =============================================================================
MAX_SPEED = 25.0
MAX_STEER = 0.6
WAYPOINT_REACHED_DISTANCE = 5.0
FINISH_REACHED_DISTANCE = 7.0
RESET_STEPS = 50
MAX_EPISODE_STEPS = 2000

WORLD_MIN_X = -100
WORLD_MAX_X = 200
WORLD_MIN_Y = -200
WORLD_MAX_Y = 150

START_POINT = (WORLD_MAX_X - 10, WORLD_MIN_Y + 10)
FINISH_Z = 11.0  # Высота финиша


# =============================================================================
# UDP ДЛЯ ПРИЁМА ДАННЫХ ОТ СПУТНИКА
# =============================================================================
class SatelliteReceiver:
    def __init__(self, port=6007):
        self.waypoints = []
        self.obstacles = []
        self.frame = 0
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", port))
        self.sock.settimeout(0.05)
        self.running = True

    def update(self):
        try:
            data, _ = self.sock.recvfrom(65536)
            received = json.loads(data.decode())
            self.obstacles = received.get('obstacles', [])
            self.waypoints = received.get('waypoints', [])
            self.frame = received.get('frame', 0)
        except socket.timeout:
            pass
        except Exception as e:
            print(f"Receive error: {e}")

    def get_waypoints(self):
        return self.waypoints

    def get_obstacles(self):
        return self.obstacles


# =============================================================================
# ИНИЦИАЛИЗАЦИЯ WEBOTS
# =============================================================================
driver = Driver()
timestep = int(driver.getBasicTimeStep())
print(f"Timestep: {timestep} ms")

gps = driver.getDevice("gps")
if gps:
    gps.enable(timestep)
    print("GPS enabled")

compass = driver.getDevice("compass")
if compass:
    compass.enable(timestep)
    print("Compass enabled")

lidar_front = driver.getDevice("lidar_front")
if lidar_front:
    lidar_front.enable(timestep)
    print("Lidar front enabled")

lidar_left = driver.getDevice("lidar_left")
if lidar_left:
    lidar_left.enable(timestep)
    print("Lidar left enabled")

lidar_right = driver.getDevice("lidar_right")
if lidar_right:
    lidar_right.enable(timestep)
    print("Lidar right enabled")

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setblocking(False)
CAR_RESET_ADDR = ("127.0.0.1", 6006)

receiver = SatelliteReceiver()


def send_waypoints_to_supervisor(waypoints):
    if not waypoints or len(waypoints) < 2:
        return

    waypoints_str = ";".join([f"{wp[0]:.2f},{wp[1]:.2f}" for wp in waypoints])
    try:
        sock.sendto(f"WAYPOINTS:{waypoints_str}".encode(), CAR_RESET_ADDR)
        print(f"  Sent {len(waypoints)} waypoints to create blue cubes")
    except Exception as e:
        print(f"  Error: {e}")


def clear_waypoints():
    try:
        sock.sendto(b"CLEAR_WAYPOINTS", CAR_RESET_ADDR)
    except:
        pass


def update_finish_z():
    """Обновление высоты финиша в car_reset"""
    try:
        sock.sendto(f"FINISH_Z:{FINISH_Z}".encode(), CAR_RESET_ADDR)
    except:
        pass


# =============================================================================
# ОСНОВНОЙ КЛАСС
# =============================================================================
class AStarNavigator:
    def __init__(self):
        self.current_waypoints = []
        self.current_target_idx = 0
        self.waiting_reset = True
        self.reset_steps = RESET_STEPS

        self.prev_distance_to_target = float('inf')
        self.episode = 0
        self.finishes = 0
        self.dtp_count = 0
        self.step_count = 0
        self.episode_steps = 0
        self.episode_reward = 0
        self.speed = 0.0
        self.steering_angle = 0.0

        update_finish_z()
        print(f"Start: {START_POINT}")
        print(f"Finish Z: {FINISH_Z}m")

    def get_position(self):
        pos = gps.getValues()
        return (pos[0], pos[1])

    def get_heading(self):
        north = compass.getValues()
        return math.atan2(north[0], north[1])

    def get_speed(self):
        return driver.getCurrentSpeed()

    def get_lidar_state(self):
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
        _, distance, _ = self.get_target_info()
        if distance is not None and distance < WAYPOINT_REACHED_DISTANCE:
            print(f"Blue cube {self.current_target_idx + 1}/{len(self.current_waypoints)} reached")
            self.current_target_idx += 1

            remaining = self.current_waypoints[self.current_target_idx:]
            if remaining:
                send_waypoints_to_supervisor(remaining)
            else:
                clear_waypoints()

            self.prev_distance_to_target = float('inf')
            return True
        return False

    def check_finish_reached(self):
        if self.current_waypoints and self.current_target_idx >= len(self.current_waypoints):
            pos = self.get_position()
            dist = math.sqrt((pos[0] - self.finish_x) ** 2 + (pos[1] - self.finish_y) ** 2)
            if dist < FINISH_REACHED_DISTANCE:
                print("\n🏁 FINISH REACHED! 🏁")
                self.finishes += 1
                return True
        return False

    def compute_reward(self, distance, angle_diff, min_front, min_left, min_right, speed):
        reward = 0.01

        if self.prev_distance_to_target != float('inf'):
            progress = self.prev_distance_to_target - distance
            if progress > 0:
                reward += min(progress * 2.0, 5.0)
            else:
                reward -= 0.2

        self.prev_distance_to_target = distance

        alignment = 1 - min(abs(angle_diff) / math.pi, 1.0)
        reward += alignment * 0.3

        if speed > 1.0:
            reward += 0.02
        else:
            reward -= 0.05

        if min_front < 1.5:
            reward -= (1.5 - min_front) * 2.0
        if min_left < 0.8:
            reward -= (0.8 - min_left) * 1.5
        if min_right < 0.8:
            reward -= (0.8 - min_right) * 1.5

        collision = (min_front < 0.5 or min_left < 0.25 or min_right < 0.25)
        if collision:
            reward -= 20.0
            self.dtp_count += 1
            print("💥 COLLISION!")
            return reward, True

        return reward, False

    def update_waypoints(self):
        waypoints = receiver.get_waypoints()
        if waypoints and len(waypoints) > 1:
            self.current_waypoints = waypoints
            self.current_target_idx = 0
            self.prev_distance_to_target = float('inf')
            send_waypoints_to_supervisor(self.current_waypoints)
            print(f"\n📍 Updated route: {len(self.current_waypoints)} waypoints")
            for i, wp in enumerate(self.current_waypoints[:5]):
                print(f"    WP {i + 1}: ({wp[0]:.1f}, {wp[1]:.1f})")
            return True
        return False

    def reset_episode(self):
        success_rate = (self.finishes / max(self.episode, 1)) * 100
        print(f"\n{'=' * 50}")
        print(f"Episode {self.episode} completed")
        print(f"  Reward: {self.episode_reward:.2f}")
        print(f"  Finishes: {self.finishes} ({success_rate:.1f}%)")
        print(f"  Collisions: {self.dtp_count}")
        print(f"{'=' * 50}\n")

        self.speed = 0.0
        self.steering_angle = 0.0
        self.waiting_reset = True
        self.reset_steps = RESET_STEPS
        self.prev_distance_to_target = float('inf')
        self.episode_reward = 0.0
        self.episode_steps = 0
        self.episode += 1

        try:
            sock.sendto(b"RESET", CAR_RESET_ADDR)
        except:
            pass

    def step(self):
        receiver.update()

        if self.waiting_reset:
            self.speed = 0.0
            self.steering_angle = 0.0
            driver.setCruisingSpeed(0)
            driver.setSteeringAngle(0)

            self.reset_steps -= 1
            if self.reset_steps <= 0:
                self.waiting_reset = False
                self.update_waypoints()
                update_finish_z()
            return

        self.episode_steps += 1
        pos = self.get_position()
        speed = self.get_speed()
        front_state, min_front, min_left, min_right = self.get_lidar_state()

        if self.episode_steps % 100 == 0:
            self.update_waypoints()

        target, distance, angle_diff = self.get_target_info()
        if target is None:
            if not self.update_waypoints():
                self.current_waypoints = []
                self.reset_episode()
                return

        self.check_waypoint_reached()

        finish = (155.2, -155.8)
        dist_to_finish = math.hypot(pos[0] - finish[0], pos[1] - finish[1])
        if dist_to_finish < FINISH_REACHED_DISTANCE:
            self.reset_episode()
            return

        if min_front < 0.4 or min_left < 0.25 or min_right < 0.25:
            self.reset_episode()
            return

        if self.episode_steps >= MAX_EPISODE_STEPS:
            print("⏱️ TIMEOUT!")
            self.reset_episode()
            return

        reward, done = self.compute_reward(distance, angle_diff, min_front,
                                           min_left, min_right, speed)
        self.episode_reward += reward

        steering = np.clip(angle_diff * 1.2, -MAX_STEER, MAX_STEER)

        if min_front < 1.5:
            speed_cmd = min(MAX_SPEED, 8.0)
        elif distance > 20:
            speed_cmd = MAX_SPEED
        elif distance > 10:
            speed_cmd = MAX_SPEED * 0.7
        else:
            speed_cmd = MAX_SPEED * 0.4

        driver.setCruisingSpeed(speed_cmd)
        driver.setSteeringAngle(steering)

        self.step_count += 1

        if self.step_count % 30 == 0:
            progress = f"{self.current_target_idx}/{len(self.current_waypoints)}"
            print(f"Ep:{self.episode:3d} | Rwd:{self.episode_reward:6.1f} | "
                  f"WP:{progress} | Dist:{distance:5.1f} | Speed:{speed_cmd:4.1f}")

        if done:
            self.reset_episode()


print("=" * 60)
print("A* NAVIGATOR with Satellite Data")
print("=" * 60)

for _ in range(50):
    driver.step()

navigator = AStarNavigator()

while driver.step() != -1:
    try:
        navigator.step()
    except Exception as e:
        print(f"Error: {e}")