from vehicle import Driver
import math
import numpy as np
from sac import SACAgent
from dqn import DQN
from box import box
import os
import json
import socket
from path_planner import PathPlanner
import level_difficult as ld

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
SUPERVISOR_ADDR = ("127.0.0.1", 6006)
sock.setblocking(False)

TIME_STEP = 32
MAX_SPEED = 40
SAC = SACAgent()
DQN_AGENT = DQN()

driver = Driver()
timestep = int(driver.getBasicTimeStep())

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

current_waypoints = []
current_target_idx = 0
waypoint_reached_distance = 5.0
obstacles = []
waiting_reset = False

possible_paths = [
    "world_data.json",
    "../ppo_waypoint_controller/world_data.json",
    "../../controllers/ppo_waypoint_controller/world_data.json"
]

for path in possible_paths:
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
            obstacles = data.get("obstacles", [])
        print(f"Loaded {len(obstacles)} obstacles from {path}")
        break
else:
    print("world_data.json not found, using empty obstacle list")

path_planner = PathPlanner()


def get_finish_position():
    finish_paths = [
        "finish_pos.txt",
        "../car_reset/finish_pos.txt",
        "../../controllers/car_reset/finish_pos.txt"
    ]
    for path in finish_paths:
        try:
            if os.path.exists(path):
                with open(path, "r") as f:
                    content = f.read().strip()
                    parts = content.split()
                    if len(parts) >= 2:
                        return (float(parts[0]), float(parts[1]))
        except:
            pass
    return None


def wait_for_finish_file():
    finish_paths = [
        "finish_pos.txt",
        "../car_reset/finish_pos.txt",
        "../../controllers/car_reset/finish_pos.txt"
    ]
    for _ in range(200):
        for path in finish_paths:
            if os.path.exists(path):
                try:
                    with open(path, "r") as f:
                        content = f.read().strip()
                        parts = content.split()
                        if len(parts) >= 2:
                            return (float(parts[0]), float(parts[1]))
                except:
                    pass
        driver.step()
    return None


def plan_route():
    global current_waypoints, current_target_idx

    finish = wait_for_finish_file()
    if finish is None:
        target_x, target_y = 155.157, -155.751
    else:
        target_x, target_y = finish

    pos = None
    for _ in range(50):
        pos_vals = gps.getValues()
        if pos_vals and len(pos_vals) >= 2:
            x, y = pos_vals[0], pos_vals[1]
            if not math.isnan(x) and not math.isnan(y):
                pos = (x, y)
                break
        driver.step()

    if pos is None:
        return

    start_pos = pos
    target_pos = (target_x, target_y)

    print(
        f"\nPlanning route from ({start_pos[0]:.1f}, {start_pos[1]:.1f}) to ({target_pos[0]:.1f}, {target_pos[1]:.1f})")

    current_waypoints = path_planner.plan_path(start_pos, target_pos, obstacles)
    current_target_idx = 0

    if current_waypoints and len(current_waypoints) > 0:
        valid_waypoints = []
        for wp in current_waypoints:
            if not math.isnan(wp[0]) and not math.isnan(wp[1]):
                valid_waypoints.append(wp)

        if valid_waypoints:
            waypoints_str = ";".join([f"{wp[0]:.2f},{wp[1]:.2f}" for wp in valid_waypoints])
            try:
                sock.sendto(f"WAYPOINTS:{waypoints_str}".encode(), SUPERVISOR_ADDR)
                print(f"Sent {len(valid_waypoints)} waypoints")
            except Exception as e:
                print(f"Error sending waypoints: {e}")


def build_lidar_info(front_data):
    n = 57
    fov = 3.4
    lidar_info = []
    total = len(front_data)
    for i in range(n):
        idx = int(i * total / n)
        dist = front_data[idx]
        angle = (i / (n - 1)) * fov - fov / 2
        lidar_info.append((dist, angle))
    return lidar_info


print("Waiting for GPS stabilization...")
for _ in range(50):
    driver.step()

plan_route()

while driver.step() != -1:
    front_data = lidar_front.getRangeImage()
    left_data = lidar_left.getRangeImage()
    right_data = lidar_right.getRangeImage()
    back_data = lidar_back.getRangeImage()
    min_front_data = min(min(front_data), min(right_data), min(left_data), min(back_data))
    min_left = min(left_data)
    min_right = min(right_data)

    front_data = np.clip(front_data, 0, 20)
    front_state = front_data[::9]
    lidar_info = build_lidar_info(front_data)

    pos = gps.getValues()
    car_x, car_y, car_z = pos[0], pos[1], pos[2]

    if math.isnan(car_x) or math.isnan(car_y):
        continue

    north = compass.getValues()
    heading = math.atan2(north[0], north[1])

    finish = get_finish_position()
    if finish:
        target_x, target_y = finish
    else:
        target_x, target_y = 155.157, -155.751

    distance = math.sqrt((car_x - target_x) ** 2 + (car_y - target_y) ** 2)

    if current_waypoints and current_target_idx < len(current_waypoints):
        target_wp_x = current_waypoints[current_target_idx][0]
        target_wp_y = current_waypoints[current_target_idx][1]
        distance_to_wp = math.sqrt((car_x - target_wp_x) ** 2 + (car_y - target_wp_y) ** 2)

        if distance_to_wp < waypoint_reached_distance:
            print(f"Waypoint {current_target_idx + 1}/{len(current_waypoints)} reached")
            current_target_idx += 1

            remaining = current_waypoints[current_target_idx:]
            if remaining:
                valid_remaining = []
                for wp in remaining:
                    if not math.isnan(wp[0]) and not math.isnan(wp[1]):
                        valid_remaining.append(wp)
                if valid_remaining:
                    waypoints_str = ";".join([f"{wp[0]:.2f},{wp[1]:.2f}" for wp in valid_remaining])
                    try:
                        sock.sendto(f"WAYPOINTS:{waypoints_str}".encode(), SUPERVISOR_ADDR)
                    except:
                        pass
            else:
                try:
                    sock.sendto(b"CLEAR_WAYPOINTS", SUPERVISOR_ADDR)
                except:
                    pass

        if current_target_idx < len(current_waypoints):
            target_wp_x = current_waypoints[current_target_idx][0]
            target_wp_y = current_waypoints[current_target_idx][1]
            dx = target_wp_x - car_x
            dy = target_wp_y - car_y
            goal_angle = math.atan2(dy, dx)
        else:
            dx = target_x - car_x
            dy = target_y - car_y
            goal_angle = math.atan2(dy, dx)
    else:
        dx = target_x - car_x
        dy = target_y - car_y
        goal_angle = math.atan2(dy, dx)

    angle_to_goal = goal_angle + heading
    angle_to_goal = (angle_to_goal + math.pi) % (2 * math.pi) - math.pi

    speed = driver.getCurrentSpeed()
    speed = np.clip(speed, -MAX_SPEED, MAX_SPEED)
    wheel_angle = driver.getSteeringAngle()

    danger_rays_left, danger_rays_front, danger_rays_right = ld.choice_ray(front_state)
    danger = ld.danger_level(danger_rays_left, danger_rays_front, danger_rays_right, speed, wheel_angle)

    SAC.step(front_state, distance, min_left, min_right, min_front_data, lidar_info, pos)

    driver.setCruisingSpeed(SAC.speed)
    driver.setSteeringAngle(SAC.steering_angle)

    if SAC.waiting_reset:
        waiting_reset = True
    elif waiting_reset:
        waiting_reset = False
        print("\nRESET COMPLETED - REPLANNING ROUTE")
        plan_route()