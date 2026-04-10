from vehicle import Driver
import math
import  numpy as np
from sac import SACAgent
from dqn import DQN
from box import box
import os
import level_difficult as ld

# Координаты финиша
TARGET_X, TARGET_Y, TARGET_Z = 121.345, -32.7707, 4.83 #большая карта
# TARGET_X, TARGET_Y, TARGET_Z = 134.851, 41.8667, 1.42109e-14 #малая карта
TIME_STEP = 32
MAX_SPEED = 40
SAC = SACAgent()
DQN_AGENT = DQN()
# SAC.load("best_model/sac_model_episode_3762777_1080.pth")

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

run_sac = False
check = False
episodes = 0

def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'reward_log.txt')

def log_reward(situations):
    if len(situations) > 2:
        with open(log_file, "a") as f:
            f.write(','.join(str(item) for item in situations) + '\n')

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

while driver.step() != -1:
    front_data = lidar_front.getRangeImage()
    left_data = lidar_left.getRangeImage()
    right_data = lidar_right.getRangeImage()
    back_data = lidar_back.getRangeImage()
    min_front_data = min(min(front_data), min(right_data), min(left_data), min(back_data))
    min_left = min(left_data)
    min_right = min(right_data)

    # берем только 57 лучей
    front_data = np.clip(front_data, 0, 20)
    front_state = front_data[::9]
    lidar_info = build_lidar_info(front_data)
    # --- Текущая позиция ---
    pos = gps.getValues()
    car_x, car_y, car_z = pos[0], pos[1], pos[2]

    # --- Курс машины ---
    roll, pitch, yaw = imu.getRollPitchYaw()

    # --- компасс ---
    north = compass.getValues()
    heading = math.atan2(north[0], north[1])  # угол направления машины (радианы)
    dx = car_x - TARGET_X
    dy = TARGET_Y - car_y
    goal_angle = math.atan2(dy, dx)
    angle_to_goal = goal_angle + heading
    angle_to_goal = (angle_to_goal + math.pi) % (2 * math.pi) - math.pi
    cos_angle = math.cos(angle_to_goal)
    sin_angle = math.sin(angle_to_goal)
    speed = driver.getCurrentSpeed()
    speed = np.clip(speed, -MAX_SPEED, MAX_SPEED)
    wheel_angle = driver.getSteeringAngle()
    distance = math.sqrt((car_x - TARGET_X)**2 + (car_y - TARGET_Y)**2)

    danger_rays_left, danger_rays_front, danger_rays_right = ld.choice_ray(front_state)

    danger = ld.danger_level(danger_rays_left, danger_rays_front, danger_rays_right, speed, wheel_angle)

    # if sum(danger) > 0.2:
    SAC.step(danger, front_state, distance, min_left, min_right, min_front_data, lidar_info)
    # if SAC.test_episode != 100:
    #     SAC.step_test(danger, front_state, distance, min_left, min_right, min_front_data, lidar_info)
    # else:
    #     break
    # DQN_AGENT.step(front_state, distance, cos_angle, sin_angle, danger, min_left, min_right)
    driver.setCruisingSpeed(SAC.speed)
    driver.setSteeringAngle(SAC.steering_angle)
    # else:
    #     SAC.episode = 0
    #     SAC.check = True
    #     driver.setCruisingSpeed(MAX_SPEED/2)
    #     if 0.99 > cos_angle:
    #         if sin_angle > 0:
    #             driver.setSteeringAngle(0.6)
    #         elif sin_angle < 0:
    #             driver.setSteeringAngle(-0.6)
    #     else:
    #         driver.setSteeringAngle(0)

