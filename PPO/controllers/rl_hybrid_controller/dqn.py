import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from box import box
import socket

# Настройка сокета для отправки сигнала Supervisor
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
SUPERVISOR_ADDR = ("127.0.0.1", 6006)

max_speed = 60.0
MAX_STEER = 0.5      # максимальный угол поворота колёс (радианы)
ACCELERATION = 0.5   # скорость нарастания скорости
BRAKE_FORCE = 1.0    # сила торможения
speed = 0.0
steering_angle = 0.0

check = 0

# --- параметры ---
STATE_SIZE = 6
ACTION_SIZE = 3
GAMMA = 0.99
LR = 1e-4
EPSILON = 0.6
EPSILON_DECAY = 0.9
EPSILON_MIN = 0.05
BATCH_SIZE = 64
MEMORY_SIZE = 200000
TAU = 0.01  # soft update
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
import os

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "reward_log.txt")

# --- сеть ---
class DuelingDQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.feature = nn.Sequential(
            # полносвязный слой
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, action_size)
        )

    def forward(self, x):
        f = self.feature(x)
        value = self.value_stream(f)
        adv = self.advantage_stream(f)
        return value + adv - adv.mean(1, keepdim=True)

# --- инициализация ---
policy_net = DuelingDQN(STATE_SIZE, ACTION_SIZE).to(device)
target_net = DuelingDQN(STATE_SIZE, ACTION_SIZE).to(device)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = deque(maxlen=MEMORY_SIZE)

class DQN:
    def __init__(self):
        self.epsilon = EPSILON
        self.step_count = 0
        self.prev_state = None
        self.prev_action = None
        self.prev_distance = None
        self.done = False
        self.reward_episod = []
        self.situation = []

        self.speed = 0.0
        self.steering_angle = 0.0

    def make_state(self, danger, distance, cos_angle, sin_angle):
        dist = distance/115
        state_vec = np.array([danger[0], danger[1], danger[2], dist, cos_angle, sin_angle], dtype=np.float32)

        return state_vec

    def select_action(self, state_vec):
        if np.random.rand() < self.epsilon:
            return random.randint(0, ACTION_SIZE - 1)
        with torch.no_grad():
            q = policy_net(torch.tensor(state_vec).unsqueeze(0).to(device))
        return int(q.argmax().item())

    def apply_action(self, action):
        if action == 0:
            self.speed += ACCELERATION
        elif action == 1:
            self.steering_angle -= 0.03
        elif action == 2:
            self.steering_angle += 0.03
        elif action == 3:
            self.speed -= ACCELERATION

        self.speed *= 0.98
        self.speed = np.clip(self.speed, -max_speed, max_speed)
        self.steering_angle = np.clip(self.steering_angle, -MAX_STEER, MAX_STEER)

    def compute_reward(self, distance, front, min_left, min_right):
        reward = 1 - (distance/115)

        reward -= 0.1

        if min(front) < 5 and distance < 5:
            box.finish = True
            self.done = True
            print('finish')
            return 100.0
        if min(front) < 0.5 or min_left < 0.2 or min_right < 0.2:
            box.dtp = True
            print('dtp')
            return -10.0
        
        return reward

    def train(self):
        if len(memory) < BATCH_SIZE:
            return

        batch = random.sample(memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.from_numpy(np.array(states, dtype=np.float32)).to(device)
        next_states = torch.from_numpy(np.array(next_states, dtype=np.float32)).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)


        q = policy_net(states).gather(1, actions)
        next_q = target_net(next_states).max(1, keepdim=True)[0]
        target = rewards + GAMMA * next_q * (1 - dones)

        loss = nn.MSELoss()(q, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for tp, pp in zip(target_net.parameters(), policy_net.parameters()):
            tp.data.copy_(TAU * pp.data + (1 - TAU) * tp.data)

    def step(self, front, distance, cos_angle, sin_angle, danger, min_left, min_right):
        state = self.make_state(danger, distance, cos_angle, sin_angle)
        loss = 0
        reward = 0
        if self.prev_state is not None:
            reward = self.compute_reward(distance, front, min_left, min_right)
            self.reward_episod.append(reward)
            memory.append((self.prev_state, self.prev_action, reward, state, self.done))
            if self.step_count % 10 == 0:
                loss = self.train()

        action = self.select_action(state)
        self.apply_action(action)

        self.prev_state = state
        self.prev_action = action
        self.prev_distance = distance
        # self.step_count += 1
        self.situation.append(reward)
        self.situation.append(distance)
        self.situation.append(0 if loss == None else loss)
        self.situation.append(self.step_count)
        self.situation.append(0 if box.dtp == False else 1)
        self.situation.append(0 if box.finish == False else 1)
        self.situation.append(self.epsilon)

        if box.dtp or box.finish:
            if self.epsilon > EPSILON_MIN:
                self.epsilon *= EPSILON_DECAY
            print(self.epsilon)
            self.reset()

    def reset(self):
        self.prev_state = None
        self.prev_action = None
        self.prev_distance = None
        self.prev_min_lidar = 5
        self.done = False
        self.speed = 0
        self.steering_angle = 0
        box.dtp = False
        box.finish = False
        sock.sendto(b"RESET", SUPERVISOR_ADDR)
