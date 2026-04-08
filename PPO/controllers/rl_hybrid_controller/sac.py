import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import math
from box import box
import socket

# ================= ПАРАМЕТРЫ ОКРУЖЕНИЯ =================
max_speed = 40.0
MAX_STEER = 0.6
ACCELERATION = 0.5

# ================= ПАРАМЕТРЫ SAC =================
STATE_SIZE = 6  # [danger_left, danger_front, danger_right, distance, cos_angle, sin_angle]
ACTION_DIM = 2  # [throttle, steering]

GAMMA = 0.99
TAU = 0.005
ALPHA = 0.2  # Температура энтропии

LR_ACTOR = 3e-4
LR_CRITIC = 3e-4

BATCH_SIZE = 64
MEMORY_SIZE = 200000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def add(self, s, a, r, s2, d):
        self.buffer.append((s, a, r, s2, d))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return map(np.array, zip(*batch))

    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.mean = nn.Linear(128, action_dim)
        self.log_std = nn.Linear(128, action_dim)

    def forward(self, state):
        x = self.net(state)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)

        x_t = normal.rsample()
        action = torch.tanh(x_t)

        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=1))


class SACAgent:
    def __init__(self):
        self.actor = Actor(STATE_SIZE, ACTION_DIM).to(device)

        self.critic1 = Critic(STATE_SIZE, ACTION_DIM).to(device)
        self.critic2 = Critic(STATE_SIZE, ACTION_DIM).to(device)

        self.target_critic1 = Critic(STATE_SIZE, ACTION_DIM).to(device)
        self.target_critic2 = Critic(STATE_SIZE, ACTION_DIM).to(device)

        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=LR_CRITIC
        )

        self.memory = ReplayBuffer(MEMORY_SIZE)

        self.prev_state = None
        self.prev_action = None
        self.done = False
        self.step_count = 0

        self.speed = 0.0
        self.steering_angle = 0.0
        self.prev_distance = None

    def make_state(self, danger, distance, cos_angle, sin_angle):
        """Создание вектора состояния"""
        dist = distance / 25.0  # Нормализация (максимум ~25м)

        state_vec = np.array([
            danger[0],  # Опасность слева [0-1]
            danger[1],  # Опасность спереди [0-1]
            danger[2],  # Опасность справа [0-1]
            dist,  # Расстояние до цели [0-1]
            cos_angle,  # Направление к цели [-1, 1]
            sin_angle  # Направление к цели [-1, 1]
        ], dtype=np.float32)

        return state_vec

    def select_action(self, state):
        """Выбор действия (детерминированный для теста, стохастический для обучения)"""
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action, _ = self.actor.sample(state)
        return action.detach().cpu().numpy()[0]

    def apply_action(self, action):
        """Применение действия к внутреннему состоянию"""
        # action в диапазоне [-1, 1] от tanh
        throttle = action[0]  # Газ/тормоз
        steer = action[1]  # Поворот

        # Масштабируем throttle до [-1, 1] для ACCELERATION
        self.speed += throttle * ACCELERATION * 2  # *2 для более быстрой реакции

        # Плавное затухание скорости
        self.speed *= 0.98

        # Ограничение скорости
        self.speed = np.clip(self.speed, -max_speed, max_speed)

        # Поворот
        self.steering_angle += steer * 0.05
        self.steering_angle = np.clip(self.steering_angle, -MAX_STEER, MAX_STEER)

    def compute_reward(self, distance, front, min_left, min_right):
        """Вычисление награды"""
        reward = 0.0

        # Награда за прогресс к цели
        if self.prev_distance is not None:
            progress = self.prev_distance - distance
            reward += progress * 5.0  # 5x множитель для важности прогресса

        self.prev_distance = distance

        # Базовая награда за выживание
        reward += 0.1

        # Штраф за столкновение
        min_front = min(front) if front else 30.0

        if min_front < 0.8:
            box.dtp = True
            reward -= 50.0
            return reward

        if min_left < 0.3 or min_right < 0.3:
            box.dtp = True
            reward -= 50.0
            return reward

        # Награда за достижение цели
        if distance < 2.0:
            box.finish = True
            reward += 100.0
            return reward

        # Штраф за медленное движение (избегать кругов на месте)
        if abs(self.speed) < 0.5 and self.step_count > 50:
            reward -= 0.2

        # Штраф за отклонение от курса (опционально)
        # reward -= abs(self.steering_angle) * 0.01

        return reward

    def train(self):
        """Обучение на батче"""
        if len(self.memory) < BATCH_SIZE:
            return 0.0

        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        # --- Обновление критиков ---
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_states)

            target_q1 = self.target_critic1(next_states, next_action)
            target_q2 = self.target_critic2(next_states, next_action)

            target_q = torch.min(target_q1, target_q2) - ALPHA * next_log_prob
            target = rewards + GAMMA * (1 - dones) * target_q

        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        critic_loss = nn.MSELoss()(current_q1, target) + \
                      nn.MSELoss()(current_q2, target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Обновление актора ---
        new_action, log_prob = self.actor.sample(states)

        q1_new = self.critic1(states, new_action)
        q2_new = self.critic2(states, new_action)

        actor_loss = (ALPHA * log_prob - torch.min(q1_new, q2_new)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- Soft update целевых сетей ---
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        return critic_loss.item()

    def step(self, danger, front, distance, cos_angle, sin_angle, min_left, min_right):
        """Один шаг агента"""
        state = self.make_state(danger, distance, cos_angle, sin_angle)

        reward = 0.0

        if self.prev_state is not None:
            reward = self.compute_reward(distance, front, min_left, min_right)

            self.memory.add(
                self.prev_state,
                self.prev_action,
                reward,
                state,
                self.done
            )

            # Обучаем каждый шаг
            self.train()

        # Выбор действия
        action = self.select_action(state)
        self.apply_action(action)

        self.prev_state = state
        self.prev_action = action
        self.step_count += 1

        return reward

    def reset(self):
        """Сброс эпизода"""
        self.prev_state = None
        self.prev_action = None
        self.done = False
        self.speed = 0.0
        self.steering_angle = 0.0
        self.prev_distance = None
        self.step_count = 0
        box.dtp = False
        box.finish = False