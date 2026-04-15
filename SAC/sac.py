import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from box import box
import socket
import math
import json

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
SUPERVISOR_ADDR = ("127.0.0.1", 6006)

# ================= ENV PARAMS =================
max_speed = 20.0
MAX_STEER = 0.6
ACCELERATION = 0.5

# ================= SAC PARAMS =================
STATE_SIZE = 60
ACTION_DIM = 2

GAMMA = 0.99
TAU = 0.005
ALPHA = 0.4

LR_ACTOR = 1e-4
LR_CRITIC = 1e-4

BATCH_SIZE = 128
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
            nn.Linear(128, 256),
            nn.ReLU(),
        )
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

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
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
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
        self.step_count = 169207
        self.episode = 0

        self.speed = 0.0
        self.steering_angle = 0.0
        self.check = True
        self.prev_distance = False
        self.finish = 0
        self.dtp = 0
        self.log_prob = 0
        self.waiting_reset = False
        self.reset_steps = 0
        self.sum_reward = 0
        self.action = [0,0]
        self.state = [0,0,0,0,0,0,0,0]
        self.check_finish = 0
        self.check_falsh = 0
        self.test_episode = 0

        # общая статистика
        self.test_finishes = 0
        self.test_failures = 0

        # текущий эпизод
        self.episode_reward = 0
        self.episode_steps = 0
        self.episode_speeds = []
        self.episode_distances = []

    def make_state(self, distance, min_lidar, front):
        dist = min(distance, 115)/115
        min_lidar = min(min_lidar,15)/15
        speed = self.speed/30
        state_vec = np.array(list(front) + [dist, self.steering_angle, speed], dtype=np.float32)
        return state_vec

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action, log_prob = self.actor.sample(state)
        return action.detach().cpu().numpy()[0], log_prob.item()

    def apply_action(self, action):
        throttle = action[0]
        steer = action[1]

        self.speed += throttle * 5
        self.steering_angle += steer * 0.1

        # self.speed *= 0.98
        self.speed = np.clip(self.speed, -0, max_speed)
        self.steering_angle = np.clip(self.steering_angle, -MAX_STEER, MAX_STEER)

    def compute_reward(self, distance, front, min_left, min_right, min_lidar, lidar_info=0):
        reward = 0
        # Подсчёт награды за приближение в виде чекпоинтов (редкие, но большие награды)
        if (self.prev_distance - distance)>=1:
            progress = 5
            self.prev_distance = distance
            self.check_falsh = 0
        else:
            progress = 0
            self.check_falsh += 1
        reward += progress

        if self.check_falsh > 100:
            reward -= 2

        # Подсчёт награды за приближение в виде пройденного расстояния (частые, но маленькие)
        check_dist = self.old_dist - distance
        if check_dist > 0:
            reward += 1
        
        self.old_dist = distance
        
        
        # Подсчёт награды/штрафа за пересечение траектории движения с препятствиями
        penalty = self.trajectory_lidar_penalty_fast(lidar_info)
        reward -= penalty

        # Подсчёт штрафа за стояние на месте (если агент едет, то штраф за время)
        reward -= (2 - abs(self.speed)) if -1 <= self.speed <= 1 else 0.01

        # Подсчёт награды за достижение цели
        if distance < 5:
            box.finish = True
            self.finish +=1
            print('Доехал!!')
            return 50.0
        # Подсчёт штрафа за дтп
        if min(front) < 0.5 or min_left < 0.2 or min_right < 0.2 or min_lidar < 0.2:
            box.dtp = True
            self.dtp +=1
            print('DTP')
            return -50.0
        
        return reward

    def predict_trajectory(self, steps=20):

        points = []

        # начальное состояние
        x, y = 0.0, 0.0

        speed = self.speed
        steering = self.steering_angle

        dt = 0.032 # Это basicTimeStep в webots

        for _ in range(steps):

            # движение вперёд
            x += math.cos(steering) * speed * dt
            y += math.sin(steering) * speed * dt

            points.append((x, y))
        return points

    def trajectory_lidar_penalty_fast(self, lidar_info):

        points = self.predict_trajectory()

        penalty = 0

        ANGLE_THRESHOLD = 0.05  # ~3 градуса

        for px, py in points:
            point_dist = math.hypot(px, py)
            point_angle = math.atan2(py, px)

            for dist, angle in lidar_info:
                diff = abs((point_angle - angle + math.pi) % (2 * math.pi) - math.pi)

                if diff < ANGLE_THRESHOLD:
                    if point_dist > dist:
                        penalty = (20-dist)/1.5
                    break

        return penalty

    # def compute_reward(self, distance, front, min_left, min_right, min_lidar):
        reward = 0

        if (self.prev_distance - distance)>=2:
            progress = 50 
            self.prev_distance = distance
        else:
            progress = 0
        # награда за движение к цели
        reward += progress

        # награда за движение к цели (за ход)
        check_dist = self.old_dist - distance
        reward += check_dist*5
        self.old_dist = distance
        

        # reward -= abs(self.steering_angle)*10
        reward -= (2 - self.speed) if self.speed <= 1 else 0.1

        # штраф за близость к препятствиям

        if min_lidar < 1:
            reward -= (1 - min_lidar)/10

        if distance < 5:
            box.finish = True
            self.finish +=1
            self.done = True
            print('Доехал!!')
            return 500.0
        if min(front) < 0.5 or min_left < 0.2 or min_right < 0.2 or min_lidar < 0.2:
            box.dtp = True
            self.done = True
            self.dtp +=1
            print('DTP')
            return -500.0
        
        # print('progress', progress, 'dist', check_dist, 'steer', abs(self.steering_angle), 'speed', ((2 - self.speed) if self.speed <= 1 else 0.1), 'lidar', (5 - min_lidar)/10 if min_lidar < 5 else 0)
        return reward*0.1

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return 0, 0, 0, 0

        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_states)

            target_q1 = self.target_critic1(next_states, next_action)
            target_q2 = self.target_critic2(next_states, next_action)

            target_q = torch.min(target_q1, target_q2) - ALPHA * next_log_prob
            target = rewards + GAMMA * (1 - dones) * target_q
            target = torch.clamp(target, -100, 100)

        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        critic_loss = nn.MSELoss()(current_q1, target) + \
                      nn.MSELoss()(current_q2, target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        new_action, log_prob = self.actor.sample(states)

        q1_new = self.critic1(states, new_action)
        q2_new = self.critic2(states, new_action)

        actor_loss = (ALPHA * log_prob - torch.min(q1_new, q2_new)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
        return actor_loss, critic_loss, q1_new.mean().item(), q2_new.mean().item()

    def step(self, danger, front, distance, min_left, min_right, min_lidar, lidar_info):
        # --- ждём пока машина стабилизируется ---
        if self.waiting_reset:
            self.speed = 0
            self.steering_angle = 0

            self.reset_steps -= 1
            if self.reset_steps <= 0:
                self.waiting_reset = False

            return
        
        if self.prev_distance == False:
            self.prev_distance = distance
            self.old_dist = 120

        # if self.step_count % 20 == 0 or self.step_count <= 1:
        self.state = self.make_state(distance, min_lidar, front)
        reward, actor_loss, critic_loss, q1_new, q2_new = 0, 0, 0, 0, 0

        # --- для динамики ---
        prev_reward = getattr(self, "last_reward", 0)
        prev_actor = getattr(self, "last_actor_loss", 0)
        prev_critic = getattr(self, "last_critic_loss", 0)
        prev_q = getattr(self, "last_q", 0)

        if self.prev_state is not None:
            self.done = box.finish or box.dtp or self.episode >= 2000
            self.memory.add(
                self.prev_state,
                self.prev_action,
                self.sum_reward,
                self.state,
                self.done
            )

            actor_loss, critic_loss, q1_new, q2_new = self.train()
            print(
            f"награда {self.sum_reward:.4f} (Δ {self.sum_reward - prev_reward:+.4f}) | "
            f"потери актера {actor_loss:.4f} (Δ {actor_loss - prev_actor:+.4f}) | "
            f"потери критика {critic_loss:.4f} (Δ {critic_loss - prev_critic:+.4f}) | "
            f"Q {q1_new:.4f} (Δ {q1_new - prev_q:+.4f}) | "
            f"энтропия {self.log_prob} | "
            f"эпизод {self.episode}"
            )
            self.last_reward = self.sum_reward
            self.last_actor_loss = actor_loss
            self.last_critic_loss = critic_loss
            self.last_q = q1_new

            self.log_reward(self.step_count, self.sum_reward, distance, actor_loss, critic_loss, q1_new, q2_new, self.log_prob)
            self.episode += 1
            self.sum_reward = 0

        self.action, self.log_prob = self.select_action(self.state)

        self.apply_action(self.action)
        reward = self.compute_reward(distance, front, min_left, min_right, min_lidar, lidar_info)
        self.sum_reward += reward

        self.prev_state = self.state
        self.prev_action = self.action
        self.step_count += 1
        if self.done:
            self.reset()
            self.episode = 0

        if  self.finish > 40 and self.finish % 20 == 0 and self.check_finish < self.finish:
            self.save(f"model/sac_model_episode_{self.step_count}_{self.finish}.pth")
            self.check_finish = self.finish
        

    def reset(self):
        self.prev_state = None
        self.prev_action = None
        self.done = False
        self.speed = 0
        self.steering_angle = 0
        box.dtp = False
        box.finish = False
        self.waiting_reset = True
        self.prev_distance = False
        self.reset_steps = 100   # сколько шагов ждать
        # if self.finish > 50:
        #     sock.sendto(b"RESET_2", SUPERVISOR_ADDR)
        # elif self.finish > 100:
        #     sock.sendto(b"RESET_3", SUPERVISOR_ADDR)
        # else:
        sock.sendto(b"RESET", SUPERVISOR_ADDR)

    def save(self, path):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "target_critic1": self.target_critic1.state_dict(),
            "target_critic2": self.target_critic2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }, path)

    def log_reward(self, step, reward, dist, actor_loss, critic_loss, q1_new, q2_new, log_prob):
        with open('logs/reward_log.txt', "a") as f:
            f.write(f"{step},{reward},{dist},{self.finish},{self.dtp},{actor_loss},{critic_loss},{q1_new},{q2_new},{log_prob}\n")

    

    def load(self, path):
        checkpoint = torch.load(path, map_location=device)

        self.actor.load_state_dict(checkpoint["actor"])
        self.critic1.load_state_dict(checkpoint["critic1"])
        self.critic2.load_state_dict(checkpoint["critic2"])
        self.target_critic1.load_state_dict(checkpoint["target_critic1"])
        self.target_critic2.load_state_dict(checkpoint["target_critic2"])

        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])

        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()
        self.target_critic1.eval()
        self.target_critic2.eval()


    def select_action_test(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            mean, log_std = self.actor(state)
            action = torch.tanh(mean)
        return action.cpu().numpy()[0]


    def step_test(self, danger, front, distance, min_left, min_right, min_lidar, lidar_info):
        if self.waiting_reset:
            self.speed = 0
            self.steering_angle = 0

            self.reset_steps -= 1
            if self.reset_steps <= 0:
                self.waiting_reset = False

            return
        
        if self.prev_distance == False:
            self.prev_distance = distance
            self.old_dist = 120

        state = self.make_state(distance, min_lidar, front)

        action = self.select_action_test(state)
        self.apply_action(action)

        reward = self.compute_reward(distance, front, min_left, min_right, min_lidar, lidar_info)


        print(
            f"test_step={self.episode_steps} | "
            f"reward={reward:.4f} | "
            f"speed={self.speed:.4f} | "
            f"steer={self.steering_angle:.4f} | "
            f"action={[round(x, 4) for x in action]}"
        )
        self.episode_reward += reward
        self.episode_steps += 1
        self.episode_speeds.append(self.speed)
        self.episode_distances.append(distance)

        if box.finish or box.dtp or self.episode_steps >= 2000:
            self.test_episode += 1

            if box.finish:
                self.test_finishes += 1
                result = "FINISH"
            else:
                self.test_failures += 1
                result = "FAIL"

            avg_speed = sum(self.episode_speeds) / len(self.episode_speeds)
            avg_distance = sum(self.episode_distances) / len(self.episode_distances)

            print("\n===== TEST EPISODE RESULT =====")
            print(f"Episode: {self.test_episode}")
            print(f"Result: {result}")
            print(f"Steps: {self.episode_steps}")
            print(f"Total reward: {self.episode_reward:.2f}")
            print(f"Avg reward: {self.episode_reward / self.episode_steps:.4f}")
            print(f"Avg speed: {avg_speed:.4f}")
            print(f"Avg distance: {avg_distance:.4f}")
            print(f"Finishes: {self.test_finishes} | Failures: {self.test_failures}")
            print("================================\n")

            log = {
                "episode": int(self.test_episode),
                "result": result,
                "steps": int(self.episode_steps),
                "total_reward": float(self.episode_reward),
                "avg_reward": float(self.episode_reward / self.episode_steps),
                "avg_speed": float(avg_speed),
                "avg_distance": float(avg_distance)
            }

            with open("test_log.json", "a") as f:
                f.write(json.dumps(log) + "\n")

            # сброс статистики эпизода
            self.episode_reward = 0
            self.episode_steps = 0
            self.episode_speeds = []
            self.episode_distances = []

            self.reset()