"""
Реализация PPO (Proximal Policy Optimization) агента для обучения с подкреплением.
Содержит нейронные сети актора и критика, буферы воспроизведения и методы обучения.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# =============================================================================
# ПАРАМЕТРЫ АЛГОРИТМА
# =============================================================================
ACTION_DIM = 2
HIDDEN_DIM = 256

GAMMA = 0.99
LAMBDA = 0.95
EPS_CLIP = 0.2
VALUE_LOSS_COEF = 0.5
ENTROPY_COEF = 0.01
LR_ACTOR = 3e-4
LR_CRITIC = 1e-3
BATCH_SIZE = 64
EPOCHS = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    """Буфер воспроизведения для хранения опыта."""

    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)


class RolloutBuffer:
    """Буфер для хранения данных текущего эпизода."""

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def __len__(self):
        return len(self.states)


class ActorNetwork(nn.Module):
    """Сеть актора - определяет политику (какое действие выбрать)."""

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
        )
        self.mean = nn.Linear(HIDDEN_DIM, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        x = self.net(state)
        mean = self.mean(x)
        std = self.log_std.exp().clamp(min=0.01, max=1.0)
        return mean, std

    def get_action(self, state):
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob

    def evaluate(self, state, action):
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return log_prob, entropy


class CriticNetwork(nn.Module):
    """Сеть критика - оценивает ценность состояния."""

    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 1)
        )

    def forward(self, state):
        return self.net(state)


class PPOAgent:
    """PPO агент для обучения с подкреплением."""

    def __init__(self):
        self.state_dim = None
        self.actor = None
        self.critic = None
        self.actor_optimizer = None
        self.critic_optimizer = None

        self.buffer = RolloutBuffer()
        self.memory = ReplayBuffer()

        self.step_count = 0
        self.initialized = False

        print("PPO Agent initialized (waiting for first state)")
        print(f"   Device: {device}")

    def _initialize_networks(self, state_dim):
        """Инициализация нейронных сетей с заданной размерностью состояния."""
        if self.initialized:
            return

        self.state_dim = state_dim
        self.actor = ActorNetwork(state_dim, ACTION_DIM).to(device)
        self.critic = CriticNetwork(state_dim).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        self.initialized = True
        print(f"Networks initialized with state_dim={state_dim}")

    def select_action(self, state):
        """
        Выбор действия на основе текущего состояния.

        Args:
            state: Вектор состояния

        Returns:
            numpy.ndarray: Действие [steering, speed]
        """
        self._initialize_networks(len(state))

        if len(state.shape) == 1:
            state = state.reshape(1, -1)

        state_tensor = torch.FloatTensor(state).to(device)

        with torch.no_grad():
            action, log_prob = self.actor.get_action(state_tensor)
            value = self.critic(state_tensor)

        action_np = action.cpu().numpy()[0]
        log_prob_np = log_prob.cpu().numpy()[0][0]
        value_np = value.cpu().numpy()[0][0]

        self.buffer.states.append(state[0])
        self.buffer.actions.append(action_np)
        self.buffer.log_probs.append(log_prob_np)
        self.buffer.values.append(value_np)

        return action_np

    def _compute_gae(self, rewards, values, dones, next_value):
        """
        Вычисление Generalized Advantage Estimation (GAE).

        Returns:
            tuple: (advantages, returns)
        """
        advantages = []
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                delta = rewards[t] + GAMMA * next_value * (1 - dones[t]) - values[t]
            else:
                delta = rewards[t] + GAMMA * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + GAMMA * LAMBDA * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        returns = [adv + val for adv, val in zip(advantages, values)]
        return advantages, returns

    def train(self):
        """Один шаг обучения PPO на собранных данных."""
        if not self.initialized:
            return

        if len(self.buffer) < BATCH_SIZE:
            return

        if len(self.buffer.states) == 0:
            return

        try:
            states = torch.FloatTensor(np.array(self.buffer.states)).to(device)
            actions = torch.FloatTensor(np.array(self.buffer.actions)).to(device)
            old_log_probs = torch.FloatTensor(np.array(self.buffer.log_probs)).to(device)
            old_values = torch.FloatTensor(np.array(self.buffer.values)).to(device)
            rewards = torch.FloatTensor(np.array(self.buffer.rewards)).to(device)
            dones = torch.FloatTensor(np.array(self.buffer.dones)).to(device)

            if len(states) == 0:
                return

            with torch.no_grad():
                next_state = states[-1:]
                next_value = self.critic(next_state)

            advantages, returns = self._compute_gae(rewards, old_values, dones, next_value.item())
            advantages = torch.FloatTensor(advantages).to(device)
            returns = torch.FloatTensor(returns).to(device)

            if advantages.std() > 1e-8:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            for _ in range(EPOCHS):
                indices = np.random.permutation(len(self.buffer))

                for start in range(0, len(self.buffer), BATCH_SIZE):
                    end = start + BATCH_SIZE
                    batch_indices = indices[start:end]

                    if max(batch_indices) >= len(states):
                        continue

                    batch_states = states[batch_indices]
                    batch_actions = actions[batch_indices]
                    batch_old_log_probs = old_log_probs[batch_indices]
                    batch_advantages = advantages[batch_indices]
                    batch_returns = returns[batch_indices]

                    log_probs, entropy = self.actor.evaluate(batch_states, batch_actions)
                    ratio = (log_probs - batch_old_log_probs).exp()
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP) * batch_advantages
                    actor_loss = -torch.min(surr1, surr2).mean() - ENTROPY_COEF * entropy.mean()

                    values = self.critic(batch_states)
                    critic_loss = VALUE_LOSS_COEF * nn.MSELoss()(values, batch_returns)

                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                    self.actor_optimizer.step()

                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                    self.critic_optimizer.step()

            self.buffer.clear()
            self.step_count += 1

        except Exception as e:
            print(f"Training error: {e}")
            self.buffer.clear()

    def save(self, path):
        """Сохранение весов нейронных сетей."""
        if not self.initialized:
            print("Agent not initialized, cannot save")
            return
        torch.save({
            "state_dim": self.state_dim,
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }, path)
        print(f"Model saved to {path}")

    def load(self, path):
        """Загрузка весов нейронных сетей."""
        checkpoint = torch.load(path, map_location=device)
        state_dim = checkpoint["state_dim"]
        self._initialize_networks(state_dim)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        print(f"Model loaded from {path}")