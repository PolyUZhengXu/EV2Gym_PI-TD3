"""
Reproduction of the Transformer-LSTM + TD3 scheduling 
framework described in 
"An Optimal Scheduling Framework for Integrated Energy 
Systems Using Deep 
Reinforcement Learning and Deep Learning Prediction Models".

This script synthesizes an integrated energy system (IES) 
dataset, constructs 
Gaussian-process-based interval labels, trains a 
Transformer-LSTM predictor, 
couples it with a TD3 scheduler via a simplified synchronous 
training loop, 
and produces quantitative metrics and academic-style 
figures.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from torch import nn
from torch.utils.data import DataLoader, Dataset


# ----------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ----------------------------------------------------------
# Synthetic data generation
# ----------------------------------------------------------

class SyntheticIESDataset:
    """Generate stylized multi-energy profiles with 15 min resolution."""

    def __init__(self, days: int = 64, seed: int = 42):
        self.days = days
        self.steps_per_day = 96
        self.seed = seed

    def generate(self) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        total_steps = self.days * self.steps_per_day
        idx = np.arange(total_steps)

        day_fraction = (idx % self.steps_per_day) / self.steps_per_day
        season_fraction = idx / total_steps

        ghi = np.clip(
            np.sin(2*np.pi * (day_fraction-0.25)) * (0.65 + 0.35 * np.sin(2 * np.pi * season_fraction))
            + 0.05 * rng.standard_normal(total_steps),
            0,
            None,
        )

        wind = np.clip(
            5
            + 1.5 * np.sin(2 * np.pi * season_fraction + 1.3 * day_fraction)
            + 0.8 * rng.standard_normal(total_steps),
            0.2,
            None,
        )

        temperature = (
            20
            + 9 * np.sin(2 * np.pi * season_fraction - 0.1)
            + 6 * np.sin(2 * np.pi * day_fraction - 0.4)
            + 1.5 * rng.standard_normal(total_steps)
        )

        electric_load = (
            2.4
            + 0.9 * np.sin(2 * np.pi * day_fraction - 0.2)
            + 0.6 * np.cos(2 * np.pi * day_fraction * 3)
            + 0.35 * (25 - temperature) / 10
            + 0.2 * rng.standard_normal(total_steps)
        )
        electric_load = np.clip(electric_load, 0.5, None)

        cooling_load = np.clip((temperature - 23) * 0.35 + 0.2 * rng.standard_normal(total_steps), 0, None)
        heating_load = np.clip((18 - temperature) * 0.4 + 0.15 * rng.standard_normal(total_steps), 0, None)

        humidity = np.clip(60 + 20 * np.sin(day_fraction * 2 * np.pi + 0.5) + 10 * rng.standard_normal(total_steps), 20, 95)
        pressure = 101 + 0.4 * np.cos(2 * np.pi * season_fraction) + 0.1 * rng.standard_normal(total_steps)

        price = 0.45 + 0.15 * np.sin(2 * np.pi * day_fraction + 0.8) + 0.05 * rng.standard_normal(total_steps)
        price = np.clip(price, 0.2, 0.8)

        df = pd.DataFrame(
            {
                "step": idx,
                "ghi": ghi,
                "wind": wind,
                "temperature": temperature,
                "electric_load": electric_load,
                "heating_load": heating_load,
                "cooling_load": cooling_load,
                "humidity": humidity / 100.0,
                "pressure": pressure,
                "price": price,
            }
        )

        # Additional derived meteorological cues to mimic richer feature space.
        df["sin_time"] = np.sin(2 * np.pi * day_fraction)
        df["cos_time"] = np.cos(2 * np.pi * day_fraction)
        df["sin_season"] = np.sin(2 * np.pi * season_fraction)
        df["cos_season"] = np.cos(2 * np.pi * season_fraction)
        df["dew_point"] = df["temperature"] - (100 - humidity) / 5
        df["ghi_log"] = np.log1p(df["ghi"])

        # Equipment-dependent potentials in MWh per 15 min interval.
        pv_rated = 3.0
        df["pv_potential"] = pv_rated * df["ghi"] / (df["ghi"].max() + 1e-6)

        def wind_curve(v: float, rated: float = 2.5, v_in: float = 3.5, v_r: float = 12, v_out: float = 20) -> float:
            if v < v_in or v > v_out:
                return 0.0
            if v <= v_r:
                return rated * (v - v_in) / (v_r - v_in)
            return rated

        df["wt_potential"] = df["wind"].apply(wind_curve)
        return df


# ----------------------------------------------------------
# Gaussian-process-based interval builder
# ----------------------------------------------------------

class SlidingGPR:
    def __init__(self, series: np.ndarray, history: int = 24, max_samples: int = 400):
        self.series = series
        self.history = history
        self.max_samples = max_samples
        kernel = 1.0 * RBF(length_scale=5.0) + WhiteKernel(noise_level=0.05)
        self.model = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=0)

    def fit(self) -> None:
        X, y = [], []
        for t in range(self.history, len(self.series)):
            window = self.series[t - self.history : t]
            X.append(window)
            y.append(self.series[t])
        X = np.array(X)
        y = np.array(y)
        if len(X) > self.max_samples:
            idx = np.linspace(0, len(X) - 1, num=self.max_samples, dtype=int)
            X = X[idx]
            y = y[idx]
        self.model.fit(X, y)

    def forecast_interval(self, history_window: np.ndarray, horizon: int = 6, alpha: float = 0.95) -> np.ndarray:
        cur = history_window.copy()
        z = 1.96 if math.isclose(alpha, 0.95) else 1.0
        preds = []
        for _ in range(horizon):
            mean, std = self.model.predict(cur.reshape(1, -1), return_std=True)
            upper = mean[0] + z * std[0]
            lower = mean[0] - z * std[0]
            preds.append((lower, upper))
            cur = np.roll(cur, -1)
            cur[-1] = mean[0]
        return np.array(preds)


def build_intervals(
    df: pd.DataFrame,
    target_cols: List[str],
    history: int,
    horizon: int,
) -> Dict[str, np.ndarray]:
    intervals: Dict[str, List[np.ndarray]] = {col: [] for col in target_cols}
    models = {}
    for col in target_cols:
        model = SlidingGPR(df[col].values, history=history)
        model.fit()
        models[col] = model
    total_steps = len(df)
    for t in range(total_steps):
        for col in target_cols:
            if t < history:
                intervals[col].append(np.zeros((horizon, 2)))
                continue
            window = df[col].values[t - history : t]
            preds = models[col].forecast_interval(window, horizon=horizon)
            intervals[col].append(preds)
    return {col: np.array(seq) for col, seq in intervals.items()}


# ----------------------------------------------------------
# Prediction dataset and model
# ----------------------------------------------------------

@dataclass
class PredictionConfig:
    seq_len: int = 8
    horizon: int = 6
    feature_cols: Tuple[str, ...] = (
        "ghi",
        "wind",
        "temperature",
        "electric_load",
        "heating_load",
        "cooling_load",
        "humidity",
        "pressure",
        "price",
        "sin_time",
        "cos_time",
        "sin_season",
        "cos_season",
        "dew_point",
        "ghi_log",
    )
    target_cols: Tuple[str, ...] = ("electric_load", "heating_load", "cooling_load", "ghi", "wind")


class PredictionDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        intervals: Dict[str, np.ndarray],
        cfg: PredictionConfig,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
    ):
        self.cfg = cfg
        self.df = df
        self.intervals = intervals
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.samples = self._build_samples()

    def _build_samples(self) -> List[int]:
        min_idx = max(self.cfg.seq_len, 24)
        max_idx = len(self.df) - self.cfg.horizon
        return list(range(min_idx, max_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        t = self.samples[idx]
        seq = self.df.iloc[t - self.cfg.seq_len : t][list(self.cfg.feature_cols)].values.astype(np.float32)
        seq = (seq - self.feature_mean) / (self.feature_std + 1e-6)

        targets = []
        for col in self.cfg.target_cols:
            targets.append(self.intervals[col][t][: self.cfg.horizon])
        target = np.stack(targets, axis=1)  # (horizon, target_dim, 2)
        return torch.from_numpy(seq), torch.from_numpy(target.astype(np.float32))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class TransformerLSTM(nn.Module):
    def __init__(self, cfg: PredictionConfig, d_model: int = 64, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Linear(len(cfg.feature_cols), d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dim_feedforward=128)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos = PositionalEncoding(d_model)
        self.lstm = nn.LSTM(d_model, d_model, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, cfg.horizon * len(cfg.target_cols) * 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.embedding(x)
        z = self.pos(z)
        z = self.encoder(z)
        z, _ = self.lstm(z)
        context = z[:, -1, :]
        out = self.head(context)
        out = out.view(-1, self.cfg.horizon, len(self.cfg.target_cols), 2)
        lower = torch.min(out[..., 0], out[..., 1])
        upper = torch.max(out[..., 0], out[..., 1])
        return torch.stack((lower, upper), dim=-1)


def train_predictor(
    model: TransformerLSTM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 25,
    lr: float = 1e-3,
) -> Dict[str, List[float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_loss": []}
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for seq, target in train_loader:
            seq = seq.to(device)
            target = target.to(device)
            pred = model(seq)
            loss = criterion(pred[..., 0], target[..., 0]) + criterion(pred[..., 1], target[..., 1])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * seq.size(0)
        train_loss /= len(train_loader.dataset)
        history["train_loss"].append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for seq, target in val_loader:
                seq = seq.to(device)
                target = target.to(device)
                pred = model(seq)
                loss = criterion(pred[..., 0], target[..., 0]) + criterion(pred[..., 1], target[..., 1])
                val_loss += loss.item() * seq.size(0)
        val_loss /= len(val_loader.dataset)
        history["val_loss"].append(val_loss)
        print(f"[Predictor] Epoch {epoch+1}/{epochs} | Train {train_loss:.4f} | Val {val_loss:.4f}")
    return history


# ----------------------------------------------------------
# TD3 components
# ----------------------------------------------------------

class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, capacity: int = 100_000):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((capacity, 1), dtype=np.float32)
        self.next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, s, a, r, s_next, d):
        self.state[self.ptr] = s
        self.action[self.ptr] = a
        self.reward[self.ptr] = r
        self.next_state[self.ptr] = s_next
        self.done[self.ptr] = d
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.state[idx]),
            torch.from_numpy(self.action[idx]),
            torch.from_numpy(self.reward[idx]),
            torch.from_numpy(self.next_state[idx]),
            torch.from_numpy(self.done[idx]),
        )


def mlp(input_dim: int, output_dim: int, hidden: Tuple[int, ...] = (128, 128), act_last: nn.Module | None = None) -> nn.Sequential:
    layers: List[nn.Module] = []
    prev = input_dim
    for h in hidden:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    if act_last:
        layers.append(act_last)
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float = 1.0):
        super().__init__()
        self.net = mlp(state_dim, action_dim, hidden=(256, 256), act_last=nn.Tanh())
        self.max_action = max_action

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) * self.max_action


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.q1 = mlp(state_dim + action_dim, 1, hidden=(256, 256))
        self.q2 = mlp(state_dim + action_dim, 1, hidden=(256, 256))

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)

    def q1_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa)


class TD3Agent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float = 1.0,
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        tau: float = 0.005,
        gamma: float = 0.99,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_freq: int = 2,
        device: torch.device | None = None,
    ):
        self.device = device or torch.device("cpu")
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.max_action = max_action
        self.tau = tau
        self.gamma = gamma
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.total_it = 0

    def act(self, state: np.ndarray, noise: float = 0.0) -> np.ndarray:
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        action = self.actor(state_t).cpu().data.numpy().flatten()
        if noise > 0:
            action = action + np.random.normal(0, noise, size=action.shape)
        return np.clip(action, -self.max_action, self.max_action)

    def train(self, replay: ReplayBuffer, batch_size: int = 128):
        if replay.size < batch_size:
            return
        self.total_it += 1
        state, action, reward, next_state, done = replay.sample(batch_size)
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1 - done) * self.gamma * target_q

        current_q1, current_q2 = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic.q1_forward(state, self.actor(state)).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


# ----------------------------------------------------------
# Integrated Energy System environment
# ----------------------------------------------------------

@dataclass
class IESConfig:
    seq_len: int = 8
    horizon: int = 6
    hydrogen_capacity: float = 12.0
    electrolysis_max: float = 1.6
    electrolysis_eff: float = 0.7
    fuel_cell_max: float = 1.2
    fuel_cell_elec_eff: float = 0.55
    fuel_cell_heat_eff: float = 0.35
    hp_elec_max: float = 1.5
    hp_cop_heat: float = 3.0
    hp_cop_cool: float = 2.4
    ac_cop: float = 0.75
    reward_sigmoid_cost_c: Tuple[float, float] = (0.5, 0.12)
    reward_sigmoid_co2_c: Tuple[float, float] = (0.25, 0.05)
    grid_heat_price: float = 0.12
    hydrogen_value: float = 0.08
    hydrogen_penalty: float = 0.02
    carbon_coeff_e: float = 0.42
    carbon_coeff_h: float = 0.27


class IESEnvironment:
    def __init__(
        self,
        df: pd.DataFrame,
        predictor: TransformerLSTM,
        cfg: IESConfig,
        pred_cfg: PredictionConfig,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        device: torch.device,
    ):
        self.df = df.reset_index(drop=True)
        self.predictor = predictor
        self.cfg = cfg
        self.pred_cfg = pred_cfg
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.device = device
        self.steps_per_day = 96
        self.train_days = list(range(0, 40))
        self.test_days = list(range(40, 56))
        self.current_indices: List[int] = []
        self.history_buffer: List[np.ndarray] = []
        self.hydrogen_level = cfg.hydrogen_capacity * 0.6
        self.day_pointer = 0
        self.info_log: List[Dict[str, float]] = []

    def _init_episode(self, day_idx: int) -> None:
        start = day_idx * self.steps_per_day
        self.current_indices = list(range(start, start + self.steps_per_day))
        self.history_buffer = []
        for t in self.current_indices[: self.cfg.seq_len]:
            feats = self.df.iloc[t][list(self.pred_cfg.feature_cols)].values.astype(np.float32)
            norm = (feats - self.feature_mean) / (self.feature_std + 1e-6)
            self.history_buffer.append(norm)
        self.hydrogen_level = 0.5 * self.cfg.hydrogen_capacity
        self.info_log = []

    def reset(self, split: str = "train") -> np.ndarray:
        day_pool = self.train_days if split == "train" else self.test_days
        day_idx = random.choice(day_pool)
        self._init_episode(day_idx)
        self.pointer = self.cfg.seq_len
        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        seq = np.stack(self.history_buffer[-self.cfg.seq_len :], axis=0)
        seq_t = torch.from_numpy(seq).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            preds = self.predictor(seq_t).cpu().numpy().reshape(-1)
        obs = np.concatenate([seq.flatten(), preds, np.array([self.hydrogen_level / self.cfg.hydrogen_capacity])])
        return obs.astype(np.float32)

    def _sigmoid_indicator(self, value: float, c: Tuple[float, float]) -> float:
        center, scale = c
        return 1 / (1 + math.exp(-(value - center) / (scale + 1e-6)))

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, float]]:
        idx = self.current_indices[self.pointer]
        row = self.df.iloc[idx]
        action = np.clip(action, -1, 1)
        pv_ratio = 0.5 * (action[0] + 1)
        wt_ratio = 0.5 * (action[1] + 1)
        el_ratio = 0.5 * (action[2] + 1)
        hp_heat_ratio = 0.5 * (action[3] + 1)
        hp_cool_ratio = 0.5 * (action[4] + 1)
        fc_ratio = 0.5 * (action[5] + 1)

        pv_avail = row["pv_potential"]
        wt_avail = row["wt_potential"]
        pv_used = pv_ratio * pv_avail
        wt_used = wt_ratio * wt_avail

        fc_power = fc_ratio * self.cfg.fuel_cell_max
        h2_needed = fc_power / max(self.cfg.fuel_cell_elec_eff, 1e-6)
        if h2_needed > self.hydrogen_level:
            fc_power = self.hydrogen_level * self.cfg.fuel_cell_elec_eff
            h2_needed = self.hydrogen_level
        self.hydrogen_level -= h2_needed
        fc_heat = h2_needed * self.cfg.fuel_cell_heat_eff

        el_power = el_ratio * self.cfg.electrolysis_max
        h2_prod = el_power * self.cfg.electrolysis_eff
        self.hydrogen_level = min(self.cfg.hydrogen_capacity, self.hydrogen_level + h2_prod)

        hp_heat_elec = hp_heat_ratio * self.cfg.hp_elec_max
        hp_cool_elec = hp_cool_ratio * self.cfg.hp_elec_max
        hp_heat = hp_heat_elec * self.cfg.hp_cop_heat
        hp_cool = hp_cool_elec * self.cfg.hp_cop_cool

        elec_load = row["electric_load"]
        heat_load = row["heating_load"]
        cool_load = row["cooling_load"]

        electric_demand = elec_load + hp_heat_elec + hp_cool_elec + el_power
        renewable_elec = pv_used + wt_used + fc_power
        grid = max(0.0, electric_demand - renewable_elec)
        waste_elec = max(0.0, renewable_elec - electric_demand)

        heat_from_hp = min(heat_load, hp_heat)
        remaining_heat = heat_load - heat_from_hp
        heat_from_fc = min(remaining_heat, fc_heat)
        heat_grid = max(0.0, remaining_heat - heat_from_fc)
        heat_waste = max(0.0, (hp_heat + fc_heat) - heat_load)

        heat_for_ac = max(fc_heat - heat_from_fc, 0.0)
        cool_from_hp = min(cool_load, hp_cool)
        cool_from_ac = min(max(cool_load - cool_from_hp, 0), heat_for_ac * self.cfg.ac_cop)
        cool_deficit = max(0.0, cool_load - cool_from_hp - cool_from_ac)

        price = row["price"]
        cost = grid * price + heat_grid * self.cfg.grid_heat_price
        cost += self.cfg.hydrogen_penalty * h2_needed - self.cfg.hydrogen_value * h2_prod

        co2 = grid * self.cfg.carbon_coeff_e + heat_grid * self.cfg.carbon_coeff_h
        imbalance = (
            abs(waste_elec - grid)
            + abs(heat_waste - heat_grid)
            + cool_deficit
            + abs(h2_prod - h2_needed)
        ) / (elec_load + heat_load + cool_load + 1e-5)

        renewable_util = (pv_used + wt_used) / (pv_avail + wt_avail + 1e-6)

        c_star = self._sigmoid_indicator(cost, self.cfg.reward_sigmoid_cost_c)
        e_star = self._sigmoid_indicator(co2, self.cfg.reward_sigmoid_co2_c)
        I = np.array([c_star, e_star, imbalance, renewable_util])
        ideal = np.array([0.0, 0.0, 0.0, 1.0])
        reward = -np.linalg.norm(I - ideal)

        info = {
            "cost": cost,
            "co2": co2,
            "imbalance": imbalance,
            "renewable": renewable_util,
            "grid": grid,
            "pv_used": pv_used,
            "wt_used": wt_used,
            "fc_power": fc_power,
            "hydrogen_level": self.hydrogen_level,
            "electric_load": elec_load,
            "heat_grid": heat_grid,
            "cool_deficit": cool_deficit,
            "reward": reward,
        }
        self.info_log.append(info)

        self.history_buffer.append(
            (row[list(self.pred_cfg.feature_cols)].values.astype(np.float32) - self.feature_mean)
            / (self.feature_std + 1e-6)
        )
        self.pointer += 1
        done = self.pointer >= len(self.current_indices)
        next_state = self._get_observation() if not done else self.reset(split="train")
        return next_state, reward, done, info


# ----------------------------------------------------------
# Synchronous training loop
# ----------------------------------------------------------

def synchronous_training(
    env: IESEnvironment,
    agent: TD3Agent,
    dataset: PredictionDataset,
    predictor: TransformerLSTM,
    device: torch.device,
    episodes: int = 32,
    warmup_steps: int = 300,
    batch_size: int = 128,
) -> Tuple[List[float], Dict[str, float]]:
    replay = ReplayBuffer(state_dim=env._get_observation().shape[0], action_dim=6, capacity=50_000)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    predictor_opt = torch.optim.Adam(predictor.parameters(), lr=5e-4)
    criterion = nn.MSELoss()
    reward_history: List[float] = []
    best_reward = -1e9
    worst_reward = 1e9
    step_count = 0

    for ep in range(episodes):
        state = env.reset(split="train")
        episode_reward = 0.0
        for _ in range(env.steps_per_day):
            noise = max(0.1, 0.5 * (1 - ep / episodes))
            if step_count < warmup_steps:
                action = np.random.uniform(-1, 1, size=6)
            else:
                action = agent.act(state, noise=noise)
            next_state, reward, done, info = env.step(action)
            replay.add(state, action, reward, next_state, float(done))
            agent.train(replay, batch_size=batch_size)
            state = next_state
            episode_reward += reward
            step_count += 1
            if done:
                break

        reward_history.append(episode_reward)
        best_reward = max(best_reward, episode_reward)
        worst_reward = min(worst_reward, episode_reward)
        print(f"[RL] Episode {ep+1}/{episodes} | Reward {episode_reward:.3f}")

        # Synchronous fine-tuning of predictor using reward feedback.
        predictor.train()
        seq_batch, target_batch = next(iter(train_loader))
        seq_batch = seq_batch.to(device)
        target_batch = target_batch.to(device)
        pred = predictor(seq_batch)
        pred_loss = criterion(pred[..., 0], target_batch[..., 0]) + criterion(pred[..., 1], target_batch[..., 1])
        if best_reward > worst_reward + 1e-6:
            norm_term = (best_reward - episode_reward) / (best_reward - worst_reward + 1e-6)
        else:
            norm_term = 0.0
        total_loss = 0.8 * pred_loss + 0.2 * torch.tensor(norm_term, dtype=torch.float32, device=device)
        predictor_opt.zero_grad()
        total_loss.backward()
        predictor_opt.step()

    stats = {
        "best_reward": best_reward,
        "worst_reward": worst_reward,
        "avg_reward": float(np.mean(reward_history[-5:])),
    }
    return reward_history, stats


# ----------------------------------------------------------
# Evaluation utilities
# ----------------------------------------------------------

def evaluate_policy(
    env: IESEnvironment,
    agent: TD3Agent,
    split: str = "test",
) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
    metrics = {"reward": [], "cost": [], "co2": [], "imbalance": [], "renewable": [], "grid": []}
    trajectories: Dict[str, List[float]] = {
        "pv_used": [],
        "wt_used": [],
        "grid": [],
        "fc_power": [],
        "hydrogen_level": [],
        "cost": [],
        "reward": [],
        "electric_load": [],
    }
    for _ in range(8):
        state = env.reset(split=split)
        episode_reward = 0.0
        for step in range(env.steps_per_day):
            action = agent.act(state, noise=0.0)
            next_state, reward, done, info = env.step(action)
            state = next_state
            episode_reward += reward
            for key in metrics:
                metrics[key].append(info.get(key, 0.0))
            if len(trajectories["pv_used"]) < env.steps_per_day:
                for k in trajectories:
                    trajectories[k].append(info.get(k, 0.0))
            if done:
                break
        metrics["reward"].append(episode_reward)
    summary = {k: float(np.mean(v)) for k, v in metrics.items()}
    return summary, trajectories


def plot_rewards(reward_curve: List[float], output_path: str) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(reward_curve, label="Episode reward", color="#1f77b4")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_typical_day(trajectory: Dict[str, List[float]], output_path: str) -> None:
    steps = np.arange(len(trajectory["pv_used"]))
    pv = np.array(trajectory["pv_used"])
    wt = np.array(trajectory["wt_used"])
    fc = np.array(trajectory["fc_power"])
    load = np.array(trajectory.get("electric_load", np.zeros_like(pv)))

    grid = np.array(trajectory["grid"])
    cost = np.array(trajectory["cost"])
    reward = np.array(trajectory["reward"])
    hydrogen = np.array(trajectory["hydrogen_level"])

    cost_norm = cost / (np.max(cost) + 1e-6)
    hydrogen_soc = hydrogen / (np.max(hydrogen) + 1e-6)

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    axes[0].plot(steps, load, label="Electric load", color="black", linewidth=1.6)
    axes[0].stackplot(steps, pv, wt, fc, labels=["PV", "WT", "FC"], colors=["#2ca02c", "#17becf", "#9467bd"], alpha=0.8)
    axes[0].set_ylabel("Power (MW)")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.3)

    axes[1].plot(steps, grid, label="Grid import (MW)", color="#d62728")
    axes[1].plot(steps, cost_norm, label="Cost (normalized)", color="#1f77b4", linestyle="--")
    axes[1].set_ylabel("Per-unit")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper right")

    axes[2].plot(steps, hydrogen_soc, label="Hydrogen SOC", color="#ff7f0e")
    axes[2].plot(steps, reward, label="Reward", color="#4c72b0")
    axes[2].set_ylabel("Per-unit / Reward")
    axes[2].set_xlabel("15-min step")
    axes[2].grid(alpha=0.3)
    axes[2].legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_prediction_interval(
    predictor: TransformerLSTM,
    dataset: PredictionDataset,
    df: pd.DataFrame,
    cfg: PredictionConfig,
    device: torch.device,
    output_path: str,
) -> None:
    predictor.eval()
    idx = random.randint(0, len(dataset) - 1)
    seq, target = dataset[idx]
    with torch.no_grad():
        pred = predictor(seq.unsqueeze(0).to(device)).cpu().numpy()[0]
    horizon = cfg.horizon
    t_start = dataset.samples[idx]
    actual = []
    for col in cfg.target_cols:
        actual.append(df.iloc[t_start : t_start + horizon][col].values)
    actual = np.stack(actual, axis=1)

    plt.figure(figsize=(8, 5))
    for i, col in enumerate(cfg.target_cols[:3]):
        plt.subplot(3, 1, i + 1)
        plt.plot(actual[:, i], label="Actual", color="black")
        plt.fill_between(
            np.arange(horizon),
            pred[:horizon, i, 0],
            pred[:horizon, i, 1],
            color="#1f77b4",
            alpha=0.3,
            label="Predicted interval",
        )
        plt.ylabel(col)
        plt.grid(alpha=0.3)
        if i == 0:
            plt.legend()
    plt.xlabel("Prediction step")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# ----------------------------------------------------------
# Main orchestration
# ----------------------------------------------------------

def main() -> None:
    set_seed(7)
    device = torch.device("cpu")
    ensure_dir("results")

    print("Generating synthetic dataset...")
    df = SyntheticIESDataset(days=56).generate()

    pred_cfg = PredictionConfig()
    feature_array = df[list(pred_cfg.feature_cols)].values.astype(np.float32)
    feature_mean = feature_array.mean(axis=0)
    feature_std = feature_array.std(axis=0)

    print("Building GPR-based interval targets...")
    intervals = build_intervals(df, list(pred_cfg.target_cols), history=24, horizon=pred_cfg.horizon)

    dataset = PredictionDataset(df, intervals, pred_cfg, feature_mean, feature_std)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(0))
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)

    predictor = TransformerLSTM(pred_cfg).to(device)
    print("Training Transformer-LSTM predictor...")
    train_predictor(predictor, train_loader, val_loader, device=device, epochs=20)

    env_cfg = IESConfig(seq_len=pred_cfg.seq_len, horizon=pred_cfg.horizon)
    env = IESEnvironment(
        df=df,
        predictor=predictor,
        cfg=env_cfg,
        pred_cfg=pred_cfg,
        feature_mean=feature_mean,
        feature_std=feature_std,
        device=device,
    )

    sample_obs = env.reset("train")
    state_dim = sample_obs.shape[0]
    action_dim = 6
    agent = TD3Agent(state_dim=state_dim, action_dim=action_dim, device=device)

    print("Coupled TD3 training with synchronous predictor updates...")
    rewards, sync_stats = synchronous_training(
        env=env,
        agent=agent,
        dataset=dataset,
        predictor=predictor,
        device=device,
        episodes=64,
        warmup_steps=300,
        batch_size=64,
    )

    print("Evaluating learned policy on held-out days...")
    metrics, trajectory = evaluate_policy(env, agent, split="test")

    ensure_dir("results")
    plot_rewards(rewards, os.path.join("results", "reward_curve.png"))
    plot_typical_day(trajectory, os.path.join("results", "typical_day.png"))
    plot_prediction_interval(
        predictor,
        dataset,
        df,
        pred_cfg,
        device,
        os.path.join("results", "prediction_interval.png"),
    )

    summary = {
        "training_rewards": rewards,
        "sync_stats": sync_stats,
        "test_metrics": metrics,
    }
    with open(os.path.join("results", "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=== Evaluation Summary ===")
    for k, v in metrics.items():
        print(f"{k:>10s}: {v:.4f}")
    print("Figures saved in ./results/")


if __name__ == "__main__":
    main()