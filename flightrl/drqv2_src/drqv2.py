import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------------------------
# 1. Helper Functions (Weight Init & Augmentation)
# -------------------------------------------------------------------
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

class RandomShiftsAug(nn.Module):
    def __init__(self, pad=4):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)

# -------------------------------------------------------------------
# 2. Encoder (CNN)
# -------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()
        # Standard DrQ CNN architecture
        self.convs = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1), nn.ReLU()
        )
        self.head = nn.Sequential(
            nn.Linear(32 * 35 * 35, 50), 
            nn.LayerNorm(50)
        )
        self.apply(weight_init)

    def forward(self, obs):
        # Scale 0-255 to -0.5 to 0.5
        obs = obs / 255.0 - 0.5
        h = self.convs(obs)
        h = h.view(h.shape[0], -1)
        return self.head(h)

# -------------------------------------------------------------------
# 3. Actor & Critic
# -------------------------------------------------------------------
class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh()
        )
        self.policy = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_shape[0])
        )
        self.apply(weight_init)

    def forward(self, obs, stddev):
        h = self.trunk(obs)
        mu = self.policy(h)
        mu = torch.tanh(mu) # Bound to [-1, 1]
        std = torch.ones_like(mu) * stddev
        
        # Create distribution
        dist = torch.distributions.Normal(mu, std)
        return dist

class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh()
        )
        # Double Q-Learning
        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
        self.apply(weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        return self.Q1(h_action), self.Q2(h_action)

# -------------------------------------------------------------------
# 4. The Agent (Main Logic)
# -------------------------------------------------------------------
class DrQV2Agent:
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim, hidden_dim, batch_size):
        self.device = device
        self.batch_size = batch_size
        self.lr = lr
        
        # Hyperparameters
        self.critic_target_tau = 0.01
        self.update_every_steps = 2
        self.stddev_schedule = 0.1 # Constant exploration noise for simplicity
        self.stddev_clip = 0.3

        # Models
        self.encoder = Encoder(obs_shape).to(device)
        self.actor = Actor(50, action_shape, feature_dim, hidden_dim).to(device)
        self.critic = Critic(50, action_shape, feature_dim, hidden_dim).to(device)
        self.critic_target = Critic(50, action_shape, feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Augmentation
        self.aug = RandomShiftsAug(pad=4)

        # Optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, step, eval_mode):
        # obs shape: (9, 64, 64) -> need (1, 9, 64, 64)
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0).float()
        
        with torch.no_grad():
            features = self.encoder(obs)
            stddev = 0.1 # Exploration noise
            if eval_mode:
                stddev = 0.0
            
            dist = self.actor(features, stddev)
            action = dist.sample()
            
            if not eval_mode:
                action = action.clamp(-1.0, 1.0)
                
        return action.cpu().numpy()[0]

    def update(self, replay_buffer, step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample()

        # --- 1. Update Critic ---
        obs = self.aug(obs)
        next_obs = self.aug(next_obs)
        
        with torch.no_grad():
            next_features = self.encoder(next_obs)
            next_dist = self.actor(next_features, stddev=0.1)
            next_action = next_dist.sample()
            next_action = next_action.clamp(-1.0, 1.0)
            
            target_Q1, target_Q2 = self.critic_target(next_features, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (not_done * 0.99 * target_V)

        # Encode current obs
        features = self.encoder(obs)
        current_Q1, current_Q2 = self.critic(features, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        # --- 2. Update Actor ---
        features = features.detach() # Detach encoder so actor doesn't shift it
        dist = self.actor(features, stddev=0.1)
        action = dist.sample()
        action = action.clamp(-1.0, 1.0)
        
        Q1, Q2 = self.critic(features, action)
        Q = torch.min(Q1, Q2)
        actor_loss = -Q.mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        # --- 3. Soft Update Target ---
        if step % self.update_every_steps == 0:
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.critic_target_tau * param.data + (1 - self.critic_target_tau) * target_param.data)