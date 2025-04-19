import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
#from IPython.display import clear_output
import os

# Environment parameters
TARGET_POSITION = 0.45
POWER = 0.0015
GRAVITY = 0.0025

# Environment setup - NO RENDERING
env = gym.make('MountainCarContinuous-v0')


# Initialize plots
plt.ion()  # Interactive mode
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
fig.suptitle('Training Performance')

# Data storage for plotting
position_errors = []
loss_history = []

def update_plots():
    # Remove the clear command as it can interfere
    ax1.clear()
    ax2.clear()
    
    ax1.plot(position_errors, 'b-')
    ax1.set_title('Position Error (x - target)')
    ax1.axhline(0, color='r', linestyle='--')  # Add target line
    ax1.set_ylabel('Error')
    ax1.grid(True)
    
    ax2.plot(loss_history, 'r-')
    ax2.set_title('Neural Network Training Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Training Steps')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.draw()
    plt.pause(0.01)  # REQUIRED for live updates
    
class iLQRController:
    def __init__(self, dynamics_model, horizon=50, max_iter=10, Q_terminal=np.diag([1000, 0]), R=0.01):
        self.horizon = horizon
        self.max_iter = max_iter
        self.Q_terminal = Q_terminal
        self.R = R * np.eye(1)
        self.dynamics_model = dynamics_model
        
    def dynamics(self, x, u):
        """ Uses the learned dynamics model instead of known equations. """
        x_next = self.dynamics_model.predict(x, u)
        A, B = self.dynamics_model.get_jacobians(x, u)
        return x_next, A, B

    def compute_trajectory(self, x0, u_seq):
        x_seq = [x0]
        for u in u_seq:
            x_next = self.dynamics_model.predict(x_seq[-1], u)
            x_seq.append(x_next)
        return np.array(x_seq)

    def backward_pass(self, x_seq, u_seq, A_list, B_list):
        Vx = self.Q_terminal @ (x_seq[-1] - [TARGET_POSITION, 0])
        Vxx = self.Q_terminal
        k = np.zeros((self.horizon, 1))
        K = np.zeros((self.horizon, 1, 2))

        for t in reversed(range(self.horizon)):
            A, B = A_list[t], B_list[t]
            Qx = A.T @ Vx
            Qu = B.T @ Vx + self.R @ u_seq[t]
            Qxx = A.T @ Vxx @ A 
            Quu = B.T @ Vxx @ B + self.R
            Qux = B.T @ Vxx @ A

            Quu_inv = np.linalg.inv(Quu)
            k[t] = -Quu_inv @ Qu
            K[t] = -Quu_inv @ Qux

            Vx = Qx + K[t].T @ Quu @ k[t] + K[t].T @ Qu
            Vxx = Qxx + K[t].T @ Quu @ K[t] + K[t].T @ Qux + Qux.T @ K[t]

        return k, K

    def forward_pass(self, x0, u_seq, k, K):
        new_u = np.zeros_like(u_seq)
        x = x0.copy()
        new_x = [x0]
        total_cost = 0

        for t in range(self.horizon):
            new_u[t] = u_seq[t] + k[t] + K[t] @ (x - new_x[t])
            new_u[t] = np.clip(new_u[t], -1, 1)
            x = self.dynamics_model.predict(x, new_u[t])
            new_x.append(x)
            
            total_cost += 0.5 * new_u[t].T @ self.R @ new_u[t]

        total_cost += 0.5 * (x - [TARGET_POSITION, 0]).T @ self.Q_terminal @ (x - [TARGET_POSITION, 0])
        return np.array(new_x), new_u, total_cost

    def optimize(self, x0, u_guess):
        u_seq = u_guess
        x_seq = self.compute_trajectory(x0, u_seq)

        for _ in range(self.max_iter):
            A_list, B_list = [], []
            for t in range(self.horizon):
                _, A, B = self.dynamics(x_seq[t], u_seq[t])
                A_list.append(A)
                B_list.append(B)

            k, K = self.backward_pass(x_seq, u_seq, A_list, B_list)
            new_x, new_u, new_cost = self.forward_pass(x0, u_seq, k, K)
            x_seq, u_seq = new_x[:-1], new_u

        return u_seq

class DynamicsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        
    def forward(self, x):
        return self.net(x)
    
    def predict(self, x, u):
        x_flat = np.asarray(x).flatten()
        u_flat = np.asarray(u).flatten()
        inp = torch.tensor(np.hstack([x_flat, u_flat]), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return self.net(inp).squeeze(0).numpy()
    
    def update(self, x, u, x_next_true):
        x = np.asarray(x).flatten()
        u = np.asarray(u).flatten()
        x_next_true = np.asarray(x_next_true).flatten()
        
        # Convert to tensors
        xu = torch.tensor(np.hstack([x, u]), dtype=torch.float32)
        y_true = torch.tensor(x_next_true, dtype=torch.float32)
        
        # Training step
        self.optimizer.zero_grad()
        y_pred = self.net(xu.unsqueeze(0)).squeeze(0)
        loss = self.loss_fn(y_pred, y_true)
        loss.backward()
        self.optimizer.step()
        
        # Store for plotting
        loss_history.append(loss.item())
        position_errors.append(x_next_true[0] - TARGET_POSITION)
        
        # Update plots every 20 steps
        if len(loss_history) % 20 == 0:
            update_plots()
    
    def get_jacobians(self, x, u):
        xu = torch.tensor(np.hstack([x, u]), dtype=torch.float32, requires_grad=True)
        output = self.net(xu.unsqueeze(0)).squeeze(0)
        
        jacobian = torch.zeros(2, 3)
        for i in range(2):
            grad = torch.autograd.grad(output[i], xu, retain_graph=True)[0]
            jacobian[i] = grad
            
        A = jacobian[:, :2].detach().numpy()
        B = jacobian[:, 2:].detach().numpy()
        return A, B

# Rest of your iLQRController class remains the same...

def main():
    dynamics_model = DynamicsModel()
    controller = iLQRController(dynamics_model)
    
    obs, _ = env.reset()
    
    #total_reward = 0
    
    #u_guess = np.random.uniform(-1, 1, (controller.horizon, 1))  # Better initialization
    
        # Improved Exploration Phase
    exploration_steps = 200  # More exploration steps
    for _ in range(exploration_steps):
        # Biased exploration - more positive actions when in valley
        if obs[0] < -0.5:
            action = np.random.uniform(0.8, 1, size=(1,))
        else:
            action = np.random.uniform(-1, 1, size=(1,))
            
        next_obs, _, _, _, _ = env.step(action)
        dynamics_model.update(obs, action, next_obs)
        obs = next_obs

    
    """for episode in range(100):
        obs, _ = env.reset()
        for step in range(200):
            x0 = np.array(obs, dtype=np.float32).flatten()
            u_opt = controller.optimize(x0, np.random.uniform(-1, 1, (controller.horizon, 1)))
            action = np.clip(u_opt[0], -1, 1)
            next_obs, reward, terminated, _, _ = env.step(action)
            
            dynamics_model.update(obs, action, next_obs)
            obs = next_obs
            
            if terminated:
                print(f"Episode {episode} succeeded in {step} steps!")
                break"""
    for episode in range(100):
        obs, _ = env.reset()
        total_reward = 0
        u_guess = np.random.uniform(-1, 1, (controller.horizon, 1))
        
        for _ in range(300):  # Episode length
            x0 = np.array(obs, dtype=np.float32).flatten()
            u_opt = controller.optimize(x0, u_guess)    
            action = np.clip(u_opt[0], -1, 1)
            next_obs, reward, terminated, _, _ = env.step(action)
            total_reward += reward
            
            dynamics_model.update(np.array(obs), action, next_obs)
            obs = next_obs
            u_guess[:-1] = np.clip(u_opt[1:], -1, 1)
            u_guess[-1] = action
            
            if terminated:
                print(f"Episode {episode}: Target reached! Total Reward: {total_reward}")
                break
    
    env.close()
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
