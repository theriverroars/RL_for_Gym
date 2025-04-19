#Uses neural netwrork for dynamics. Jacobians are calculated manually.
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Environment parameters
TARGET_POSITION = 0.45
POWER = 0.0015
GRAVITY = 0.0025

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
        print("Shape of A:", A.shape, "Shape of B:", B.shape)
        # Approximate linearization
        #AL = np.array([[1, 1], 
        #             [3 * GRAVITY * np.sin(3 * x[0]), 1]])
        
        #B = np.array([[0], [POWER]])
        
        return x_next, A, B

    def compute_trajectory(self, x0, u_seq):
        x_seq = [x0]
        for u in u_seq:
            x_next = self.dynamics_model.predict(x_seq[-1], u)[0] #self.dynamics(x_seq[-1], u)[0]
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
            x = self.dynamics_model.predict(x, new_u[t])#x, _, _ = self.dynamics(x, new_u[t])
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
                A, B = self.dynamics_model.get_jacobians(x_seq[t], u_seq[t])#_, A, B = self.dynamics(x_seq[t], u_seq[t])
                A_list.append(A)
                B_list.append(B)

            k, K = self.backward_pass(x_seq, u_seq, A_list, B_list)
            new_x, new_u, new_cost = self.forward_pass(x0, u_seq, k, K)
            x_seq, u_seq = new_x[:-1], new_u

        return u_seq
    
    
class DynamicsModel(nn.Module):
    """ Neural network to approximate the unknown dynamics. """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )  # Fixed missing closing parenthesis
        
        self.trained = False
        self.X_train = []
        self.y_train = []
        self.losses = []
        self.nn_errors = []
        self.actual_errors = []
        self.nn_vs_actual_diff = []
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        
    def forward(self, x):
        return self.net(x)
    
    def actual_dynamics(self, x, u):
        pos, vel = x
        # Handle both scalar and array inputs for u
        u_val = u[0] if isinstance(u, (np.ndarray, list)) else u
        new_vel = vel + u_val * POWER - GRAVITY * np.cos(3 * pos)
        new_pos = pos + new_vel
        return np.array([new_pos, new_vel])
    
    def predict(self, x, u):
        x_flat = np.asarray(x).flatten()
        u_flat = np.asarray(u).flatten()
        print("x_flat shape:", x_flat.shape, "u_flat shape:", u_flat.shape) 
        
        inp = torch.tensor(np.hstack([x_flat, u_flat]), dtype=torch.float32).unsqueeze(0)
        print("Input shape for NN:", inp.shape)  # Debugging line
        
        with torch.no_grad():
            return self.net(inp).squeeze(0).numpy()
        
    def update(self, x, u, x_next_true):
        # Store new data point
        self.X_train.append(np.hstack([x, u]))
        self.y_train.append(x_next_true)
        
        # Calculate errors
        x_next_actual = self.actual_dynamics(x, u)
        x_next_nn = self.predict(x, u)
        
        self.actual_errors.append(np.linalg.norm(x_next_actual - x_next_true))
        self.nn_errors.append(np.linalg.norm(x_next_nn - x_next_true))
        self.nn_vs_actual_diff.append(np.linalg.norm(x_next_nn - x_next_actual))
        
        # Retrain model periodically
        if len(self.X_train) % 50 == 0 and len(self.X_train) > 0:
            X = torch.tensor(np.array(self.X_train), dtype=torch.float32)
            y = torch.tensor(np.array(self.y_train), dtype=torch.float32)
            
            self.train()
            for _ in range(10):
                self.optimizer.zero_grad()
                outputs = self(X)
                loss = self.loss_fn(outputs, y)
                loss.backward()
                self.optimizer.step()
                self.losses.append(loss.item())
                
            self.trained = True
    
    def get_jacobians(self, x, u):
        xu = torch.tensor(np.hstack([x, u]), dtype=torch.float32).requires_grad_(True)
        output = self.net(xu.unsqueeze(0)).squeeze(0)
        
        jacobian = torch.zeros(2, 3)
        for i in range(2):
            grad = torch.autograd.grad(output[i], xu, retain_graph=True)[0]
            jacobian[i] = grad
            
        A = jacobian[:, :2].detach().numpy()
        B = jacobian[:, 2:].detach().numpy()
        return A, B

# ... [Keep the iLQRController class identical to previous version but with this fix] ...

def main():
    env = gym.make('MountainCarContinuous-v0', render_mode='human')
    dynamics_model = DynamicsModel()
    controller = iLQRController(dynamics_model, horizon=40, max_iter=5)
    
    obs, _ = env.reset()
    total_reward = 0
    u_guess = np.zeros((controller.horizon, 1))
    
    # Phase 1: Random Exploration
    exploration_steps = 50
    for _ in range(exploration_steps):
        action = np.random.uniform(-1, 1, size=(1,))
        next_obs, _, _, _, _ = env.step(action)
        dynamics_model.update(obs, action, next_obs)
        obs = next_obs

    # Phase 2: iLQR with Learned Dynamics
    for _ in range(10000):  
        u_opt = controller.optimize(np.array(obs), u_guess)
        action = np.clip(u_opt[0], -1, 1)
        next_obs, reward, terminated, _, _ = env.step([action])
        total_reward += reward
        
        dynamics_model.update(np.array(obs), action, next_obs)
        obs = next_obs
        u_guess[:-1] = np.clip(u_opt[1:], -1, 1)
        u_guess[-1] = action
        
        if terminated:
            print(f"Target reached! Total Reward: {total_reward}")
            break

    env.close()
    
    # Plotting
    plt.figure(figsize=(12, 8))
    
    # Training Loss
    plt.subplot(3, 1, 1)
    plt.plot(dynamics_model.losses)
    plt.title('Training Loss')
    plt.xlabel('Training Epochs')
    plt.ylabel('MSE Loss')
    
    # Prediction Errors
    plt.subplot(3, 1, 2)
    plt.plot(dynamics_model.actual_errors, label='Actual Dynamics Error')
    plt.plot(dynamics_model.nn_errors, label='NN Prediction Error')
    plt.title('Prediction Errors Comparison')
    plt.xlabel('Time Step')
    plt.ylabel('L2 Norm Error')
    plt.legend()
    
    # Model Comparison
    plt.subplot(3, 1, 3)
    plt.plot(dynamics_model.nn_vs_actual_diff)
    plt.title('NN vs Actual Dynamics Difference')
    plt.xlabel('Time Step')
    plt.ylabel('Norm Difference')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()