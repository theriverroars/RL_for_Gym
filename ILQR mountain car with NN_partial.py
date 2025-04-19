#ILQR for Mountain Car with Neural Network Dynamics Model. A, B calcuulated with known dynamics.
import numpy as np
import gymnasium as gym
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
#from IPython.display import clear_output
import os

# Environment parameters
TARGET_POSITION = 0.45
POWER = 0.0015
GRAVITY = 0.0025

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
    #ax2.set_xlabel('Training Steps')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.draw()
    plt.pause(0.01)  # REQUIRED for live updates

class DynamicsModel:
    """ Neural network to approximate the unknown dynamics. """
    def __init__(self):
        self.model = MLPRegressor(hidden_layer_sizes=(64, 64), activation='relu', solver='adam', max_iter=500, warm_start=True)
        self.trained = False
        self.X_train = []
        self.y_train = []
    
    def predict(self, x, u):
        """ Predicts next state using learned dynamics. """
        if self.trained:
            inp = np.hstack([x, u]).reshape(1, -1)
            return self.model.predict(inp)[0]
        else:
            # If not trained, return an estimate with noise
            return np.array([x[0] + x[1], x[1] + u[0] * POWER - GRAVITY * np.cos(3 * x[0])]) + np.random.normal(0, 0.01, 2)
    
    def update(self, x, u, x_next):
        """ Collects new data and retrains the model. """
        self.X_train.append(np.hstack([x, u]))
        self.y_train.append(x_next)
        
        if len(self.X_train) > 50:  # Train after collecting more data
            self.model.fit(np.array(self.X_train), np.array(self.y_train))
            self.trained = True
            if hasattr(self.model, "loss_"):
                loss_history.append(self.model.loss_)
        
        #Update position error for plotting
        position_errors.append(x[0] - TARGET_POSITION)
        
        if len(loss_history) % 20 == 0:
            update_plots()
        
        

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
        
        # Approximate linearization
        A = np.array([[1, 1], 
                     [3 * GRAVITY * np.sin(3 * x[0]), 1]])
        B = np.array([[0], [POWER]])
        
        return x_next, A, B

    def compute_trajectory(self, x0, u_seq):
        x_seq = [x0]
        for u in u_seq:
            x_next, _, _ = self.dynamics(x_seq[-1], u)
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
            x, _, _ = self.dynamics(x, new_u[t])
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

def main():
    env = gym.make('MountainCarContinuous-v0', render_mode='human')
    dynamics_model = DynamicsModel()
    controller = iLQRController(dynamics_model, horizon=40, max_iter=5)
    
    obs, _ = env.reset()
    count = 0
    total_reward = 0
    u_guess = np.zeros((controller.horizon, 1))
    
    # **Phase 1: Random Exploration**
    exploration_steps = 50#np.random.randint(20, 50)  # Variable exploration phase
    for _ in range(exploration_steps):
        action = np.random.normal(0, 0.5, size=(1,))  # More varied actions
        action = np.clip(action, -1, 1)
        
        next_obs, _, _, _, _ = env.step(action)
        dynamics_model.update(obs, action, next_obs)
        obs = next_obs

    # **Phase 2: iLQR with Learned Dynamics**
    for _ in range(1000):  
        count += 1
        u_opt = controller.optimize(np.array(obs), u_guess)
        
        action = np.clip(u_opt[0], -1, 1)[0]
        next_obs, reward, terminated, _, _ = env.step([action])
        total_reward += reward
        
        # Update the neural network with new data
        dynamics_model.update(obs, action, next_obs)

        obs = next_obs
        u_guess[:-1] = np.clip(u_opt[1:], -1, 1)
        u_guess[-1] = action
        
        if terminated:
            print("Target reached!", "Total Reward:", total_reward, "Steps:", count)
            break
        
    plt.savefig("training_performance.png")
    env.close()

if __name__ == "__main__":
    main()