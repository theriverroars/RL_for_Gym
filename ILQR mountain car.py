import numpy as np
import gymnasium as gym

# Environment parameters
TARGET_POSITION = 0.45
POWER = 0.0015
GRAVITY = 0.0025

class iLQRController:
    def __init__(self, horizon=50, max_iter=10, Q_terminal=np.diag([1000, 0]), R=0.01):
        self.horizon = horizon
        self.max_iter = max_iter
        self.Q_terminal = Q_terminal
        self.R = R * np.eye(1)
        
    def dynamics(self, x, u):
        x_next = np.array([
            x[0] + x[1],
            x[1] + u[0]*POWER - GRAVITY*np.cos(3*x[0])
        ])
        A = np.array([[1, 1], 
                     [3*GRAVITY*np.sin(3*x[0]), 1]])
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
            
            # Calculate running cost
            total_cost += 0.5 * new_u[t].T @ self.R @ new_u[t]

        # Terminal cost
        total_cost += 0.5 * (x - [TARGET_POSITION, 0]).T @ self.Q_terminal @ (x - [TARGET_POSITION, 0])
        return np.array(new_x), new_u, total_cost

    def optimize(self, x0):
        u_seq = np.zeros((self.horizon, 1))  # Initial guess
        x_seq = self.compute_trajectory(x0, u_seq)

        for _ in range(self.max_iter):
            # Linearize dynamics along trajectory
            A_list, B_list = [], []
            for t in range(self.horizon):
                _, A, B = self.dynamics(x_seq[t], u_seq[t])
                A_list.append(A)
                B_list.append(B)

            # Backward pass
            k, K = self.backward_pass(x_seq, u_seq, A_list, B_list)

            # Forward pass with line search
            new_x, new_u, new_cost = self.forward_pass(x0, u_seq, k, K)
            
            # Update trajectory
            x_seq, u_seq = new_x[:-1], new_u

        return u_seq
    


def main():
    env = gym.make('MountainCarContinuous-v0', render_mode='human')
    controller = iLQRController(horizon=40, max_iter=5)
    
    obs, _ = env.reset()
    count = 0
    total_reward = 0
    for _ in range(1000):  # Max episodes
        count += 1
        # Get optimal control sequence
        u_opt = controller.optimize(np.array(obs))
        
        # Apply first action from optimized sequence
        action = np.clip(u_opt[0], -1, 1)[0]
        obs, reward, terminated, _, _ = env.step([action])
        total_reward += reward
        if terminated:
            print("Target reached!", "Total Reward:", total_reward, "Steps:", count)
            break

    env.close()
    
    


if __name__ == "__main__":
    main()