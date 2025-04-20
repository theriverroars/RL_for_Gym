import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib import animation
from tqdm import tqdm
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

# Neural Network Dynamics Model
class DynamicsModel(nn.Module):
    def __init__(self, state_dim, control_dim, hidden_dim=128):
        super(DynamicsModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + control_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return state + self.network(x)  # Predict state difference (residual)

# Wrapper for CartPole environment to modify it for the swing-up task
class CartPoleSwingUpEnv(gym.Wrapper):
    def __init__(self):
        # Create the standard CartPole environment
        env = gym.make('CartPole-v1')
        super(CartPoleSwingUpEnv, self).__init__(env)
        
        # Override action space for continuous control
        self.action_space = gym.spaces.Box(
            low=-10.0, high=10.0, shape=(1,), dtype=np.float32
        )
        
        # State space remains the same: [x, x_dot, theta, theta_dot]
        # But we'll normalize theta to be between -pi and pi
        
        # Environment constants
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.length = 0.5  # half the pole's length
        self.total_mass = self.masscart + self.masspole
        self.polemass_length = self.masspole * self.length
        self.max_force = 10.0
        self.dt = 0.02  # seconds between state updates
        
        # Task settings
        self.target_theta = 0.0  # upright position
        self.max_steps = 500
        self.steps = 0
    
    def reset(self, **kwargs):
        # Reset to a random position with pendulum pointing down
        self.steps = 0
        obs, info = self.env.reset(**kwargs)
        
        # Modify the initial state so the pendulum points down
        # Standard CartPole has theta=0 as upright, we want it to start at π (pointing down)
        obs = np.array(obs, dtype=np.float32)
        obs[2] = np.pi  # Set theta to pi (pointing down)
        obs[3] = 0.0    # Set theta_dot to 0
        
        return obs, info
    
    def step(self, action):
        self.steps += 1
        
        # Clip action
        force = np.clip(action[0], -self.max_force, self.max_force)
        
        # Extract state
        x, x_dot, theta, theta_dot = self.unwrapped.state
        
        # Normalize theta to be between -pi and pi
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi
        
        # Apply dynamics (simplified from CartPole's C implementation)
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        
        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / \
                   (self.length * (4.0/3.0 - self.masspole * costheta**2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        
        # Update state using Euler integration
        x += self.dt * x_dot
        x_dot += self.dt * xacc
        theta += self.dt * theta_dot
        theta_dot += self.dt * thetaacc
        
        # Normalize theta to [-pi, pi]
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi
        
        # Update environment state
        self.unwrapped.state = (x, x_dot, theta, theta_dot)
        
        # Calculate reward based on closeness to upright position
        upright = np.cos(theta)  # 1 when upright, -1 when hanging down
        x_penalty = 0.1 * (x**2)  # Small penalty for distance from center
        velocity_penalty = 0.1 * (x_dot**2 + theta_dot**2)  # Small penalty for high velocities
        control_penalty = 0.01 * (force**2)  # Small penalty for large forces
        
        # Reward: more positive when upright (+1) and zero when hanging down (-1)
        reward = (upright + 1) / 2 - x_penalty - velocity_penalty - control_penalty
        
        # Check for termination
        terminated = False
        # Don't terminate early to allow for exploration and swing-up
        # Only terminate if the cart goes too far
        if abs(x) > self.unwrapped.x_threshold:
            terminated = True
            reward = -10  # Penalty for going out of bounds
        
        # Check for truncation (max steps)
        truncated = self.steps >= self.max_steps
        
        # Create observation
        obs = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
        
        return obs, reward, terminated, truncated, {}

# Collect data for dynamics model training
def collect_data(env, num_episodes=100, steps_per_episode=200):
    states = []
    actions = []
    next_states = []
    rewards = []
    
    for _ in tqdm(range(num_episodes), desc="Collecting data"):
        obs, _ = env.reset()
        
        for _ in range(steps_per_episode):
            # Random action
            action = env.action_space.sample()
            
            # Store current state and action
            states.append(obs)
            actions.append(action)
            
            # Step environment
            next_obs, reward, terminated, truncated, _ = env.step(action)
            
            # Store result
            next_states.append(next_obs)
            rewards.append(reward)
            
            # Update state
            obs = next_obs
            
            if terminated or truncated:
                break
    
    return (np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32).reshape(-1, 1),
            np.array(next_states, dtype=np.float32),
            np.array(rewards, dtype=np.float32))

# Train dynamics model
def train_dynamics_model(model, states, actions, next_states, epochs=100, batch_size=128):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # Convert to tensors
    states_tensor = torch.from_numpy(states)
    actions_tensor = torch.from_numpy(actions)
    next_states_tensor = torch.from_numpy(next_states)
    
    # Training loop
    losses = []
    dataset_size = len(states)
    
    for epoch in tqdm(range(epochs), desc="Training dynamics model"):
        epoch_loss = 0
        # Shuffle data
        indices = np.random.permutation(dataset_size)
        
        # Mini-batch training
        for start_idx in range(0, dataset_size, batch_size):
            end_idx = min(start_idx + batch_size, dataset_size)
            batch_indices = indices[start_idx:end_idx]
            
            state_batch = states_tensor[batch_indices]
            action_batch = actions_tensor[batch_indices]
            next_state_batch = next_states_tensor[batch_indices]
            
            # Forward pass
            predicted_next_states = model(state_batch, action_batch)
            loss = criterion(predicted_next_states, next_state_batch)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * (end_idx - start_idx)
        
        avg_loss = epoch_loss / dataset_size
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    return losses

# Cost function for swing-up task
def cost_function(states, actions, goal_state, Q, R, QN):
    """
    Compute cost for a trajectory
    Q: State cost matrix
    R: Control cost matrix
    QN: Terminal state cost matrix
    """
    T = len(actions)
    cost = 0
    
    # Running cost
    for t in range(T):
        state_diff = states[t] - goal_state
        cost += state_diff.dot(Q).dot(state_diff) + actions[t].dot(R).dot(actions[t])
    
    # Terminal cost
    state_diff = states[-1] - goal_state
    cost += state_diff.dot(QN).dot(state_diff)
    
    return cost

# iLQR algorithm
class iLQR:
    def __init__(self, dynamics_model, env, horizon=100):
        self.dynamics_model = dynamics_model
        self.env = env
        self.horizon = horizon
        
        # State and control dimensions
        self.state_dim = 4  # [x, x_dot, theta, theta_dot]
        self.control_dim = 1  # force
        
        # Cost function weights
        self.Q = np.diag([0.1, 0.1, 10.0, 0.1])  # State cost
        self.R = np.diag([0.01])                 # Control cost
        self.QN = np.diag([1.0, 1.0, 100.0, 1.0])  # Terminal cost
        
        # Goal state (upright pendulum, centered cart)
        self.goal_state = np.array([0.0, 0.0, 0.0, 0.0])
    
    def rollout(self, x0, us):
        """Simulate trajectory with given initial state and controls"""
        T = len(us)
        xs = np.zeros((T+1, self.state_dim))
        xs[0] = x0
        
        # Convert to tensor for NN dynamics
        states_tensor = torch.from_numpy(xs[0:1].astype(np.float32))
        
        for t in range(T):
            action_tensor = torch.from_numpy(us[t:t+1].astype(np.float32))
            next_state_tensor = self.dynamics_model(states_tensor, action_tensor)
            
            # Update state
            xs[t+1] = next_state_tensor.detach().numpy()[0]
            states_tensor = next_state_tensor.detach()
        
        return xs
    
    def compute_derivatives(self, xs, us):
        """Compute linearized dynamics and quadratized cost around trajectory"""
        T = len(us)
        fx = [None] * T  # df/dx
        fu = [None] * T  # df/du
        
        lx = [None] * (T+1)  # dl/dx
        lu = [None] * T      # dl/du
        lxx = [None] * (T+1) # d^2l/dx^2
        luu = [None] * T     # d^2l/du^2
        lux = [None] * T     # d^2l/dudx
        
        eps = 1e-4  # Finite difference epsilon
        
        # Linearize dynamics using finite differences
        for t in range(T):
            state = xs[t]
            action = us[t]
            
            # Create tensors
            state_tensor = torch.from_numpy(state.astype(np.float32))
            action_tensor = torch.from_numpy(action.astype(np.float32))
            
            # Enable autograd
            state_tensor.requires_grad_(True)
            action_tensor.requires_grad_(True)
            
            # Forward pass
            next_state_tensor = self.dynamics_model(state_tensor.unsqueeze(0), 
                                                    action_tensor.unsqueeze(0)).squeeze(0)
            
            # Compute Jacobians
            fx_t = np.zeros((self.state_dim, self.state_dim))
            fu_t = np.zeros((self.state_dim, self.control_dim))
            
            for i in range(self.state_dim):
                # Create a unit vector for backprop
                unit = torch.zeros_like(next_state_tensor)
                unit[i] = 1.0
                
                # Backprop
                next_state_tensor.backward(unit, retain_graph=True)
                
                # Extract gradients
                fx_t[i] = state_tensor.grad.numpy()
                fu_t[i] = action_tensor.grad.numpy()
                
                # Reset gradients
                state_tensor.grad.zero_()
                action_tensor.grad.zero_()
            
            fx[t] = fx_t
            fu[t] = fu_t
            
            # Compute cost derivatives
            state_diff = xs[t] - self.goal_state
            lx[t] = 2 * self.Q.dot(state_diff)
            lu[t] = 2 * self.R.dot(us[t])
            lxx[t] = 2 * self.Q
            luu[t] = 2 * self.R
            lux[t] = np.zeros((self.control_dim, self.state_dim))
        
        # Terminal cost derivatives
        state_diff = xs[T] - self.goal_state
        lx[T] = 2 * self.QN.dot(state_diff)
        lxx[T] = 2 * self.QN
        
        return fx, fu, lx, lu, lxx, luu, lux
    
    def backward_pass(self, fx, fu, lx, lu, lxx, luu, lux):
        """Backward pass to compute optimal control law"""
        T = len(fu)
        
        # Initialize value function
        Vx = lx[T]
        Vxx = lxx[T]
        
        # Gains
        k = [None] * T
        K = [None] * T
        
        for t in range(T-1, -1, -1):
            Qx = lx[t] + fx[t].T.dot(Vx)
            Qu = lu[t] + fu[t].T.dot(Vx)
            Qxx = lxx[t] + fx[t].T.dot(Vxx).dot(fx[t])
            Quu = luu[t] + fu[t].T.dot(Vxx).dot(fu[t])
            Qux = lux[t] + fu[t].T.dot(Vxx).dot(fx[t])
            
            # Ensure Quu is positive definite
            Quu_reg = Quu + 1e-3 * np.eye(self.control_dim)
            
            # Compute gains
            try:
                k[t] = -np.linalg.solve(Quu_reg, Qu)
                K[t] = -np.linalg.solve(Quu_reg, Qux)
            except np.linalg.LinAlgError:
                # If matrix is singular, use pseudoinverse
                k[t] = -np.linalg.pinv(Quu_reg).dot(Qu)
                K[t] = -np.linalg.pinv(Quu_reg).dot(Qux)
            
            # Update value function
            Vx = Qx + K[t].T.dot(Quu).dot(k[t]) + K[t].T.dot(Qu) + Qux.T.dot(k[t])
            Vxx = Qxx + K[t].T.dot(Quu).dot(K[t]) + K[t].T.dot(Qux) + Qux.T.dot(K[t])
            Vxx = 0.5 * (Vxx + Vxx.T)  # Ensure symmetry
        
        return k, K
    
    def forward_pass(self, xs, us, k, K, alpha=1.0):
        """Forward pass to compute new trajectory"""
        T = len(us)
        new_xs = np.zeros_like(xs)
        new_us = np.zeros_like(us)
        
        new_xs[0] = xs[0]
        
        # State tensor for dynamics model
        state_tensor = torch.from_numpy(new_xs[0:1].astype(np.float32))
        
        for t in range(T):
            # Compute feedback control
            state_diff = new_xs[t] - xs[t]
            new_us[t] = us[t] + alpha * k[t] + K[t].dot(state_diff)
            
            # Clip control
            new_us[t] = np.clip(new_us[t], -self.env.max_force, self.env.max_force)
            
            # Apply dynamics
            action_tensor = torch.from_numpy(new_us[t:t+1].astype(np.float32))
            next_state_tensor = self.dynamics_model(state_tensor, action_tensor)
            
            # Update state
            new_xs[t+1] = next_state_tensor.detach().numpy()[0]
            state_tensor = next_state_tensor.detach()
        
        return new_xs, new_us
    
    def optimize(self, x0, max_iterations=50):
        """Main iLQR optimization loop"""
        # Initialize with zero controls
        us = np.zeros((self.horizon, self.control_dim))
        
        # Initial rollout
        xs = self.rollout(x0, us)
        
        # Cost of initial trajectory
        cost = cost_function(xs, us, self.goal_state, self.Q, self.R, self.QN)
        
        # Optimization loop
        for iteration in range(max_iterations):
            print(f"Iteration {iteration+1}, Cost: {cost:.4f}")
            
            # Linearize dynamics and quadratize cost
            fx, fu, lx, lu, lxx, luu, lux = self.compute_derivatives(xs, us)
            
            # Backward pass
            k, K = self.backward_pass(fx, fu, lx, lu, lxx, luu, lux)
            
            # Line search
            alpha = 1.0
            max_line_search = 10
            ls_success = False
            
            for ls_iter in range(max_line_search):
                new_xs, new_us = self.forward_pass(xs, us, k, K, alpha)
                new_cost = cost_function(new_xs, new_us, self.goal_state, self.Q, self.R, self.QN)
                
                if new_cost < cost:
                    xs, us = new_xs, new_us
                    cost = new_cost
                    ls_success = True
                    break
                
                alpha *= 0.5
            
            # Check for convergence
            if not ls_success:
                print("Line search failed to improve cost")
                break
            
            if iteration > 0 and alpha < 1e-3:
                print("Converged (small step size)")
                break
        
        return xs, us, cost

# Evaluate a policy on the environment
def evaluate_policy(env, dynamics_model, policy=None, num_episodes=3, render=True):
    total_rewards = []
    
    if render:
        env = RecordVideo(env, "videos/cartpole-swingup")
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            if policy is None:
                # Random action if no policy provided
                action = env.action_space.sample()
            else:
                # Use the learned policy
                action = policy(obs)
            
            obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
        
        total_rewards.append(episode_reward)
        print(f"Episode {episode+1}: Reward = {episode_reward}")
    
    avg_reward = np.mean(total_rewards)
    print(f"Average reward over {num_episodes} episodes: {avg_reward:.2f}")
    return avg_reward

# Create a policy from optimized trajectory
def create_ilqr_policy(ilqr, xs, us):
    k, K = ilqr.compute_derivatives(xs, us)[0:2]  # Get the k and K matrices
    
    def policy(state):
        # Find the closest state in the trajectory
        diffs = xs[:-1] - state
        distances = np.sum(diffs**2, axis=1)
        closest_idx = np.argmin(distances)
        
        # Apply feedback control
        state_diff = state - xs[closest_idx]
        action = us[closest_idx] + K[closest_idx].dot(state_diff)
        
        # Clip action
        action = np.clip(action, -ilqr.env.max_force, ilqr.env.max_force)
        return action
    
    return policy

# Visualize cart pole trajectories
def plot_trajectory(states, goal_state=None, title="Cart Pole Trajectory"):
    plt.figure(figsize=(12, 10))
    
    # Time axis
    t = np.arange(len(states))
    
    labels = ['Cart Position', 'Cart Velocity', 'Pendulum Angle', 'Pendulum Velocity']
    
    for i in range(4):
        plt.subplot(4, 1, i+1)
        plt.plot(t, states[:, i])
        
        if goal_state is not None:
            plt.axhline(y=goal_state[i], color='r', linestyle='--', label='Goal')
            plt.legend()
        
        plt.ylabel(labels[i])
        plt.grid(True)
    
    plt.xlabel('Time Step')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# Main function
def main():
    # Create CartPole swing-up environment
    env = CartPoleSwingUpEnv()
    
    # Create neural network dynamics model
    dynamics_model = DynamicsModel(4, 1, hidden_dim=128)
    
    # Collect training data
    print("Collecting training data...")
    states, actions, next_states, rewards = collect_data(env, num_episodes=200, steps_per_episode=200)
    
    # Train dynamics model
    print("Training dynamics model...")
    losses = train_dynamics_model(dynamics_model, states, actions, next_states, epochs=200)
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Dynamics Model Training Loss')
    plt.grid(True)
    plt.show()
    
    # Initialize iLQR controller
    ilqr = iLQR(dynamics_model, env, horizon=200)
    
    # Initial state (pendulum pointing down)
    obs, _ = env.reset()
    x0 = np.array(obs, dtype=np.float32)
    
    # Optimize trajectory
    print("Optimizing trajectory with iLQR...")
    xs, us, cost = ilqr.optimize(x0, max_iterations=30)
    
    # Print final cost
    print(f"Final cost: {cost:.4f}")
    
    # Create a policy from the optimized trajectory
    ilqr_policy = create_ilqr_policy(ilqr, xs, us)
    
    # Plot optimized trajectory
    plot_trajectory(xs, ilqr.goal_state, "Optimized iLQR Trajectory")
    
    # Plot control sequence
    plt.figure(figsize=(10, 6))
    plt.plot(us)
    plt.xlabel('Time Step')
    plt.ylabel('Force')
    plt.title('Optimized Control Sequence')
    plt.grid(True)
    plt.show()
    
    # Evaluate the learned policy
    print("Evaluating policy...")
    evaluate_policy(env, dynamics_model, ilqr_policy, num_episodes=3, render=True)

if __name__ == "__main__":
    main()