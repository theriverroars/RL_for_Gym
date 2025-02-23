import numpy as np
import gymnasium as gym
from scipy.linalg import solve_continuous_are
import time
import matplotlib.pyplot as plt

# Define the dynamics and cost functions
def dynamics(x, u):
    """
    Dynamics of the Continuous Mountain Car environment.
    x: state vector [position, velocity]
    u: control input [force]
    """
    power = 0.0015
    position, velocity = float(x[0]), float(x[1]) 
    force = np.clip(u, -1.0, 1.0)  # Clip force to valid range
    new_velocity = velocity + power * force - 0.0025 * np.cos(3 * position)
    new_position = position + new_velocity
    new_velocity = np.clip(new_velocity, -0.07, 0.07)  # Clip velocity to valid range
    return np.array([new_position, new_velocity])

def cost(x, u):
    """
    Cost function for the Continuous Mountain Car problem.
    x: state vector [position, velocity]
    u: control input [force]
    """
    position, velocity = float(x[0]), float(x[1]) 
    target_position = 0.45  # Target position (top of the hill)
    return (position - target_position) ** 2 + 0.1 * (u ** 2)  # Quadratic cost

def linearize_dynamics(x, u):
    """
    Linearize the dynamics around the current state and control.
    Returns A (state Jacobian) and B (control Jacobian).
    """
    position, velocity = float(x[0]), float(x[1])  # Ensure scalars
    A = np.array([[1, 1], [0.0025 * 3 * np.sin(3 * position), 1]], dtype=np.float64)
    B = np.array([[0], [0.0015]], dtype=np.float64)
    return A, B

def LQR(state, u, goal):
    """
    Solve the Linear Quadratic Regulator problem using the continuous Riccati equation.
    state: initial state vector [position, velocity]
    u: control input [force]
    goal: target state vector [position, velocity]
    """
    # Define the state and control dimensions
    state_dim = 2
    control_dim = 1

    # Define the state and control weights
    Q = np.diag([2.0, 2.0])  # State cost
    R = np.array([[0.01]])  # Control cost
    
    # Linearize the dynamics
    A, B = linearize_dynamics(state, u)
    
    # Solve the continuous Riccati equation
    P = solve_continuous_are(A, B, Q, R)
    
    # Compute the LQR gain
    K = -np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
    
    # Compute the control input
    u = K @ (state.reshape(-1, 1) - goal.reshape(-1, 1))
    return u

def main():
    env = gym.make("MountainCarContinuous-v0", render_mode="human")
    state, _ = env.reset()
    state = state.reshape(2, )
    goal = np.array([0.45, 0])
    
    print("Goal shape", goal.shape)
    total_cost = 0  
    max_iter = 500
    u = 0
    
    # Lists to store data for plotting
    position_errors = []
    velocity_errors = []
    velocities = []
    controls = []
    error = np.zeros(2)
    
    print(f"Initial State: {state}")    
    for i in range(max_iter):
        env.render()
        time.sleep(0.01)
        state = state.reshape(2, )
        print("State:", state.shape)
        print("Goal:", goal.shape)
        # Calculate the error
        error = state - goal
        print("Error:", error.shape)
        position_errors.append(error[0])  # Position error
        velocity_errors.append(error[1])  # Velocity error
        velocities.append(state[1])  # Velocity
        
        # Solve the LQR problem
        u = LQR(state, u, goal)
        u = np.clip(u, -1, 1)
        controls.append(u[0])  # Control input
        
        # Take a step in the environment
        next_state, reward, done, _, _ = env.step(u)
        
        # Calculate the cost
        total_cost -= reward
        
        # Update the state
        state = next_state
        
        # Print the current step information
        print(f"Step: {i}, State: {state}, Control: {u}, Cost: {total_cost}")
        
        if done:
            break
    
    env.close()
    
    # Plot the results
    plt.figure()
    plt.plot(position_errors, label='Position Error')
    plt.plot(velocity_errors, label='Velocity Error')
    plt.plot(velocities, label='Velocity')
    plt.plot(controls, label='Control Input')
    plt.legend()
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.title('LQR Control of Mountain Car')
    plt.show()

if __name__ == "__main__":
    main()