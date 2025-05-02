import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# === PARAMETERS ===
N = 260            # Number of time steps
dt = 2             # Time step size
omega_ast = np.array([0.0, 0.0, 0.0004288])  # Castalia rotation (rad/s)
omega_cross = np.array([
    [0, -omega_ast[2], omega_ast[1]],
    [omega_ast[2], 0, -omega_ast[0]],
    [-omega_ast[1], omega_ast[0], 0]
])

m_wet = 1400.0
m_dry = 1000.0
v_ex = 225.0 * 9.80665  # ~2206.5 m/s
T_min = 20.0
T_max = 80.0

r_0 = np.array([-1066.1, -157.5, 1060.6])
v_0 = np.array([0.1606, -1.890, 0.0863])
r_f = np.array([-345.0, -67.0, 370.0])
v_f = np.array([0.0287, -0.148, 0.0])
gravity_ast = np.array([0.0, 0.0, 3e-5])

t_f = dt * N

# === INITIAL GUESS ===
q0_guess = np.log(m_wet - (T_min / v_ex) * t_f)
max_iters = 10
tol = 1e-3

# Function to plot trajectory
def plot_trajectory(solution, title):
    r = solution['r']
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectory
    ax.plot(r[:, 0], r[:, 1], r[:, 2], 'b-', linewidth=2)
    
    # Plot start and end points
    ax.plot([r[0, 0]], [r[0, 1]], [r[0, 2]], 'go', markersize=8, label='Start')
    ax.plot([r[-1, 0]], [r[-1, 1]], [r[-1, 2]], 'ro', markersize=8, label='End')
    
    # Plot asteroid center
    ax.plot([0], [0], [0], 'yo', markersize=10, label='Asteroid Center')
    
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title(title)
    ax.legend()
    
    plt.tight_layout()
    return fig

# Function to plot thrust profile
def plot_thrust_profile(solution):
    a_t = solution['a_t']
    q = solution['q']
    thrust_magnitude = np.array([np.linalg.norm(a_t[i]) * np.exp(q[i]) for i in range(len(a_t))])
    
    # Time vector
    time = np.arange(0, len(a_t)) * dt
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot thrust magnitude
    ax1.plot(time, thrust_magnitude, 'b-', linewidth=2)
    ax1.set_ylabel('Thrust [N]')
    ax1.set_title('Thrust Magnitude Profile')
    ax1.grid(True)
    
    # Plot mass evolution
    mass = np.exp(q[:-1])  # Convert q to mass
    ax2.plot(time, mass, 'g-', linewidth=2)
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Mass [kg]')
    ax2.set_title('Spacecraft Mass Profile')
    ax2.grid(True)
    
    plt.tight_layout()
    return fig

prev_qN = None
iteration_results = []

# Try multiple solvers in case one fails
solvers_to_try = [cp.ECOS, cp.SCS, cp.OSQP]
solver_names = ["ECOS", "SCS", "OSQP"]
solver_options = [
    {"abstol": 1e-6, "reltol": 1e-6, "max_iters": 1000},
    {"eps": 1e-5, "max_iters": 5000, "verbose": False},
    {"eps_abs": 1e-5, "eps_rel": 1e-5, "max_iter": 10000}
]

for scp_iter in range(max_iters):
    # === VARIABLES ===
    r = cp.Variable((N+1, 3))
    v = cp.Variable((N+1, 3))
    a_t = cp.Variable((N, 3))
    q = cp.Variable(N+1)
    a_t_norm = cp.Variable(N)

    constraints = [
        r[0] == r_0,
        v[0] == v_0,
        q[0] == np.log(m_wet)
    ]

    for k in range(N):
        coriolis = omega_cross @ v[k]
        centrifugal = omega_cross @ (omega_cross @ r[k])

        constraints += [
            r[k+1] == r[k] + dt * v[k],
            v[k+1] == v[k] + dt * (a_t[k] - 2 * coriolis - centrifugal + gravity_ast),
            q[k+1] == q[k] - dt * a_t_norm[k] / v_ex,
            cp.norm(a_t[k], 2) <= a_t_norm[k]
        ]

        # Trust region: linearization around q0_guess
        dq = q[k] - q0_guess
        lower_bound = T_min * np.exp(-q0_guess) * (1 - dq + 0.5 * dq**2)
        upper_bound = T_max * np.exp(-q0_guess) * (1 - dq)

        constraints += [
            lower_bound <= a_t_norm[k],
            a_t_norm[k] <= upper_bound,
            q[k] >= np.log(m_dry)
        ]

    constraints += [
        r[N] == r_f,
        v[N] == v_f,
        q[N] >= np.log(m_dry)
    ]

    objective = cp.Maximize(q[N])
    problem = cp.Problem(objective, constraints)
    
    # Try different solvers
    solved = False
    for i, solver in enumerate(solvers_to_try):
        try:
            print(f"SCP Iter {scp_iter}: Trying {solver_names[i]} solver...")
            problem.solve(solver=solver, **solver_options[i])
            
            if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                solved = True
                print(f"Solved with {solver_names[i]}")
                break
            else:
                print(f"{solver_names[i]} failed with status: {problem.status}")
        except Exception as e:
            print(f"{solver_names[i]} failed with error: {e}")
    
    if not solved:
        print(f"SCP Iter {scp_iter}: All solvers failed.")
        break
    
    if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        print(f"SCP Iter {scp_iter}: Optimization failed with status {problem.status}")
        break

    qN = q.value[-1]
    fuel_used = m_wet - np.exp(qN)
    
    print(f"Iteration {scp_iter}, q[N] = {qN}, Final mass = {np.exp(qN):.4f}, Fuel used = {fuel_used:.4f}")
    
    iteration_results.append({
        'iteration': scp_iter,
        'final_mass': np.exp(qN),
        'fuel_used': fuel_used
    })

    # Check convergence
    if prev_qN is not None and np.abs(qN - prev_qN) < tol:
        print("Converged.")
        break

    prev_qN = qN
    q0_guess = qN  # Update linearization point

# Store final solution for inspection
if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
    final_solution = {
        'r': r.value,
        'v': v.value,
        'a_t': a_t.value,
        'q': q.value,
        'fuel_used': m_wet - np.exp(q.value[-1]),
        'final_mass': np.exp(q.value[-1])
    }
    
    # Calculate final delta-v
    delta_v = v_ex * np.log(m_wet / final_solution['final_mass'])
    print(f"\nResults Summary:")
    print(f"Final mass: {final_solution['final_mass']:.4f} kg")
    print(f"Fuel used: {final_solution['fuel_used']:.4f} kg")
    print(f"Delta-V: {delta_v:.4f} m/s")
    
    # Plot results
    traj_fig = plot_trajectory(final_solution, "Optimal Approach Trajectory to Castalia")
    thrust_fig = plot_thrust_profile(final_solution)
    
    plt.figure()
    iterations = [r['iteration'] for r in iteration_results]
    fuel_used = [r['fuel_used'] for r in iteration_results]
    plt.plot(iterations, fuel_used, 'o-')
    plt.xlabel('SCP Iteration')
    plt.ylabel('Fuel Used [kg]')
    plt.title('Convergence of SCP Algorithm')
    plt.grid(True)
    
    plt.show()
else:
    print("Failed to find a solution.")