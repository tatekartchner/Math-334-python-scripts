import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from tkinter import Tk, Label, Entry, Button, DoubleVar, Scale, HORIZONTAL

def plot_direction_field(matrix, x_range, y_range, vector_scale, density=20):
    """
    Plot a direction field for a given matrix and range of values.

    Parameters:
    - matrix (numpy.ndarray): A 2x2 matrix defining the system.
    - x_range (tuple): The range of x values as (xmin, xmax).
    - y_range (tuple): The range of y values as (ymin, ymax).
    - vector_scale (float): Scale factor for the normalized vectors.
    - density (int): The density of the grid for plotting.
    """
    # Generate a grid of points
    x = np.linspace(x_range[0], x_range[1], density)
    y = np.linspace(y_range[0], y_range[1], density)
    X, Y = np.meshgrid(x, y)
    
    # Compute the vector field
    U = matrix[0, 0] * X + matrix[0, 1] * Y
    V = matrix[1, 0] * X + matrix[1, 1] * Y
    
    # Normalize the vectors
    magnitude = np.sqrt(U**2 + V**2)
    
    # To avoid division by zero
    magnitude[magnitude == 0] = 1
    
    U_norm = U / magnitude
    V_norm = V / magnitude

    # Scale the normalized vectors
    U_scaled = U_norm * vector_scale
    V_scaled = V_norm * vector_scale

    # Plot the direction field
    plt.quiver(X, Y, U_scaled, V_scaled, angles='xy', scale_units='xy', scale=1, color="blue", alpha=0.6)

def trajectory_plotter(matrix, x_range, y_range, initial_conditions, t_max=10, t_points=500, vector_scale = 0.5):
    """
    Plot trajectories of a dynamical system over a direction field.

    Parameters:
    - matrix (numpy.ndarray): A 2x2 matrix defining the system.
    - x_range (tuple): The range of x values as (xmin, xmax).
    - y_range (tuple): The range of y values as (ymin, ymax).
    - initial_conditions (list of tuples): Initial conditions for trajectories.
    - t_max (float): Maximum time for trajectory simulation.
    - t_points (int): Number of time points for the simulation.
    """
    # Define the system of ODEs
    def system(state, t):
        x, y = state
        dxdt = matrix[0, 0] * x + matrix[0, 1] * y
        dydt = matrix[1, 0] * x + matrix[1, 1] * y
        return [dxdt, dydt]

    # Time points for integration
    t = np.linspace(0, t_max, t_points)

    # Plot the direction field
    plot_direction_field(matrix, x_range, y_range, vector_scale=vector_scale)

    # Define colors for trajectories
    colors = ['red', 'green', 'orange', 'purple', 'brown', 'cyan']
    
    # Simulate and plot each trajectory
    for idx, ic in enumerate(initial_conditions):
        trajectory = odeint(system, ic, t)
        plt.plot(trajectory[:, 0], trajectory[:, 1], label=f"IC: {ic}", color=colors[idx % len(colors)])

    # Finalize the plot
    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.axhline(0, color='k', linewidth=0.5)
    plt.axvline(0, color='k', linewidth=0.5)
    plt.grid(alpha=0.3)
    plt.title("Direction Field with Trajectories")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc='best')
    plt.show()

def create_gui():
    """
    Create a GUI for entering the matrix and trajectory initial conditions.
    """
    def submit_and_plot():
        try:
            # Get matrix values
            a = float(entry_a.get())
            b = float(entry_b.get())
            c = float(entry_c.get())
            d = float(entry_d.get())
            
            # Create the matrix
            matrix = np.array([[a, b], [c, d]])
            
            # Get vector scale
            vector_scale = scale_var.get()
            
            # Parse initial conditions (if any)
            ics_text = entry_ics.get().replace(" ", "")  # Remove spaces
            if ics_text:
                initial_conditions = [
                    tuple(map(float, pair.strip("()").split(',')))
                    for pair in ics_text.split('),(')
                ]
                # Plot the direction field with trajectories
                trajectory_plotter(matrix, x_range=(-5, 5), y_range=(-5, 5), initial_conditions=initial_conditions, t_max=10, t_points=500, vector_scale = vector_scale)
            else:
                # Only plot the direction field if no initial conditions
                plot_direction_field(matrix, x_range=(-5, 5), y_range=(-5, 5), vector_scale=vector_scale, density=20)
                plt.xlim(-5, 5)
                plt.ylim(-5, 5)
                plt.title("Direction Field")
                plt.xlabel("x")
                plt.ylabel("y")
                plt.grid(alpha=0.3)
                plt.show()
                
        except ValueError:
            print("Please enter valid numerical values.")

    # Create the main window
    root = Tk()
    root.title("Direction Field and Trajectories")
    
    # Matrix input
    Label(root, text="Matrix (2x2):").grid(row=0, column=0, columnspan=2)
    Label(root, text="a:").grid(row=1, column=0)
    entry_a = Entry(root, width=5)
    entry_a.grid(row=1, column=1)

    Label(root, text="b:").grid(row=1, column=2)
    entry_b = Entry(root, width=5)
    entry_b.grid(row=1, column=3)

    Label(root, text="c:").grid(row=2, column=0)
    entry_c = Entry(root, width=5)
    entry_c.grid(row=2, column=1)

    Label(root, text="d:").grid(row=2, column=2)
    entry_d = Entry(root, width=5)
    entry_d.grid(row=2, column=3)

    # Initial conditions input
    Label(root, text="Initial Conditions ((x1,y1), (x2,y2), ...):").grid(row=3, column=0, columnspan=4)
    entry_ics = Entry(root, width=40)
    entry_ics.grid(row=4, column=0, columnspan=4)

    # Vector length scale
    Label(root, text="Vector Scale:").grid(row=5, column=0, columnspan=2)
    scale_var = DoubleVar(value=1.0)
    vector_scale = Scale(root, variable=scale_var, from_=0.1, to=3.0, resolution=0.1, orient=HORIZONTAL)
    vector_scale.grid(row=5, column=2, columnspan=2)

    # Submit button
    Button(root, text="Plot", command=submit_and_plot).grid(row=6, column=0, columnspan=4)
    
    # Run the main loop
    root.mainloop()

# Run the GUI
create_gui()
