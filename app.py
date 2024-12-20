# /// script
# dependencies = [
#     "streamlit",
#     "numpy",
#     "matplotlib",
#     "scipy",
# ]
# ///

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from scipy.integrate import odeint


def system(state, t, a, b, c, d):
    """
    Define the 2D system of ODEs:
    dx/dt = ax + by
    dy/dt = cx + dy
    """
    x, y = state
    return [a * x + b * y, c * x + d * y]


def plot_phase_portrait(a, b, c, d):
    # Create a grid of initial conditions
    x = np.linspace(-2, 2, 20)
    y = np.linspace(-2, 2, 20)
    X, Y = np.meshgrid(x, y)

    # Calculate direction vectors
    U = a * X + b * Y
    V = c * X + d * Y

    # Normalize vectors for better visualization
    norm = np.sqrt(U**2 + V**2)
    U = U / norm
    V = V / norm

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.quiver(X, Y, U, V)

    # Plot some example trajectories
    t = np.linspace(0, 5, 100)
    initial_conditions = [[-1, -1], [1, 1], [-1, 1], [1, -1], [0, 1], [1, 0]]

    for init in initial_conditions:
        solution = odeint(system, init, t, args=(a, b, c, d))
        ax.plot(solution[:, 0], solution[:, 1], "r-", linewidth=1, alpha=0.5)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Phase Portrait")
    ax.grid(True)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

    return fig


def main():
    st.title("2D System Phase Portrait Visualization")

    st.sidebar.header("System Parameters")
    st.sidebar.write("dx/dt = ax + by")
    st.sidebar.write("dy/dt = cx + dy")

    # Parameter sliders
    a = st.sidebar.slider("a", min_value=-2.0, max_value=2.0, value=0.0, step=0.1)
    b = st.sidebar.slider("b", min_value=-2.0, max_value=2.0, value=1.0, step=0.1)
    c = st.sidebar.slider("c", min_value=-2.0, max_value=2.0, value=-1.0, step=0.1)
    d = st.sidebar.slider("d", min_value=-2.0, max_value=2.0, value=0.0, step=0.1)

    # Display the phase portrait
    fig = plot_phase_portrait(a, b, c, d)
    st.pyplot(fig)

    # System information
    st.sidebar.markdown("### System Information")
    eigenvals = np.linalg.eigvals([[a, b], [c, d]])
    st.sidebar.write("Eigenvalues:", eigenvals)

    # Classification of equilibrium point
    real_parts = eigenvals.real
    if np.all(real_parts < 0):
        stability = "Stable"
    elif np.all(real_parts > 0):
        stability = "Unstable"
    else:
        stability = "Saddle point"

    st.sidebar.write("Equilibrium type:", stability)


if __name__ == "__main__":
    main()
