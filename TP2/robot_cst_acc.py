import numpy as np

def move_robot(x, process_std, dt=1.):
    """
    Predicts the next state of a robot with a constant acceleration model.

    Args:
    x (list): Current state of the robot [position, velocity, acceleration].
    dt (float): Time step between each measurement update.
    process_std (float): standard deviation in process (m/s)

    Returns: Predicted state of the robot [position, velocity, acceleration].
    """
    # Define the state transition matrix
    F = [[1, dt, 0.5 * dt ** 2],
         [0, 1, dt],
         [0, 0, 1]]

    # Define the process noise covariance matrix
    q = process_std * np.random.randn()
    Q = [[q, 0, 0],
         [0, q, 0],
         [0, 0, q]]

    # Predict the next state using the constant acceleration model and Kalman filter equations
    next_x = np.dot(F, x)
    next_x[1] += process_std*np.random.randn()

    return next_x

def locate_robot(x, measurement_std):
    """

    Args:
        x: Current position of the robot
        measurement_std: standard deviation in measurement m

    Returns: measurement of new position in meters

    """
    return x + np.random.randn()*measurement_std