#!/usr/bin/env python3

import numpy as np
from typing import Optional


class MotionModel:
    """
    Velocity motion model for Turtlebot3 (differential drive)
    """

    def __init__(self, alpha: Optional[np.ndarray] = None):
        """
        Motion noise parameters [a1, a2, a3, a4, a5, a6]

        Args:
            alpha (np.ndarray, optional): _description_. Defaults to None.
        """
        if alpha is None:
            # Default values
            alpha = np.array([0.01, 0.01, 0.01, 0.01, 0.001, 0.001])
        self.alpha = alpha

    def predict(self, state: np.ndarray, v: float, w: float, dt: float) -> np.ndarray:
        """
        Predict the new robot pose given control (v, w)

        Args:
            state [x, y, theta]: current robot pose
            v (float): linear velocity (m/s)
            w (float): angular velocity (rad/s)
            dt (float): time step (s)

        Returns:
            predicted pose: [x', y', theta']
        """

        x, y, theta = state

        if abs(w) < 1e-6:  # Straight line motion (w ~ 0)
            x_new = x + v * dt * np.cos(theta)
            y_new = y + v * dt * np.sin(theta)
            theta_new = theta

        else:
            x_new = x - (v / w) * np.sin(theta) + (v / w) * np.sin(theta + w * dt)
            y_new = y + (v / w) * np.cos(theta) - (v / w) * np.cos(theta + w * dt)
            theta_new = theta + w * dt

        # Wrap angle b/w [-pi, pi]
        theta_new = np.arctan2(np.sin(theta_new), np.cos(theta_new))

        return np.array([x_new, y_new, theta_new])

    def jacobian_state(self, state: np.ndarray, v: float, w: float, dt: float) -> np.ndarray:

        _, _, theta = state

        if abs(w) < 1e-6:
            G_x = np.array(
                [
                    [1, 0, -v * dt * np.sin(theta)],
                    [0, 1, v * dt * np.cos(theta)],
                    [0, 0, 1],
                ]
            )

        else:
            G_x = np.array(
                [
                    [1, 0, -(v / w) * np.cos(theta) + (v / w) * np.cos(theta + w * dt)],
                    [0, 1, -(v / w) * np.sin(theta) + (v / w) * np.sin(theta + w * dt)],
                    [0, 0, 1],
                ]
            )

        return G_x

    def jacobian_control(
        self, state: np.ndarray, v: float, w: float, dt: float
    ) -> np.ndarray:

        _, _, theta = state

        if abs(w) < 1e-6:
            V = np.array(
                [
                    [dt * np.cos(theta), -0.5 * dt**2 * v * np.sin(theta)],
                    [dt * np.sin(theta), 0.5 * dt**2 * v * np.cos(theta)],
                    [0, dt],
                ]
            )

        else:
            c = np.cos(theta)
            s = np.sin(theta)
            c_new = np.cos(theta + w * dt)
            s_new = np.sin(theta + w * dt)

            V = np.array(
                [
                    [(-s + s_new) / w, (v / w**2) * (s - s_new) + (v / w) * dt * c_new],
                    [(c - c_new) / w, -(v / w**2) * (c - c_new) + (v / w) * dt * s_new],
                    [0, dt],
                ]
            )

        return V

    def motion_noise(
        self, state: np.ndarray, v: float, w: float, dt: float
    ) -> np.ndarray:

        a1, a2, a3, a4, a5, a6 = self.alpha

        sigma_v_sq = a1 * (v**2) + a2 * (w**2)
        sigma_w_sq = a3 * (v**2) + a4 * (w**2)

        # Control space covariance M
        M = np.diag([sigma_v_sq, sigma_w_sq])

        # Control jacobian V
        V = self.jacobian_control(state, v, w, dt)

        # Propagate to state space: R = V * M * V^T
        R = V @ M @ V.T
        
        # Finally, Rotation Noise
        sigma_g_sq = a5 * (v**2) + a6 * (w**2)
        R[2, 2] += sigma_g_sq
        
        return R
