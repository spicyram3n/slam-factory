import numpy as np
from typing import Tuple

class ObservationModel:
     
    def __init__(self, sigma_r: float = 0.1, sigma_phi: float = 0.05):
        self.sigma_r = sigma_r
        self.sigma_phi = sigma_phi
        self.Q = np.diag([sigma_r**2, sigma_phi**2])
        
    def predict(self, state: np.ndarray, landmark: np.ndarray) -> np.ndarray:
        
        """
        Args:
            robot_pose: [x, y, theta]
            landmark: [mx, my]

        Returns:
            z_pred: [range, bearing] predicted measurement
        """
        x, y, theta = state
        mx, my = landmark
        
        dx = mx - x
        dy = my - y
        
        # range
        r = np.sqrt(dx**2 + dy**2)
        
        # transform bearing to robot frame
        c = np.cos(theta)
        s = np.sin(theta)
        dx_r = c * dx + s * dy
        dy_r = -s * dx + c * dy
        phi = np.arctan2(dy_r, dx_r)
        
        return np.array([r, phi])
    
    def jacobian(self, state: np.ndarray, landmark: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        
        x, y, theta = state
        mx, my = landmark

        dx = mx - x
        dy = my - y
        r2 = dx**2 + dy**2
        r = np.sqrt(r2)

        # Robot frame rotation (same as liuji1031's formulation)
        c = np.cos(theta)
        s = np.sin(theta)
        dx_r = c * dx + s * dy
        dy_r = -s * dx + c * dy

        # Helper terms for bearing Jacobian
        rho_x = -dy_r / r2
        rho_y = dx_r / r2

        # Jacobian w.r.t. robot pose [x, y, theta]
        H_r = np.array([
            [-dx / r, -dy / r, 0],           # d(range)/d(x,y,theta)
            [rho_x,   rho_y,   dy_r * rho_x - dx_r * rho_y]  # d(bearing)/d(x,y,theta)
        ])

        # Jacobian w.r.t. landmark [mx, my]
        H_m = np.array([
            [dx / r,  dy / r],               # d(range)/d(mx,my)
            [-rho_x,  -rho_y]                # d(bearing)/d(mx,my)
        ])

        return H_r, H_m
        
        return