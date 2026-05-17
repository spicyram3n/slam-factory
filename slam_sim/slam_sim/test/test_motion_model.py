# #!/usr/bin/env python3
# from slam_sim.core.motion_model import MotionModel
# import numpy as np

# state = np.array([0, 0, 0])
# v = 1.0
# w = 0.5
# dt = 1.0

# V = MotionModel()

# def test_predict(state: np.ndarray, v: float, w: float, dt: float):
#     return V.predict(state, v, w, dt)

# print(test_predict(state, v, w, dt))

#!/usr/bin/env python3

import numpy as np
from slam_sim.core.motion_model import MotionModel

# simple unit test
def test_predict_turning():

    model = MotionModel()

    state = np.array([0.0, 0.0, 0.0])

    v = 1.0
    w = 0.5
    dt = 1.0

    predicted = model.predict(state, v, w, dt)

    expected = np.array([
        0.95885108,
        0.24483488,
        0.5
    ])

    assert np.allclose(predicted, expected, atol=1e-6)
