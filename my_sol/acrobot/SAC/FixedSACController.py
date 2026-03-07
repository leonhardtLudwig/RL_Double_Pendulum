import numpy as np
from double_pendulum.controller.SAC.SAC_controller import SACController

class FixedSACController(SACController):
    def __init__(self, model_path, dynamics_func, dt, scaling=True):
        # bypassa l'__init__ del padre e lo reimplementa correttamente
        from double_pendulum.controller.abstract_controller import AbstractController
        AbstractController.__init__(self)

        from stable_baselines3 import SAC
        self.model = SAC.load(model_path)
        self.dynamics_func = dynamics_func
        self.dt = dt
        self.scaling = scaling

        # warmup con dimensione corretta
        obs_dim = self.model.observation_space.shape[0]
        self.model.predict(np.zeros(obs_dim))

    def get_control_output_(self, x, t=None):
        obs = self.dynamics_func.normalize_state(x)
        action, _ = self.model.predict(obs, deterministic=True)
        return self.dynamics_func.unscale_action(action)
