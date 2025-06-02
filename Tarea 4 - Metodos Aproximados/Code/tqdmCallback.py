from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm
class TQDMCallback(BaseCallback):
    """
    Custom callback to display progress bar during training.
    """

    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.pbar = tqdm(
            total=total_timesteps,
            desc="Training Progress",
            unit="timesteps",
            disable=verbose == 0,
        )

    def _on_training_start(self):
        self.pbar = tqdm(total=self.locals["total_timesteps"], desc="Training Progress")

    def _on_step(self):
        self.pbar.update(self.model.num_timesteps - self.pbar.n)
        return True

    def _on_training_end(self):
        self.pbar.close()

