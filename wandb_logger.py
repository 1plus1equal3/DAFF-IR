import wandb
from datetime import datetime
from .config import PROJECT_NAME, WANDB_KEY_PATHS

class WandbLogger:
    def __init__(self, project_name=PROJECT_NAME, api_key_path=WANDB_KEY_PATHS, config=None):
        self.project_name = project_name
        self.api_key_path = api_key_path
        # Initialize run
        self.init_run()

    def init_run(self):
        wandb.login(key=self._get_wandb_key())
        self.run = wandb.init(project=self.project_name, name=str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")), config=self.config)

    def wandb_log_metric(self, dict):
        self.run.log(dict)

    def wandb_image(self, plot, caption: str):
        self.run.log({caption: wandb.Image(plot, caption)})