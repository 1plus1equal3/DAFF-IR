import torch
from trainer import DaffIRTrainer
from config import *
from inference import infer, visualize_results
from utils import *
from wandb_logger import WandbLogger
from data.dataset import DegradationDataset, build_dataset
from data.utils import get_other_paths, get_haze_paths, get_all_paths
from visualization import visualize
from eval import evaluate
from config import *

#!TODO Prepare dataset here
train_path_ds = get_all_paths(train_path, train_haze_base_path, train_clear_base_path, train=True)
test_path_ds = get_all_paths(test_path, test_haze_base_path, test_clear_base_path, train=False)
train_dataset, test_dataset = build_dataset(train_path_ds, test_path_ds, chosen_degradation, degrade_type=None)

#!TODO Define DaffIRTrainer class here
trainer = DaffIRTrainer()

#!TODO Restore checkpoint
model_dict = {
    'daff_ir': trainer.daffir,
    'guided_filter': trainer.guided_filter,
    'optimizer': trainer.optimizer,
    'gf_optimizer': trainer.gf_optimizer,
    'scheduler': trainer.scheduler,
    'gf_scheduler': trainer.gf_scheduler
}
start_epoch = restore_checkpoint(CKPT_DIR, **model_dict)

# Init Wandb logger
wandb_logger = WandbLogger(PROJECT_NAME, WANDB_KEY_PATHS)

#!TODO Main training loop
######
###########
######
# Training function
def train(epoch, initial_epoch=1, gradient_accumulation_steps=ACCUMULATE_GRADIENT_STEPS, vis_epoch=VIS_EPOCH, vis_num=VIS_NUM, save_epoch=SAVE_EPOCH):
    for step, batch in enumerate(train_dataset):
        # Stop training if we reach the end of the epoch
        if step >= STEPS_PER_EPOCH:
            break
        # Train DAFFIR without adversarial loss
        trainer.train_daffir(batch, step + 1, gradient_accumulation_steps)
        # Logging step
        metric_values = {
            name: metric.compute().item()
            for name, metric in trainer.training_metrics.items()
        }
        logging(step, STEPS_PER_EPOCH, metric_values)
        wandb_logger.wandb_log_metric(metric_values)
    # Reset metrics for the next epoch
    trainer.reset_metrics()
    print('\n')
    # Logout epoch and current learning rate
    param_dict = {
        'epoch': epoch,
        'current_lr': trainer.optimizer.param_groups[0]['lr'],
        'gf_current_lr': trainer.gf_optimizer.param_groups[0]['lr'],
    }
    wandb_logger.wandb_log_metric(param_dict)
    #!FIXME: Visualization
    if epoch % vis_epoch == 0:
        fig = visualize(test_dataset, vis_num)
        wandb_logger.wandb_image(fig, caption=f"Epoch {epoch} Visualization")
    if epoch % save_epoch == 0: 
        save_checkpoint(epoch, **model_dict)


#!TODO START TRAINING HERE
######
###########
######
def main(epochs, train_config, eval_config):
    # Custom training loop
    for epoch in range(train_config['initial_epoch'], epochs):
        try:
            print(f'Epoch {epoch}/{epochs}')
            # Training
            train(epoch, **train_config)
            # Evaluating
            eval_epoch = eval_config['eval_epoch']
            if epoch % eval_epoch == 0:
                evaluate(**eval_config)
        except KeyboardInterrupt:
            print("\nTraining interrupted by user. Saving final checkpoint...")
            save_checkpoint(epoch) # Save the final checkpoint
            print("Final checkpoint saved. Exiting gracefully.")
            print("\n")
            return

if __name__ == "__main__":       
    main(EPOCHS, train_config, eval_config)
