import wandb
import yaml
import os
from train import train_sweep
# Load YAML file into a nested dictionary
def read_yaml(file_path):
    with open(file_path, 'r') as yaml_file:
        nested_dict = yaml.safe_load(yaml_file)
    return nested_dict

configs_dir = os.path.join('configs')
if __name__ == "__main__":
    yaml_file_path = os.path.join(configs_dir, 'sweep_config.yaml')
    nested_dict = read_yaml(yaml_file_path)
    sweep_id = wandb.sweep(nested_dict, project="gaussian_splattingUW")
    print(f'Created sweep with ID: {sweep_id}')
    # run the sweep
    wandb.agent(sweep_id, function=train_sweep, count=2)
