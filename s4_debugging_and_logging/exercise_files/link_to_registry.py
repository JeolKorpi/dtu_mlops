import torch
import wandb
from vae_mnist_bugs import Model

api = wandb.Api()
artifact_name = f"jeolkorpi-netlight-org/wandb-registry-DTU_MLOPS/vae_model:v0"
artifact = api.artifact(name = artifact_name, type='model')
artifact_dir = artifact.download("wandb-registry-DTU_MLOPS")
model = Model()
model.load_state_dict(torch.load("wandb-registry-DTU_MLOPS/model.ckpt"))