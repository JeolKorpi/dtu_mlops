"""Adapted from https://github.com/Jackson-Kang/Pytorch-VAE-tutorial/blob/master/01_Variational_AutoEncoder.ipynb.

A simple implementation of Gaussian MLP Encoder and Decoder trained on MNIST
"""

import wandb
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.utils import save_image
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
from pathlib import Path
import yaml

# Use profiler schedule to limit profiling overhead
from torch.profiler import schedule

# Load sweep configuration from YAML file
with open("sweep.yaml") as f:
    sweep_config = yaml.safe_load(f)

###### UNCOMMENT FOR DEBUGGING ######
# import pdb
# pdb.set_trace()

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


class Encoder(nn.Module):
    """Gaussian MLP Encoder."""

    def __init__(self, input_dim, hidden_dim, latent_dim) -> None:
        super().__init__()

        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_mean = nn.Linear(hidden_dim, latent_dim)
        self.FC_var = nn.Linear(hidden_dim, latent_dim)
        self.training = True

    def forward(self, x):
        """Forward pass of the encoder module."""
        h_ = torch.relu(self.FC_input(x))
        mean = self.FC_mean(h_)
        log_var = self.FC_var(h_)
        std = torch.exp(0.5 * log_var)
        z = self.reparameterization(mean, std)
        return z, mean, log_var

    def reparameterization(self, mean, std):
        """Reparameterization trick to sample z values."""
        epsilon = torch.randn_like(std)
        return mean + std * epsilon


class Decoder(nn.Module):
    """Decoder module for VAE."""

    def __init__(self, latent_dim, hidden_dim, output_dim) -> None:
        super().__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """Forward pass of the decoder module."""
        h = torch.relu(self.FC_hidden(x))
        return torch.sigmoid(self.FC_output(h))


class Model(nn.Module):
    """VAE Model."""

    def __init__(self, encoder, decoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        """Forward pass of the VAE model."""
        z, mean, log_var = self.encoder(x)
        x_hat = self.decoder(z)

        return x_hat, mean, log_var


if __name__ == "__main__":
    wandb.login()

    project = "M14_module"

    config = {
        "batch_size": 100,
        "x_dim": 784,
        "hidden_dim": 400,
        "latent_dim": 20,
        "lr": 1e-3,
        "epochs": 5,
    }

    run = wandb.init(project=project, config=config)

    with run:
        # Model Hyperparameters
        dataset_path = "datasets"

        # Data loading
        mnist_transform = transforms.Compose([transforms.ToTensor()])

        train_dataset = MNIST(
            dataset_path, transform=mnist_transform, train=True, download=True
        )
        test_dataset = MNIST(
            dataset_path, transform=mnist_transform, train=False, download=True
        )

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=4,
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=4,
        )

        encoder = Encoder(
            input_dim=config["x_dim"],
            hidden_dim=config["hidden_dim"],
            latent_dim=config["latent_dim"],
        )
        decoder = Decoder(
            latent_dim=config["latent_dim"],
            hidden_dim=config["hidden_dim"],
            output_dim=config["x_dim"],
        )

        model = Model(encoder=encoder, decoder=decoder).to(DEVICE)

        BCE_loss = nn.BCELoss()

        def loss_function(x, x_hat, mean, log_var):
            """Elbo loss function."""
            reproduction_loss = nn.functional.binary_cross_entropy(
                x_hat, x, reduction="sum"
            )
            kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
            return reproduction_loss + kld, reproduction_loss, kld

        optimizer = Adam(model.parameters(), lr=config["lr"])

        # Create log directory for profiler traces
        log_dir = Path("./log/vae_profile")
        log_dir.mkdir(parents=True, exist_ok=True)

        # Profile activities based on device availability
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

        print("Start training VAE...")
        model.train()

        with profile(
            activities=activities,
            schedule=schedule(wait=1, warmup=1, active=3, repeat=2),
            profile_memory=True,
            with_stack=True,
            on_trace_ready=tensorboard_trace_handler(str(log_dir) + "_train"),
        ) as prof:
            for epoch in range(config["epochs"]):
                overall_loss = 0
                for batch_idx, (x, _) in enumerate(train_loader):
                    if batch_idx % 100 == 0:
                        print(batch_idx)
                    x = x.view(config["batch_size"], config["x_dim"])
                    x = x.to(DEVICE)

                    x_hat, mean, log_var = model(x)
                    loss, reproduction_loss, kld = loss_function(
                        x, x_hat, mean, log_var
                    )

                    overall_loss += loss.item()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Calculate reconstruction accuracy (% of pixels correctly reconstructed)
                    x_hat_binary = (x_hat > 0.5).float()
                    accuracy = (x_hat_binary == x).float().mean().item()

                    run.log(
                        {
                            "train_loss": loss.item(),
                            "train_reconstruction_loss": reproduction_loss.item(),
                            "train_kld": kld.item(),
                            "train_reconstruction_accuracy": accuracy,
                        }
                    )
                    # Only profile first 20 batches to avoid overwhelming the profiler
                    if batch_idx < 20:
                        prof.step()
                print(
                    "\tEpoch",
                    epoch + 1,
                    "complete!",
                    "\tAverage Loss: ",
                    overall_loss / (batch_idx * config["batch_size"]),
                )
        print("Finish!!")

        # Generate reconstructions
        print("Generating reconstructions and samples...")
        model.eval()
        with profile(
            activities=activities,
            profile_memory=True,
            with_stack=True,
            on_trace_ready=tensorboard_trace_handler(str(log_dir) + "_eval"),
        ) as prof:
            with torch.no_grad():
                for batch_idx, (x, _) in enumerate(test_loader):
                    if batch_idx % 100 == 0:
                        print(batch_idx)
                    x = x.view(config["batch_size"], config["x_dim"])
                    x = x.to(DEVICE)
                    x_hat, _, _ = model(x)
                    loss, reproduction_loss, kld = loss_function(x, x_hat, _, _)
                    run.log({"reconstruction_example_loss": loss.item()})
                    prof.step()
                    break

            save_image(x.view(config["batch_size"], 1, 28, 28), "orig_data.png")
            save_image(
                x_hat.view(config["batch_size"], 1, 28, 28), "reconstructions.png"
            )

            # Generate samples
            noise = torch.randn(config["batch_size"], config["latent_dim"]).to(DEVICE)
            generated_images = decoder(noise)
            prof.step()

        save_image(
            generated_images.view(config["batch_size"], 1, 28, 28),
            "generated_sample.png",
        )

        # Log images to W&B
        run.log({
            "original_images": wandb.Image("orig_data.png"),
            "reconstructed_images": wandb.Image("reconstructions.png"),
            "generated_samples": wandb.Image("generated_sample.png"),
        })

        # Save model as W&B artifact
        torch.save(model.state_dict(), "model.pth")
        artifact = wandb.Artifact(
            name="vae_mnist_model",
            type="model",
            description="A VAE model trained on MNIST digits",
            metadata={
                "latent_dim": config["latent_dim"],
                "hidden_dim": config["hidden_dim"],
                "epochs": config["epochs"],
                "learning_rate": config["lr"],
            },
        )
        artifact.add_file("model.pth")
        run.log_artifact(artifact)
        run.finish()
    
    print("Over and out.")