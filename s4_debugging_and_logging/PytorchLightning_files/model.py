"""
A simple implementation of a CNN on the corrupt MNIST dataset
"""

import wandb
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pytorch_lightning as L
from pytorch_lightning import Trainer, seed_everything, Callback
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

class JoelsAwesomeModel(L.LightningModule):
    """Just my Awesome Model - now adapted for Lightning!"""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            # Input: (batch, 1, 28, 28)
            nn.Conv2d(1, 32, 3, 1),     # Output: (batch, 32, 26, 26) - 1→32 channels, 28-3+1=26 spatial
            nn.ReLU(),
            # Input: (batch, 32, 26, 26)
            nn.Conv2d(32, 64, 3, 1),    # Output: (batch, 64, 24, 24) - 32→64 channels, 26-3+1=24 spatial
            nn.ReLU(),
            # Input: (batch, 64, 24, 24)
            nn.Conv2d(64, 128, 3, 1),   # Output: (batch, 128, 22, 22) - 64→128 channels, 24-3+1=22 spatial
            nn.ReLU(),
            # Input: (batch, 128, 22, 22)
            nn.AdaptiveAvgPool2d(1),    # Output: (batch, 128, 1, 1) - global average pooling
        )

        self.classifier = nn.Sequential(
            # Input: (batch, 128, 1, 1)
            nn.Flatten(),               # Output: (batch, 128)
            nn.Dropout(0.5),            # Output: (batch, 128) - randomly zeroes 50% of activations
            nn.Linear(128, 10)          # Output: (batch, 10) - maps to 10 classes
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        """Forward pass."""
        if x.ndim != 4:
            raise ValueError('Expected input to a 4D tensor')
        if x.shape[1] != 1 or x.shape[2] != 28 or x.shape[3] != 28:
            raise ValueError('Expected each sample to have shape [1, 28, 28]')
        return self.classifier(self.backbone(x))
    
    def training_step(self, batch):
        img, target = batch
        y_pred = self(img)
        loss = self.loss_fn(y_pred, target)
        acc = (target == y_pred.argmax(dim=-1)).float().mean()
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        # self.logger.experiment.log({'logits': wandb.Histrogram(y_pred)})
        return loss
    
    def validation_step(self, batch):
        img, target = batch
        y_pred = self(img)
        loss = self.loss_fn(y_pred, target)
        acc = (target == y_pred.argmax(dim=-1)).float().mean()
        self.log("val_loss", loss, on_epoch=True)
        self.log('val_acc', acc, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)

class MNISTDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage: str):

        train_images = torch.load(f"{self.data_dir}/train_images.pt")
        train_target = torch.load(f"{self.data_dir}/train_target.pt")
        test_images = torch.load(f"{self.data_dir}/test_images.pt")
        test_target = torch.load(f"{self.data_dir}/test_target.pt")

        # Create TensorDatasets from images and targets
        mnist_full = TensorDataset(train_images, train_target)
        mnist_test_full = TensorDataset(test_images, test_target)

        # Split train data into train (90%) and validation (10%)
        self.mnist_train, self.mnist_val = random_split(
            mnist_full, [0.9, 0.1], generator=torch.Generator()
        )
        
        # Assign test dataset
        self.mnist_test = mnist_test_full
        self.mnist_predict = mnist_test_full

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=4, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=2, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size)
    
class PrintCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is started!")
    def on_train_end(self, trainer, pl_module):
        print("Training is done.")


if __name__ == "__main__":
    seed_everything(42, workers=True)   
    model = JoelsAwesomeModel()
    mnist = MNISTDataModule(f'./data')

    checkpoint_callback = ModelCheckpoint(
        dirpath="./models", monitor="val_loss", mode="min"
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=3, verbose=True, mode="min"
    )

    trainer = Trainer(default_root_dir=f'./logs',
                      precision="16-true",
                      max_epochs=10,
                      limit_train_batches=0.2,
                      deterministic=True, 
                      logger=L.loggers.WandbLogger(project="DTU_MLOPS"),
                      callbacks=[PrintCallback(),
                                 early_stopping_callback, 
                                 checkpoint_callback]
    )

    trainer.fit(model, mnist)