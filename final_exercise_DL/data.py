import torch
from torchvision import transforms
import matplotlib.pyplot as plt  # only needed for plotting
from mpl_toolkits.axes_grid1 import ImageGrid  # only needed for plotting
import os

DATA_PATH = "./corruptmnist"

if not os.path.exists(DATA_PATH):
    print(os.getcwd())
    raise FileNotFoundError(f"The specified data path does not exist: {DATA_PATH}")

def corrupt_mnist():
    """Return train and test dataloaders for corrupt MNIST."""

    # Download and load the training data
    train_images, train_target = [], []
    for i in range(6):
        try:
            train_images.append(torch.load(f"{DATA_PATH}/train_images_{i}.pt"))
            train_target.append(torch.load(f"{DATA_PATH}/train_target_{i}.pt"))
        except Exception as e:
            print(f"Error loading dataset, retrying... {e}")
    
    train_images = torch.cat(train_images)
    train_target = torch.cat(train_target)

    # Download and load the test data
    test_images: torch.Tensor = torch.load(f"{DATA_PATH}/test_images.pt")
    test_target: torch.Tensor = torch.load(f"{DATA_PATH}/test_target.pt")

    train_images = train_images.unsqueeze(1).float()
    test_images = test_images.unsqueeze(1).float()
    train_target = train_target.long()
    test_target = test_target.long()

    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)

    return train_set, test_set

def show_image_and_target(images: torch.Tensor, target: torch.Tensor) -> None:
    """Plot images and their labels in a grid."""
    row_col = int(len(images) ** 0.5)
    fig = plt.figure(figsize=(10.0, 10.0))
    grid = ImageGrid(fig, 111, nrows_ncols=(row_col, row_col), axes_pad=0.2)
    for ax, im, label in zip(grid, images, target):
        ax.imshow(im.squeeze(), cmap="gray")
        ax.set_title(f"({label.item()})")
        ax.axis("off")
    plt.show()


if __name__ == "__main__":
    train_set, test_set = corrupt_mnist()
    print(f"Size of training set: {len(train_set)}")
    print(f"Size of test set: {len(test_set)}")
    print(f"Shape of a training point {(train_set[0][0].shape, train_set[0][1].shape)}")
    print(f"Shape of a test point {(test_set[0][0].shape, test_set[0][1].shape)}")
    show_image_and_target(test_set.tensors[0][1:100], test_set.tensors[1][1:100])