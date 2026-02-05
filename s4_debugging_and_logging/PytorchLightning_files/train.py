import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import typer

from data import corrupt_mnist
from model import JoelsAwesomeModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def train(lr: float = 1e-3, batch_size: int = 32, epochs: int = 5) -> None:
    """Train a model on MNIST."""
    print("Training day and night")
    print(f"{lr=}, {batch_size=}, {epochs=}")

    model = JoelsAwesomeModel().to(DEVICE)
    train_set, _ = corrupt_mnist()

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    statistics = {"train_loss": [], "train_accuracy": []}
    for epoch in range(epochs):
        model.train()
        for i, (img, target) in enumerate(trainloader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(img)
            loss = criterion(y_pred, target)
            loss.backward()
            optimizer.step()
            statistics["train_loss"].append(loss.item())

            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            statistics["train_accuracy"].append(accuracy)

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

    print("Training complete")
    torch.save(model.state_dict(), "models/model.pth")

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    # Calculate moving average loss per epoch

    axs[0].set_title(f"CNN Performance, \n(lr={lr}, epochs={epochs}, batch={batch_size})", fontsize=11)
    ax0_twin = axs[0].twiny()
    axs[0].plot(statistics["train_loss"], alpha=0.7, label="Batch loss")
    epoch_loss = [statistics["train_loss"][i] for i in range(0, len(statistics["train_loss"]), len(trainloader))]
    ax0_twin.plot(range(len(epoch_loss)), epoch_loss, alpha=0.9, linewidth=2, label="Epoch loss", color="orange")
    axs[0].set_ylabel("Loss", fontsize=10)
    axs[0].set_xlabel("Batch", fontsize=10)
    lines1, labels1 = axs[0].get_legend_handles_labels()
    lines2, labels2 = ax0_twin.get_legend_handles_labels()
    axs[0].legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)
    axs[0].grid(alpha=0.15)

    axs[1].plot(statistics["train_accuracy"], alpha=0.7, label="Batch accuracy")
    epoch_accuracy = [
        statistics["train_accuracy"][i] for i in range(0, len(statistics["train_accuracy"]), len(trainloader))
    ]
    ax1_twin = axs[1].twiny()
    ax1_twin.plot(
        range(len(epoch_accuracy)), epoch_accuracy, alpha=0.9, linewidth=2, label="Epoch accuracy", color="orange"
    )
    axs[1].set_title("Train accuracy", fontsize=11)
    axs[1].set_ylabel("Accuracy", fontsize=10)
    axs[1].set_xlabel("Batch", fontsize=10)
    lines3, labels3 = axs[1].get_legend_handles_labels()
    lines4, labels4 = ax1_twin.get_legend_handles_labels()
    axs[1].legend(lines3 + lines4, labels3 + labels4, loc="upper left", fontsize=9)
    axs[1].grid(alpha=0.15)
    fig.savefig("reports/figures/CNN_training_statistics.png")


if __name__ == "__main__":
    typer.run(train)
