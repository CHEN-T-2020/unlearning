import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import time

from function import *


def show_cifar_images(datasets, indices, figsize=(15, 5)):
    # 创建Matplotlib图形和轴对象，行数为1，列数等于数据集数量
    fig, axs = plt.subplots(1, len(datasets), figsize=figsize)

    for i, dataset in enumerate(datasets):
        # 从数据集中获取图像和标签
        image, label = dataset[indices]

        # 在每个子图中显示图像
        axs[i].imshow(image.permute(1, 2, 0))
        axs[i].set_title(f"CIFAR10 Image - Label: {label}")
        axs[i].set_xticks([])
        axs[i].set_yticks([])

    # 调整布局以避免重叠
    plt.tight_layout()

    # 显示Matplotlib图形
    plt.show()


def plot_losses(losses_list, labels, save_path=None):
    plt.title("Losses on Different Sets (pre-trained model)")

    for losses, label in zip(losses_list, labels):
        plt.hist(
            losses,
            density=True,
            alpha=0.5,
            bins=50,
            label=f"{label} (Avg: {np.mean(losses):.4f})",
        )

    plt.xlabel("Loss", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.xlim((0, np.max(np.concatenate(losses_list))))
    plt.yscale("log")
    plt.legend(frameon=False, fontsize=14)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", format="pdf")
        plt.show()
    else:
        plt.show()


def download_checkpoint(checkpoint_dir, net, epoch):
    """Download a specific checkpoint from a specified directory.

    Args:
      checkpoint_dir : str.
        Directory containing checkpoints.
      net : nn.Module.
        Model to load the checkpoint into.
      epoch : int.
        Epoch number to download.
    """
    checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pth")
    net.load_state_dict(torch.load(checkpoint_path))
    print(f"Checkpoint at epoch {epoch} loaded.")


def compute_metrics(net, train_loader, test_loader, forget_loader):
    # Calculate accuracies
    train_acc = accuracy(net, train_loader)
    time.sleep(1)
    test_acc = accuracy(net, test_loader)
    time.sleep(1)

    # Calculate losses
    forget_losses = compute_losses(net, forget_loader)
    time.sleep(1)
    test_losses = compute_losses(net, test_loader)
    time.sleep(1)

    # Calculate MIA scores
    samples_mia = np.concatenate((test_losses, forget_losses)).reshape((-1, 1))
    labels_mia = [0] * len(test_losses) + [1] * len(forget_losses)
    mia_scores = simple_mia_customized(samples_mia, labels_mia)

    return train_acc, test_acc, mia_scores


def compute_and_plot_metrics(
    net, train_loader, test_loader, forget_loader, checkpoint_epochs, checkpoint_dir
):
    train_accuracies = []
    test_accuracies = []
    mia_accuracies = []
    mia_recalls = []
    mia_f1_scores = []

    for epoch in checkpoint_epochs:
        # Download checkpoint
        download_checkpoint(checkpoint_dir, net, epoch)

        # Compute metrics once and reuse
        train_acc, test_acc, mia_scores = compute_metrics(
            net, train_loader, test_loader, forget_loader
        )

        # Store results
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        mia_accuracies.append(mia_scores["accuracy"])
        mia_recalls.append(mia_scores["recall"])
        mia_f1_scores.append(mia_scores["f1_score"])

    # Create two subplots (2 rows, 1 column)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot results in the first subplot
    ax1.plot(checkpoint_epochs, train_accuracies, label="Train Accuracy", marker="s")
    ax1.plot(checkpoint_epochs, test_accuracies, label="Test Accuracy", marker="s")
    ax1.plot(checkpoint_epochs, mia_accuracies, label="MIA Accuracy", marker="o")
    ax1.plot(checkpoint_epochs, mia_recalls, label="MIA Recall", marker="o")
    ax1.plot(checkpoint_epochs, mia_f1_scores, label="MIA F1 Score", marker="o")

    ax1.set_title("Metrics Over Epochs")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Score")
    ax1.set_ylim(0, 1.1)
    ax1.legend()

    # Add a table in the second subplot
    ax2.axis("off")
    table_data = [
        ["Epochs", *checkpoint_epochs],
        ["Train Accuracy", *train_accuracies],
        ["Test Accuracy", *test_accuracies],
        ["MIA Accuracy", *mia_accuracies],
        ["MIA Recall", *mia_recalls],
        ["MIA F1 Score", *mia_f1_scores],
    ]

    table = ax2.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)  # Adjust table size

    # Adjust subplot parameters to give specified padding
    fig.subplots_adjust(hspace=0.5)

    plt.show()
