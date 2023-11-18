import matplotlib.pyplot as plt
import numpy as np


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
