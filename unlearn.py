import torch
from torch import nn
import torch.nn.functional as F
from torch import optim

from tqdm import tqdm
import time
import torchvision
from torchvision import transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def unlearning_blur(net, retain, forget, validation, epochs=1):
    blur = transforms.Compose(
        [
            # transforms.Grayscale(num_output_channels=3),  # 将图像转换为灰度（3通道） ？？？why
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            # transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),  # ColorJitter 以避免输出与输入完全相同
        ]
    )

    less_blur = transforms.Compose(
        [
            transforms.GaussianBlur(kernel_size=5, sigma=(0.05, 1)),  # 减小模糊效果
        ]
    )

    epochs = epochs

    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    net.train()

    for epoch in range(epochs):
        time.sleep(1)
        # Wrap the retain DataLoader with tqdm for progress monitoring
        for inputs, targets in tqdm(
            forget, desc=f"Epoch {epoch + 1}/{epochs}", leave=False
        ):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(inputs)
            outputs = F.softmax(outputs, dim=1)

            # Blur input images
            blurred = less_blur(inputs)
            blurred_outputs = net(blurred)
            blurred_outputs = F.softmax(blurred_outputs, dim=1)

            loss = criterion(outputs, blurred_outputs)
            loss.backward()
            optimizer.step()

        scheduler.step()

    net.eval()
    return net


def unlearning_fine_tune(net, retain, forget, validation, epochs=5):
    """Unlearning by fine-tuning.

    Fine-tuning is a very simple algorithm that trains using only
    the retain set.

    Args:
      net : nn.Module.
        pre-trained model to use as base of unlearning.
      retain : torch.utils.data.DataLoader.
        Dataset loader for access to the retain set. This is the subset
        of the training set that we don't want to forget.
      forget : torch.utils.data.DataLoader.
        Dataset loader for access to the forget set. This is the subset
        of the training set that we want to forget. This method doesn't
        make use of the forget set.
      validation : torch.utils.data.DataLoader.
        Dataset loader for access to the validation set. This method doesn't
        make use of the validation set.
    Returns:
      net : updated model
    """
    epochs = epochs

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    net.train()

    for _ in range(epochs):
        for inputs, targets in retain:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        scheduler.step()
        time.sleep(0.5)

    net.eval()
    return net
