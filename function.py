from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from sklearn import linear_model, model_selection
import os


def accuracy(net, loader):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    """Return accuracy on a dataset given by the data loader."""
    correct = 0
    total = 0
    for inputs, targets in tqdm(loader, desc="Calculating accuracy", leave=False):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = net(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    # print(f"correct: {correct}")
    # print(f"total: {total}")
    return correct / total


def compute_losses(net, loader):
    """Auxiliary function to compute per-sample losses"""
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    criterion = nn.CrossEntropyLoss(reduction="none")
    all_losses = []

    for inputs, targets in tqdm(loader, desc="Computing losses", leave=False):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        logits = net(inputs)
        # losses = criterion(logits, targets).numpy(force=True)
        losses = criterion(logits, targets).detach().cpu().numpy()
        for l in losses:
            all_losses.append(l)

    return np.array(all_losses)


import os


def train_for_n_epochs(
    net, train_loader, n_epochs, checkpoint_dir=None, save_interval=10
):
    """Train the model for n epochs and save checkpoints.

    Args:
      net : nn.Module.
        Model to be trained.
      train_loader : torch.utils.data.DataLoader.
        Dataset loader for the training set.
      n_epochs : int.
        Number of training epochs.
      checkpoint_dir : str or None.
        Directory to save the checkpoints. If None, checkpoints won't be saved.
      save_interval : int.
        Epoch interval to save checkpoints.
    Returns:
      net : trained model.
    """
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    #  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    net.train()

    for epoch in range(n_epochs):
        time.sleep(1)
        # Wrap the train_loader with tqdm for progress monitoring
        for inputs, targets in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{n_epochs}", leave=False
        ):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        #   scheduler.step()

        if checkpoint_dir is not None and (epoch + 1) % save_interval == 0:
            # Save checkpoint every save_interval epochs
            checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch + 1}.pth")
            torch.save(net.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch + 1}.")

    net.eval()
    return net


def simple_mia(sample_loss, members, n_splits=10, random_state=0):
    """Computes cross-validation score of a membership inference attack.

    Args:
      sample_loss : array_like of shape (n,).
        objective function evaluated on n samples.
      members : array_like of shape (n,),
        whether a sample was used for training.
      n_splits: int
        number of splits to use in the cross-validation.
    Returns:
      scores : array_like of size (n_splits,)
    """

    unique_members = np.unique(members)
    if not np.all(unique_members == np.array([0, 1])):
        raise ValueError("members should only have 0 and 1s")

    attack_model = linear_model.LogisticRegression()
    cv = model_selection.StratifiedShuffleSplit(
        n_splits=n_splits, random_state=random_state
    )
    return model_selection.cross_val_score(
        attack_model, sample_loss, members, cv=cv, scoring="accuracy"
    )


def simple_mia_customized(sample_loss, members, n_splits=10, random_state=0):
    """Computes cross-validation score of a membership inference attack.
    return the accuracy, recall, f1 score

    """

    unique_members = np.unique(members)
    if not np.all(unique_members == np.array([0, 1])):
        raise ValueError("members should only have 0 and 1s")

    attack_model = linear_model.LogisticRegression()
    cv = model_selection.StratifiedShuffleSplit(
        n_splits=n_splits, random_state=random_state
    )

    accuracy_scores = model_selection.cross_val_score(
        attack_model, sample_loss.reshape(-1, 1), members, cv=cv, scoring="accuracy"
    )

    recall_scores = model_selection.cross_val_score(
        attack_model, sample_loss.reshape(-1, 1), members, cv=cv, scoring="recall"
    )

    f1_scores = model_selection.cross_val_score(
        attack_model, sample_loss.reshape(-1, 1), members, cv=cv, scoring="f1"
    )

    return {
        "accuracy": np.mean(accuracy_scores),
        "recall": np.mean(recall_scores),
        "f1_score": np.mean(f1_scores),
    }
