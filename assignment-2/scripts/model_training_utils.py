import collections
import matplotlib.pyplot as plt

import numpy as np
import torch
from tqdm import tqdm


def get_accuracy(prediction, label):
    batch_size, n_class = prediction.shape
    predicted_classes = prediction.argmax(dim=1)
    correct_predictions = predicted_classes.eq(label).sum()
    accuracy = correct_predictions / batch_size
    return accuracy


def train(dataloader, model, loss_fn, optimizer, epoch, device, wandb_run):
    model.train()

    epoch_losses = []
    epoch_accuracies = []

    for batch in tqdm(dataloader, desc=f"Training Epoch {epoch + 1}"):
        tweet_ids, tweets, labels = batch

        tweets = tweets.to(device)
        labels = labels.squeeze().to(device)

        predictions = model(tweets)
        accuracy = get_accuracy(predictions, labels).item()

        loss = loss_fn(predictions, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())
        epoch_accuracies.append(accuracy)

    wandb_run.log(
        {
            "epoch": epoch,
            "train_loss": np.mean(epoch_losses),
            "train_accuracy": np.mean(epoch_accuracies),
        }
    )

    return np.mean(epoch_losses), np.mean(epoch_accuracies)


def evaluate(dataloader, model, loss_fn, epoch, device, wandb_run):
    model.eval()

    epoch_losses = []
    epoch_accuracies = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating Epoch {epoch + 1}"):
            tweet_ids, tweets, labels = batch

            tweets = tweets.to(device)
            labels = labels.squeeze().to(device)

            predictions = model(tweets)
            accuracy = get_accuracy(predictions, labels).item()

            loss = loss_fn(predictions, labels)

            epoch_losses.append(loss.item())
            epoch_accuracies.append(accuracy)

    wandb_run.log(
        {
            "epoch": epoch,
            "development_loss": np.mean(epoch_losses),
            "development_accuracy": np.mean(epoch_accuracies),
        }
    )

    return np.mean(epoch_losses), np.mean(epoch_accuracies)


def training_loop(
    n_epochs: int,
    train_dataloader,
    development_dataloader,
    model,
    criterion,
    optimizer,
    device,
    wandb_run,
):
    metrics = collections.defaultdict(list)

    best_valid_loss = float("inf")

    for epoch in range(1, n_epochs + 1):
        train_loss, train_acc = train(
            train_dataloader, model, criterion, optimizer, epoch, device, wandb_run
        )
        valid_loss, valid_acc = evaluate(
            development_dataloader, model, criterion, epoch, device, wandb_run
        )

        metrics["train_losses"].append(train_loss)
        metrics["train_accs"].append(train_acc)

        metrics["valid_losses"].append(valid_loss)
        metrics["valid_accs"].append(valid_acc)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), "../lstm.pt")

        print(f"epoch: {epoch}")
        print(f"train_loss: {train_loss:.3f}, train_acc: {train_acc:.3f}")
        print(f"valid_loss: {valid_loss:.3f}, valid_acc: {valid_acc:.3f}")

    wandb_run.finish()

    return metrics


def plot_metrics(metrics):

    n_epochs = len(metrics["train_losses"])

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(metrics["train_losses"], label="train loss")
    ax.plot(metrics["valid_losses"], label="valid loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_xticks(range(n_epochs))
    ax.legend()
    ax.grid()

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(metrics["train_accs"], label="train accuracy")
    ax.plot(metrics["valid_accs"], label="valid accuracy")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_xticks(range(n_epochs))
    ax.legend()
    ax.grid()
