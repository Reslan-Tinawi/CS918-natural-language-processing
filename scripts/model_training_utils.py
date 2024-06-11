import collections

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score
from tqdm import tqdm
import seaborn as sns


def get_accuracy(y_pred, y_true):
    """
    Calculates the accuracy of the predicted classes.

    Args:
        y_pred (torch.Tensor): Predicted classes tensor of shape (batch_size, n_class).
        y_true (torch.Tensor): True classes tensor of shape (batch_size,).

    Returns:
        float: Accuracy of the predicted classes.

    """
    batch_size, n_class = y_pred.shape
    predicted_classes = y_pred.argmax(dim=1)
    correct_predictions = predicted_classes.eq(y_true).sum()
    accuracy = correct_predictions / batch_size
    return accuracy


def train_one_epoch(
    dataloader, model, loss_fn, optimizer, epoch, device, is_bert, wandb_run
):
    """
    Trains the model for one epoch.

    Args:
        dataloader (torch.utils.data.DataLoader): The data loader for the training dataset.
        model (torch.nn.Module): The model to be trained.
        loss_fn (torch.nn.Module): The loss function used for training.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        epoch (int): The current epoch number.
        device (torch.device): The device on which the training will be performed.
        is_bert (bool): Indicates whether the model is a BERT model or not.
        wandb_run (wandb.Run): The WandB run object for logging.

    Returns:
        Tuple[float, float]: The average loss and accuracy for the epoch.
    """
    # set model to training
    model.train()

    epoch_losses = []
    epoch_accuracies = []

    for batch in tqdm(dataloader, desc=f"Training Epoch {epoch}"):
        if is_bert:
            tweet_ids, tweets_token_ids, attention_mask, labels = batch
            tweets_token_ids = tweets_token_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device, dtype=torch.long)
        else:
            tweet_ids, tweets_token_ids, labels = batch
            tweets_token_ids = tweets_token_ids.to(device)
            labels = labels.squeeze().to(device)

        if is_bert:
            predictions = model(tweets_token_ids, attention_mask)
        else:
            predictions = model(tweets_token_ids)

        # for attention model
        if isinstance(predictions, tuple) and len(predictions) == 2:
            predictions, attention_weights = predictions

        accuracy = get_accuracy(predictions, labels).item()

        loss = loss_fn(predictions, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())
        epoch_accuracies.append(accuracy)

    if wandb_run is not None:
        wandb_run.log(
            {
                "epoch": epoch,
                "train_loss": np.mean(epoch_losses),
                "train_accuracy": np.mean(epoch_accuracies),
            }
        )

    return np.mean(epoch_losses), np.mean(epoch_accuracies)


def evaluate_one_epoch(dataloader, model, loss_fn, epoch, device, is_bert, wandb_run):
    """
    Evaluate the model on one epoch of the given dataloader.

    Args:
        dataloader (torch.utils.data.DataLoader): The dataloader containing the evaluation data.
        model (torch.nn.Module): The model to evaluate.
        loss_fn (torch.nn.Module): The loss function to calculate the loss.
        epoch (int): The current epoch number.
        device (torch.device): The device to perform the evaluation on.
        is_bert (bool): Indicates whether the model is a BERT model or not.
        wandb_run (wandb.Run): The WandB run object to log the evaluation metrics.

    Returns:
        Tuple[float, float]: The average loss and accuracy for the epoch.
    """

    # set model to evaluate
    model.eval()

    epoch_losses = []
    epoch_accuracies = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating Epoch {epoch}"):

            if is_bert:
                tweet_ids, tweets_token_ids, attention_mask, labels = batch
                tweets_token_ids = tweets_token_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device, dtype=torch.long)
            else:
                tweet_ids, tweets_token_ids, labels = batch
                tweets_token_ids = tweets_token_ids.to(device)
                labels = labels.squeeze().to(device)

            if is_bert:
                predictions = model(tweets_token_ids, attention_mask)
            else:
                predictions = model(tweets_token_ids)

            # for attention model
            if isinstance(predictions, tuple) and len(predictions) == 2:
                predictions, _ = predictions

            accuracy = get_accuracy(predictions, labels).item()

            loss = loss_fn(predictions, labels)

            epoch_losses.append(loss.item())
            epoch_accuracies.append(accuracy)

    if wandb_run is not None:

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
    is_bert: bool,
    wandb_run,
    model_path,
):
    metrics = collections.defaultdict(list)

    best_valid_loss = float("inf")

    for epoch in range(1, n_epochs + 1):

        train_loss, train_acc = train_one_epoch(
            train_dataloader,
            model,
            criterion,
            optimizer,
            epoch,
            device,
            is_bert,
            wandb_run,
        )

        valid_loss, valid_acc = evaluate_one_epoch(
            development_dataloader, model, criterion, epoch, device, is_bert, wandb_run
        )

        metrics["train_losses"].append(train_loss)
        metrics["train_accs"].append(train_acc)

        metrics["valid_losses"].append(valid_loss)
        metrics["valid_accs"].append(valid_acc)

        if valid_loss < best_valid_loss:
            print(f"Saving model with valid loss: {valid_loss:.3f}")
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), model_path)

        print(f"epoch: {epoch}")
        print(f"train_loss: {train_loss:.3f}, train_acc: {train_acc:.3f}")
        print(f"valid_loss: {valid_loss:.3f}, valid_acc: {valid_acc:.3f}")

    return metrics


def plot_metrics(metrics, model_name):

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
    plt.show()

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(metrics["train_accs"], label="train accuracy")
    ax.plot(metrics["valid_accs"], label="valid accuracy")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_xticks(range(n_epochs))
    ax.legend()
    ax.grid()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, model_name):

    # computer F1-macro score
    f1_macro = f1_score(y_true, y_pred, average="macro")

    labels = ["positive", "negative", "neutral"]
    # compute confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred, normalize="true", labels=labels)

    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)

    disp.plot(cmap="Blues")

    plt.title(f"{model_name}\nF1-macro: {f1_macro:.3f}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show()


def get_labels_and_predictions(model, dataloader, label_encoder, device, is_bert):
    # get model predictions
    model.eval()

    predictions_list = []
    true_labels_list = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            if is_bert:
                tweet_ids, tweets_token_ids, attention_mask, batch_labels = batch
                tweets_token_ids = tweets_token_ids.to(device)
                attention_mask = attention_mask.to(device)
                batch_labels = batch_labels.to(device, dtype=torch.long)
            else:
                tweet_ids, tweets_token_ids, batch_labels = batch
                tweets_token_ids = tweets_token_ids.to(device)
                batch_labels = batch_labels.squeeze().to(device)

            if is_bert:
                batch_predictions = model(tweets_token_ids, attention_mask)
            else:
                batch_predictions = model(tweets_token_ids)

            # for attention model
            if isinstance(batch_predictions, tuple) and len(batch_predictions) == 2:
                batch_predictions, attention_weights = batch_predictions

            predicted_classes = batch_predictions.argmax(dim=1)

            predictions_list.append(predicted_classes)
            true_labels_list.append(batch_labels)

    y_pred = torch.cat(predictions_list).cpu().numpy()
    y_true = torch.cat(true_labels_list).cpu().numpy()

    # convert labels and predictions to original classes
    y_pred = label_encoder.inverse_transform(y_pred)
    y_true = label_encoder.inverse_transform(y_true)

    return y_true, y_pred
