import collections

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score
from tqdm import tqdm


def get_accuracy(prediction, label):
    batch_size, n_class = prediction.shape
    predicted_classes = prediction.argmax(dim=1)
    correct_predictions = predicted_classes.eq(label).sum()
    accuracy = correct_predictions / batch_size
    return accuracy


def train(dataloader, model, loss_fn, optimizer, epoch, device, is_bert, wandb_run):
    model.train()

    epoch_losses = []
    epoch_accuracies = []

    for batch in tqdm(dataloader, desc=f"Training Epoch {epoch}"):
        if is_bert:
            tweet_ids, tweet_token_ids, attention_mask, labels = batch
            tweet_token_ids = tweet_token_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device, dtype=torch.long)
        else:
            tweet_ids, tweets, labels = batch
            tweets = tweets.to(device)
            labels = labels.squeeze().to(device)

        if is_bert:
            prediction = model(tweet_token_ids, attention_mask)
        else:
            predictions = model(tweets)

        # for attention model
        if isinstance(predictions, tuple):
            predictions, _ = predictions

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


def evaluate(dataloader, model, loss_fn, epoch, device, is_bert, wandb_run):
    model.eval()

    epoch_losses = []
    epoch_accuracies = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating Epoch {epoch}"):

            if is_bert:
                tweet_ids, tweet_token_ids, attention_mask, labels = batch
                tweet_token_ids = tweet_token_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device, dtype=torch.long)
            else:
                tweet_ids, tweets, labels = batch
                tweets = tweets.to(device)
                labels = labels.squeeze().to(device)

            if is_bert:
                predictions = model(tweets, attention_mask)
            else:
                predictions = model(tweets)

            # for attention model
            if isinstance(predictions, tuple):
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
    is_bert,
    wandb_run,
    model_path
):
    metrics = collections.defaultdict(list)

    best_valid_loss = float("inf")

    for epoch in range(1, n_epochs + 1):

        train_loss, train_acc = train(
            train_dataloader,
            model,
            criterion,
            optimizer,
            epoch,
            device,
            is_bert,
            wandb_run,
        )

        valid_loss, valid_acc = evaluate(
            development_dataloader, model, criterion, epoch, device, is_bert, wandb_run
        )

        metrics["train_losses"].append(train_loss)
        metrics["train_accs"].append(train_acc)

        metrics["valid_losses"].append(valid_loss)
        metrics["valid_accs"].append(valid_acc)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), model_path)

        print(f"epoch: {epoch}")
        print(f"train_loss: {train_loss:.3f}, train_acc: {train_acc:.3f}")
        print(f"valid_loss: {valid_loss:.3f}, valid_acc: {valid_acc:.3f}")

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


def plot_classification_results(model, dataloader, label_encoder, device):
    # get model predictions
    model.eval()

    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            tweet_ids, tweets, labels = batch

            tweets = tweets.to(device)
            labels = labels.squeeze().to(device)

            output = model(tweets)

            # for attention model
            if isinstance(output, tuple):
                output, _ = output

            output = output.argmax(dim=1)

            predictions.append(output)
            true_labels.append(labels)

    predictions = torch.cat(predictions).cpu().numpy()
    true_labels = torch.cat(true_labels).cpu().numpy()

    # convert labels and predictions to original classes
    predictions = label_encoder.inverse_transform(predictions)
    true_labels = label_encoder.inverse_transform(true_labels)

    # computer F1-macro score
    f1_macro = f1_score(true_labels, predictions, average="macro")

    cm = confusion_matrix(
        true_labels, predictions, labels=label_encoder.classes_, normalize="true"
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=label_encoder.classes_
    )

    disp.plot(cmap="Blues")

    # set title
    plt.title(f"F1-macro: {f1_macro:.3f}")


def visualize_attention(input_tokens, attention_weights):
    # Convert attention weights to numpy array if not already
    attention_weights = attention_weights.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(len(input_tokens) / 2, 2))
    heatmap = ax.pcolor(attention_weights, cmap=plt.cm.Blues)

    # Set the labels
    ax.set_xticklabels(input_tokens, minor=False, rotation="vertical")
    ax.set_yticklabels(["Attention"], minor=False)

    # Move ticks to the center
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")

    plt.xticks(np.arange(len(input_tokens)) + 0.5, input_tokens)
    plt.yticks(np.arange(1) + 0.5, ["Attention"])

    plt.colorbar(heatmap)
    plt.show()
