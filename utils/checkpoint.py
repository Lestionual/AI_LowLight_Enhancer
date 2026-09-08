import torch


def save_checkpoint(model, optimizer, epoch, loss, filename):

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }

    torch.save(checkpoint, filename)


def load_checkpoint(
    model,
    optimizer,
    filename,
    device,
    load_optimizer=True
):

    checkpoint = torch.load(
        filename,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if load_optimizer:

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]

    return epoch, loss