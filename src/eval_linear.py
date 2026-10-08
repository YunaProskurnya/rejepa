import torch
import torch.nn as nn
import torch.nn.functional as F


def evaluate_linear_probe(
    target_encoder,
    train_loader,
    val_loader,
    num_classes,
    device,
    epochs=30,
    lr=1e-3,
):
    target_encoder.to(device)
    target_encoder.eval()

    head = nn.Linear(target_encoder.embed_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)

    for _ in range(epochs):
        for images, labels in train_loader:
            with torch.no_grad():
                features = target_encoder(images.to(device)).mean(dim=1)

            optimizer.zero_grad()
            loss = F.cross_entropy(head(features), labels.to(device))
            loss.backward()
            optimizer.step()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            features = target_encoder(images.to(device)).mean(dim=1)
            predictions = head(features).argmax(dim=1)
            correct += (predictions == labels.to(device)).sum().item()
            total += labels.numel()

    return correct / total
