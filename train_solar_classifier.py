import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from PIL import Image
import cv2


def get_solar_root() -> Path:
    """
    Return the local folder where you unzipped the Planets & Moons dataset.
    Folder structure:
        Planet and Moons/
            Earth/
            Jupiter/
            Makemake/
            Mars/
            Mercury/
            Moon/
            Neptune/
            Pluto/
            Saturn/
            Uranus/
            Venus/
    """
    root = Path.home() / "Downloads" / "Planets and Moons"
    print("Solar System dataset root:", root)

    print("Immediate subfolders (candidate classes):")
    for p in root.iterdir():
        if p.is_dir():
            print("  -", p.name)

    return root


def cv2_loader(path: str):
    """
    Load an image with OpenCV and convert to a PIL Image (RGB).
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"[WARN] Could not read image, using dummy: {path}")
        return Image.new("RGB", (224, 224), (0, 0, 0))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)


def build_datasets(root: Path, val_frac: float = 0.2):
    """
    Build datasets using torchvision.datasets.ImageFolder.

    We want the *object-level* folders (Earth, Mars, Moon, etc.) to be classes,
    so we point ImageFolder directly at the folder that contains those.
    """
    data_root = root

    print("Using data_root:", data_root)

    train_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    val_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    full_dataset = datasets.ImageFolder(
        root=str(data_root),
        transform=train_tfms,
        loader=cv2_loader,
    )

    class_names = full_dataset.classes
    print("Solar System classes:", class_names)

    val_size = int(len(full_dataset) * val_frac)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    val_dataset.dataset.transform = val_tfms

    print(f"[Solar] Train size: {train_size}, Val size: {val_size}")
    return train_dataset, val_dataset, class_names


def train_model(train_ds, val_ds, class_names,
                epochs: int = 20, batch_size: int = 32,
                lr: float = 1e-4, device: str = "cpu"):

    num_classes = len(class_names)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                preds = out.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total if total > 0 else 0.0
        print(f"[Solar] Epoch {epoch+1}/{epochs} - "
              f"train loss: {train_loss:.4f}, val acc: {val_acc:.3f}")

    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    solar_root = get_solar_root()

    train_ds, val_ds, class_names = build_datasets(
        solar_root,
        val_frac=0.2,
    )

    model = train_model(
        train_ds,
        val_ds,
        class_names,
        epochs=20,
        batch_size=32,
        lr=1e-4,
        device=device,
    )

    os.makedirs("models", exist_ok=True)

    save_path = "models/solar_system_classifier.pth"
    torch.save(
        {
            "model_state": model.state_dict(),
            "class_names": class_names,
        },
        save_path,
    )
    print("Saved Solar System model to", save_path)


if __name__ == "__main__":
    main()