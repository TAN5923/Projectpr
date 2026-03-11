from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


def get_dataloader(data_dir, batch_size=32, val_split=0.2):
    """
    Loads images from a  PlantVillage folder and returns
    a train loader and a val loader.
    val_split : fraction of data held back for validation (0.2 = 20%)
    """

    # ── Training transform 
    # Augmentation (random flips/rotations) makes the model more robust.
    # Each time an image is loaded for training it looks slightly different.
    # Normalization is required because MobileNetV2 was pretrained on ImageNet
    # which used these exact mean and std values.

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),        # randomly mirror the image
        transforms.RandomRotation(15),            # randomly rotate up to 15 degrees
        transforms.ColorJitter(                   # randomly change brightness/contrast
            brightness=0.2,
            contrast=0.2
        ),
        transforms.ToTensor(),                    # convert image to numbers (0.0 - 1.0)
        transforms.Normalize(                     # shift pixel range to what MobileNetV2 expects
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # ── Validation transform ──────────────────────────────────────────────────
    # No augmentation here — validation should be consistent every time
    # so we get a fair accuracy measurement.

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # ── Load dataset ──────────────────────────────────────────────────────────
    # ImageFolder reads the folder structure automatically.
    # Each subfolder name becomes a class label.
    # e.g. Tomato__Early_blight/ → class "Tomato__Early_blight"

    full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)

    # ── Train / val split ─────────────────────────────────────────────────────
    # Split within each client's own data.
    # 80% used for training, 20% held back to measure accuracy during training.

    val_size   = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Apply the correct transform to each split
    # (val_dataset needs val_transform, not train_transform)
    val_dataset.dataset.transform = val_transform

    # ── DataLoaders ───────────────────────────────────────────────────────────
    # DataLoader serves images in batches during training.
    # shuffle=True for training so the model doesn't see images in the same order
    # shuffle=False for validation so results are consistent

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    print(f"Loaded: {train_size} train images | {val_size} val images")
    print(f"Classes: {len(full_dataset.classes)}")

    return train_loader, val_loader, full_dataset.classes