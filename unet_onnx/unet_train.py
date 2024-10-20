import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import os
import cv2
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Define constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
LEARNING_RATE = 0.0001
BATCH_SIZE = 16
NUM_EPOCHS = 100
NUM_CLASSES = 2  # Only fire and smoke

# Early stopping parameters
EARLY_STOP_PATIENCE = 10  # Increase from 5 to 10
EARLY_STOP_THRESHOLD = 1e-5  # Decrease from 1e-4 to 1e-5


# Define UNET class
class UNET(nn.Module):
    def __init__(self, in_channels, classes):
        super(UNET, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in [64, 128, 256, 512]:
            self.downs.append(self.__double_conv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed([64, 128, 256, 512]):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(self.__double_conv(feature * 2, feature))

        self.bottleneck = self.__double_conv(512, 1024)
        self.final_conv = nn.Conv2d(64, classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()  # Add sigmoid activation

    def __double_conv(self, in_channels, out_channels):
        conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        return conv

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            # Always resize x to match skip_connection
            x = F.interpolate(
                x, size=skip_connection.shape[2:], mode="bilinear", align_corners=False
            )

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.sigmoid(self.final_conv(x))  # Apply sigmoid activation


# Define CustomDataset class
class CustomDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_size=(256, 256)):
        self.img_dir = img_dir
        self.label_dir = img_dir.replace("images", "labels")
        self.transform = transform
        self.target_size = target_size
        self.images = [
            f for f in os.listdir(img_dir) if f.endswith(".png") or f.endswith(".jpg")
        ]
        self.class_map = {
            0: 0,  # Fire
            1: 1,  # Smoke
        }

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.images[index])
        image = Image.open(img_path).convert("RGB")
        image = image.resize(self.target_size, Image.BILINEAR)

        label_path = os.path.join(
            self.label_dir, os.path.splitext(self.images[index])[0] + ".txt"
        )
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                labels = f.read().strip().split("\n")
            mask = self.create_mask(self.target_size, labels)
        else:
            mask = np.zeros(
                (self.target_size[1], self.target_size[0], NUM_CLASSES),
                dtype=np.float32,
            )

        if self.transform:
            image = self.transform(image)

        mask = torch.from_numpy(mask).float()
        mask = mask.permute(2, 0, 1)  # Change from (H, W, C) to (C, H, W)
        return image, mask

    def create_mask(self, image_size, labels):
        mask = np.zeros((image_size[1], image_size[0], NUM_CLASSES), dtype=np.uint8)
        for label in labels:
            parts = label.split()
            if len(parts) < 5:
                continue
            try:
                class_id = int(parts[0])
                mapped_class_id = self.class_map[class_id]
                polygon = np.array(
                    [float(x) for x in parts[1:]], dtype=np.float32
                ).reshape(-1, 2)
                polygon[:, 0] *= image_size[0]
                polygon[:, 1] *= image_size[1]
                polygon = polygon.astype(np.int32)

                # Create a temporary mask for this polygon
                temp_mask = np.zeros(image_size, dtype=np.uint8)
                cv2.fillPoly(temp_mask, [polygon], color=1)

                # Add the temporary mask to the appropriate channel
                mask[:, :, mapped_class_id] |= temp_mask
            except (ValueError, IndexError, KeyError):
                pass
        return mask.astype(np.float32)


# Define loss functions
def dice_loss(pred, target, smooth=1.0):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = 1 - (
        (2.0 * intersection + smooth)
        / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)
    )
    return loss.mean()


def binary_focal_loss(pred, target, alpha=0.25, gamma=2.0):
    bce_loss = F.binary_cross_entropy(pred, target, reduction="none")
    pt = torch.exp(-bce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * bce_loss
    return focal_loss.mean()


def combined_loss(pred, target):
    focal = binary_focal_loss(pred, target)
    dice = dice_loss(pred, target)
    return focal + dice


# Define training and validation functions
def train_function(data, model, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    epoch_losses = []
    progress_bar = tqdm(data, desc="Training", leave=False)

    for X, y in progress_bar:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        preds = model(X)
        loss = loss_fn(preds, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        epoch_losses.append(loss.item())

        progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(data)
    return avg_loss, epoch_losses


def validate(model, val_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    progress_bar = tqdm(val_loader, desc="Validating", leave=False)

    with torch.no_grad():
        for X, y in progress_bar:
            X, y = X.to(device), y.to(device)
            preds = model(X)
            loss = loss_fn(preds, y)
            total_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

    return total_loss / len(val_loader)


# Initialize model, optimizer, and loss function
unet = UNET(in_channels=3, classes=NUM_CLASSES).to(DEVICE)
optimizer = optim.Adam(unet.parameters(), lr=LEARNING_RATE)
loss_function = combined_loss


def test(model, test_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    total_fire_iou = 0
    total_smoke_iou = 0
    num_samples = 0

    with torch.no_grad():
        for X, y in tqdm(test_loader, desc="Testing"):
            X, y = X.to(device), y.to(device)
            preds = model(X)
            loss = loss_fn(preds, y)
            total_loss += loss.item() * X.size(0)

            # Calculate IoU for fire and smoke
            fire_iou = calculate_iou(preds[:, 0], y[:, 0])
            smoke_iou = calculate_iou(preds[:, 1], y[:, 1])

            total_fire_iou += fire_iou * X.size(0)
            total_smoke_iou += smoke_iou * X.size(0)
            num_samples += X.size(0)

    avg_loss = total_loss / num_samples
    avg_fire_iou = total_fire_iou / num_samples
    avg_smoke_iou = total_smoke_iou / num_samples

    return avg_loss, avg_fire_iou, avg_smoke_iou


def calculate_iou(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection / (union + 1e-6)).item()


def main():
    parser = argparse.ArgumentParser(description="Train U-Net model")
    parser.add_argument(
        "--dataset",
        type=str,
        default=r"/home/hslima/firefront-benchmark/onnx/YOLODataset",
        help="Path to the dataset directory",
    )
    args = parser.parse_args()

    # Create datasets and dataloaders
    train_set = CustomDataset(
        img_dir=os.path.join(args.dataset, "images", "train"),
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
        ]),
        target_size=(256, 256),
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

    val_set = CustomDataset(
        img_dir=os.path.join(args.dataset, "images", "val"),
        transform=transforms.Compose([transforms.ToTensor()]),
        target_size=(256, 256),
    )

    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    test_set = CustomDataset(
        img_dir=os.path.join(args.dataset, "images", "test"),
        transform=transforms.Compose([transforms.ToTensor()]),
        target_size=(256, 256),
    )

    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    # Create optimizer and scheduler
    optimizer = optim.Adam(unet.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    # Training loop
    best_loss = float("inf")
    patience_counter = 0
    all_losses = []

    train_losses = []
    val_losses = []

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        avg_loss, epoch_losses = train_function(train_loader, unet, optimizer, loss_function, DEVICE)
        all_losses.extend(epoch_losses)
        print(f"Training Loss: {avg_loss:.4f}")

        # Validation step
        val_loss = validate(unet, val_loader, loss_function, DEVICE)
        print(f"Validation Loss: {val_loss:.4f}")

        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Print current learning rate
        current_lr = scheduler.optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr}")

        # Early stopping check
        if val_loss < best_loss - EARLY_STOP_THRESHOLD:
            best_loss = val_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': best_loss,
            }, 'best_unet_model.pth')
            print("New best model saved!")
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

        # Calculate and log IoU every 5 epochs
        if (epoch + 1) % 5 == 0:
            _, train_fire_iou, train_smoke_iou = test(unet, train_loader, loss_function, DEVICE)
            _, val_fire_iou, val_smoke_iou = test(unet, val_loader, loss_function, DEVICE)
            print(f"Train Fire IoU: {train_fire_iou:.4f}, Train Smoke IoU: {train_smoke_iou:.4f}")
            print(f"Val Fire IoU: {val_fire_iou:.4f}, Val Smoke IoU: {val_smoke_iou:.4f}")

        train_losses.append(avg_loss)
        val_losses.append(val_loss)

    print("Training completed!")

    # Plot train and validation loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("train_val_loss_curve.png")
    plt.close()

    # After training, evaluate on the test set
    print("Evaluating on test set...")
    test_loss, test_fire_iou, test_smoke_iou = test(
        unet, test_loader, loss_function, DEVICE
    )
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Fire IoU: {test_fire_iou:.4f}")
    print(f"Test Smoke IoU: {test_smoke_iou:.4f}")

    # Save final model
    torch.save(
        {
            "model_state_dict": unet.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "test_loss": test_loss,
            "test_fire_iou": test_fire_iou,
            "test_smoke_iou": test_smoke_iou,
        },
        "final_unet_model.pth",
    )


if __name__ == "__main__":
    main()
