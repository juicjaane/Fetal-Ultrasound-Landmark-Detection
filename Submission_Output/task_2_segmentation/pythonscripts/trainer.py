import torch
import torch.nn as nn


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device,
        learning_rate=1e-4,
        pos_weight=None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        if pos_weight is not None:
            self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.bce_loss = nn.BCEWithLogitsLoss()

        self.history = {"train_dice": [], "val_dice": []}

    def combined_loss(self, logits, targets):
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice_loss = 1.0 - (2.0 * intersection + 1.0) / (union + 1.0)
        bce = self.bce_loss(logits, targets)
        return 0.5 * bce + 0.5 * dice_loss.mean()

    def dice_metric_soft(self, logits, targets):
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        return ((2.0 * intersection + 1.0) / (union + 1.0)).mean()

    def train(self, epochs, model_path):
        best_val_dice = 0.0
        print(f"Starting training for {epochs} epochs...")

        for epoch in range(epochs):
            self.model.train()
            train_dice = 0.0

            for imgs, masks in self.train_loader:
                imgs, masks = imgs.to(self.device), masks.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(imgs)
                loss = self.combined_loss(logits, masks)
                loss.backward()
                self.optimizer.step()

                train_dice += self.dice_metric_soft(logits, masks).item()

            train_dice /= len(self.train_loader)

            self.model.eval()
            val_dice = 0.0
            with torch.no_grad():
                for imgs, masks in self.val_loader:
                    imgs, masks = imgs.to(self.device), masks.to(self.device)
                    logits = self.model(imgs)
                    val_dice += self.dice_metric_soft(logits, masks).item()

            val_dice /= len(self.val_loader)

            self.history["train_dice"].append(train_dice)
            self.history["val_dice"].append(val_dice)

            print(
                f"Epoch [{epoch+1}/{epochs}] | "
                f"Train Dice: {train_dice:.4f} | Val Dice: {val_dice:.4f}"
            )

            if val_dice > best_val_dice:
                best_val_dice = val_dice
                torch.save(self.model.state_dict(), model_path)
                print(f"✓ Saved best model to {model_path}")

        print(f"Best Validation Dice: {best_val_dice}")
        return self.history
