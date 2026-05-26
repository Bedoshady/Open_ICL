import os
import csv
import time


class TrainingLogger:
    """
    Simple CSV logger that records per-epoch training metrics.
    
    Writes to ``log_path`` with columns:
        epoch, loss, lr, margin, usb_size, epoch_time_s
    """
    def __init__(self, log_path="logs/training_log.csv"):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        self._epoch_start = None
        self.fieldnames = ["epoch", "loss", "lr", "margin", "usb_size", "epoch_time_s"]

        # Write header (overwrite previous run)
        with open(self.log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def epoch_start(self):
        """Call at the beginning of each epoch."""
        self._epoch_start = time.time()

    def epoch_end(self, epoch, loss, lr, margin, usb_size):
        """Call at the end of each epoch to log metrics."""
        elapsed = time.time() - self._epoch_start if self._epoch_start else 0.0
        row = {
            "epoch": epoch,
            "loss": f"{loss:.6f}",
            "lr": f"{lr:.8f}",
            "margin": f"{margin:.4f}",
            "usb_size": usb_size,
            "epoch_time_s": f"{elapsed:.1f}",
        }
        with open(self.log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)


class EarlyStopping:
    """
    Stops training when the loss has not improved by at least ``min_delta``
    for ``patience`` consecutive epochs.
    """
    def __init__(self, patience=7, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0

    def step(self, loss):
        """
        Returns True if training should stop.
        """
        if self.best_loss is None:
            self.best_loss = loss
            return False

        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
            return False

        self.counter += 1
        if self.counter >= self.patience:
            print(f"[EarlyStopping] No improvement for {self.patience} epochs. Stopping.")
            return True
        return False
