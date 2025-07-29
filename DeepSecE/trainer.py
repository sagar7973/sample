import os
import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from collections import deque

from DeepSecE.utils import metrics


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class LearningRateSaturationStopping:
    """Early stopping based on learning rate saturation"""
    def __init__(self, patience: int = 10, min_lr_threshold: float = 1e-6, 
                 checkpoint_dir: str = 'logs', window_size: int = 5):
        self.patience = patience
        self.min_lr_threshold = min_lr_threshold
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.checkpoint_dir = checkpoint_dir
        self.window_size = window_size
        self.lr_history = deque(maxlen=window_size)
        self.score_history = deque(maxlen=window_size)

    def __call__(self, score, model, current_lr, goal: str = "maximize"):
        """
        score: the metric to monitor (e.g., F1)
        current_lr: current learning rate
        goal: "maximize" (default) or "minimize"
        """
        if goal == "minimize":
            score = -score

        self.lr_history.append(current_lr)
        self.score_history.append(score)

        # Check if learning rate has saturated
        if current_lr <= self.min_lr_threshold:
            print(f"Learning rate saturated: {current_lr} <= {self.min_lr_threshold}")
            self.early_stop = True
            return

        # Check if performance has plateaued despite lr changes
        if len(self.score_history) >= self.window_size:
            score_std = np.std(list(self.score_history))
            if score_std < 1e-4:  # Very small variation in scores
                print(f"Performance plateaued with score std: {score_std}")
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
                    return

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(self.checkpoint_dir, 'checkpoint.pt'))


class EarlyStopping:
    def __init__(self, patience: int = 10, checkpoint_dir: str = 'logs'):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.checkpoint_dir = checkpoint_dir

    def __call__(self, score, model, goal: str = "maximize"):
        """
        score: the metric to monitor (e.g., F1)
        goal:  "maximize" (default) or "minimize"
        """
        if goal == "minimize":
            score = -score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(self.checkpoint_dir, 'checkpoint.pt'))


def _prepare_targets_and_preds_for_binary(logits: torch.Tensor, labels, device: torch.device):
    """
    Returns:
        y_float:   float tensor same shape as logits for BCEWithLogitsLoss
        y_int:     int tensor (N,) for accuracy/metrics
        prob_2c:   probs as (N, 2) for metrics(truth, preds, probs)
        pred_int:  predicted labels (N,) as ints
    """
    # logits can be (N, 1) or (N,) -> flatten to (N,)
    logits_flat = logits.view(-1)
    y_float = torch.tensor(labels, device=device, dtype=torch.float32).view_as(logits_flat)
    y_int = y_float.long()

    prob_pos = torch.sigmoid(logits_flat)                      # (N,)
    pred_int = (prob_pos > 0.5).long()                         # (N,)
    prob_2c = torch.stack([1.0 - prob_pos, prob_pos], dim=1)   # (N, 2)

    return y_float, y_int, prob_2c, pred_int


def train(model, iterator, criterion, optimizer, device):
    model.train()
    avg_loss = 0.0
    avg_acc = 0.0
    data_size = 0

    for labels, strs, toks in iterator:
        toks = toks.to(device)
        logits = model(strs, toks)

        # Binary vs multi-class branch based on logits' last dimension
        if logits.dim() == 1 or logits.shape[-1] == 1:
            # ----- Binary classification -----
            y_float, y_int, _, pred_int = _prepare_targets_and_preds_for_binary(logits, labels, device)
            # reshape logits to match y_float
            loss = criterion(logits.view_as(y_float), y_float)
            acc = accuracy_score(y_int.cpu().numpy(), pred_int.cpu().numpy())
            batch_size = y_int.size(0)
        else:
            # ----- Multi-class classification -----
            y = torch.tensor(labels, device=device, dtype=torch.long)
            loss = criterion(logits, y)
            prob = torch.softmax(logits, dim=1)
            _, pred = torch.max(prob, dim=1)
            acc = accuracy_score(y.cpu().numpy(), pred.cpu().numpy())
            batch_size = y.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss += loss.item() * batch_size
        avg_acc += acc * batch_size
        data_size += batch_size

    return avg_loss / data_size, avg_acc / data_size


def test(model, iterator, criterion, device, return_array: bool = False):
    model.eval()
    avg_loss = 0.0
    data_size = 0

    truth, probs, preds = [], [], []

    with torch.no_grad():
        for labels, strs, toks in iterator:
            toks = toks.to(device)
            logits = model(strs, toks)

            if logits.dim() == 1 or logits.shape[-1] == 1:
                # ----- Binary classification -----
                y_float, y_int, prob_2c, pred_int = _prepare_targets_and_preds_for_binary(logits, labels, device)
                loss = criterion(logits.view_as(y_float), y_float)
                truth.append(y_int.cpu().numpy())
                probs.append(prob_2c.cpu().numpy())
                preds.append(pred_int.cpu().numpy())
                batch_size = y_int.size(0)
            else:
                # ----- Multi-class classification -----
                y = torch.tensor(labels, device=device, dtype=torch.long)
                loss = criterion(logits, y)
                prob = torch.softmax(logits, dim=1)
                _, pred = torch.max(prob, dim=1)
                truth.append(y.cpu().numpy())
                probs.append(prob.cpu().numpy())
                preds.append(pred.cpu().numpy())
                batch_size = y.size(0)

            avg_loss += loss.item() * batch_size
            data_size += batch_size

    truth = np.concatenate(truth)
    probs = np.concatenate(probs)
    preds = np.concatenate(preds)

    metrics_dict = metrics(truth, preds, probs)

    if return_array:
        return avg_loss / data_size, metrics_dict, truth, preds
    return avg_loss / data_size, metrics_dict
