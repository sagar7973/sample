import os
import time
import logging
from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lrs
from sklearn.metrics import confusion_matrix
from tensorboardX import SummaryWriter

from warmup_scheduler import GradualWarmupScheduler
from esm import Alphabet

from DeepSecE.model import EffectorTransformer, ESM1bModel
from DeepSecE.dataset import TXSESequenceDataSet
from DeepSecE.utils import label2index, viz_conf_matrix
from DeepSecE.trainer import train, test, set_seed, LearningRateSaturationStopping


def get_current_lr(optimizer):
    """Get current learning rate from optimizer"""
    for param_group in optimizer.param_groups:
        return param_group['lr']


def run_single_training_iteration(args, iteration_num):
    """Run a single training iteration with random sampling"""
    
    # Set different seed for each iteration to ensure different random sampling
    iteration_seed = args.seed + iteration_num
    set_seed(iteration_seed)

    # Logging
    log_dir = os.path.join(args.log_dir, f'Iteration_{iteration_num+1}', f'Fold_{args.fold_num + 1}')
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        handlers=[logging.FileHandler(os.path.join(log_dir, "training.log"), mode='w', encoding='utf-8')],
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%F %T",
        level=logging.INFO
    )
    writer = SummaryWriter(log_dir)

    logging.info(f"Starting training iteration {iteration_num+1} with seed {iteration_seed}")
    logging.info(f"Sampling {args.n_t6se_samples} T6SE and {args.n_non_t6se_samples} non-T6SE sequences")

    # Device and model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.model == "effectortransformer":
        model = EffectorTransformer(1280, 33, hid_dim=args.hid_dim, num_layers=args.num_layers,
                                    heads=args.num_heads, dropout_rate=args.dropout_rate, num_classes=1)
    elif args.model == "esm1bmodel":
        model = ESM1bModel(1280, 33, unfreeze_last=True, hid_dim=args.hid_dim,
                           dropout_rate=args.dropout_rate, num_classes=1)
    else:
        raise ValueError("Invalid model type!")
    model.to(device)

    # Dataset with random sampling
    alphabet = Alphabet.from_architecture("roberta_large")
    train_dataset = TXSESequenceDataSet(
        t6se_fasta_path=args.t6se_data_path,
        non_t6se_fasta_path=args.non_t6se_data_path,
        transform=label2index, 
        mode='train', 
        kfold=args.kfold,
        fold_num=args.fold_num, 
        seed=iteration_seed,
        n_t6se_samples=args.n_t6se_samples,
        n_non_t6se_samples=args.n_non_t6se_samples
    )
    
    valid_dataset = TXSESequenceDataSet(
        t6se_fasta_path=args.t6se_data_path,
        non_t6se_fasta_path=args.non_t6se_data_path,
        transform=label2index, 
        mode='valid', 
        kfold=args.kfold,
        fold_num=args.fold_num, 
        seed=iteration_seed,
        n_t6se_samples=args.n_t6se_samples,
        n_non_t6se_samples=args.n_non_t6se_samples
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              collate_fn=alphabet.get_batch_converter(), 
                              num_workers=args.num_workers, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size,
                              collate_fn=alphabet.get_batch_converter(), 
                              num_workers=args.num_workers)

    logging.info(f"Training set size: {len(train_dataset)}")
    logging.info(f"Validation set size: {len(valid_dataset)}")

    # Loss & Optimizer
    criterion = nn.BCEWithLogitsLoss()
    if args.model == "esm1bmodel" and hasattr(model, 'pretrained_model'):
        optimizer = optim.Adam([
            {'params': filter(lambda p: p.requires_grad, model.pretrained_model.parameters()), 'lr': args.lr / 10},
            {'params': model.clf.parameters(), 'lr': args.lr}
        ], weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Scheduler
    if args.lr_scheduler is None:
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=args.warm_epochs)
    else:
        if args.lr_scheduler == 'step':
            after_scheduler = lrs.StepLR(optimizer, step_size=args.lr_decay_steps, gamma=args.lr_decay_rate)
        elif args.lr_scheduler == 'cosine':
            after_scheduler = lrs.CosineAnnealingLR(optimizer, T_max=args.lr_decay_steps, eta_min=args.lr_decay_min_lr)
        else:
            raise ValueError("Invalid scheduler type!")
        scheduler = GradualWarmupScheduler(optimizer, 1, args.warm_epochs, after_scheduler)

    # Learning rate saturation stopping
    lr_stopping = LearningRateSaturationStopping(
        patience=args.patience, 
        min_lr_threshold=args.min_lr_threshold,
        checkpoint_dir=log_dir
    )

    # Training Loop
    best_f1 = 0.0
    for epoch in range(args.max_epochs):
        start_time = time.time()

        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        valid_loss, valid_metrics = test(model, valid_loader, criterion, device)

        current_lr = get_current_lr(optimizer)
        scheduler.step()
        epoch_time = time.time() - start_time

        logging.info(f"Epoch {epoch+1:02d} | Time: {epoch_time:.2f}s | LR: {current_lr:.2e}")
        logging.info(f"Train Loss: {train_loss:.4f} | Acc: {train_acc*100:.2f}%")
        logging.info(f"Valid Loss: {valid_loss:.4f} | Acc: {valid_metrics['Accuracy']*100:.2f}%")
        logging.info(f"Valid F1: {valid_metrics['F1-score']:.4f} | AUPRC: {valid_metrics['AUPRC']:.4f}")

        writer.add_scalar("Train/Loss", train_loss, epoch+1)
        writer.add_scalar("Train/Accuracy", train_acc, epoch+1)
        writer.add_scalar("Train/LearningRate", current_lr, epoch+1)
        writer.add_scalar("Valid/Loss", valid_loss, epoch+1)
        for key, value in valid_metrics.items():
            writer.add_scalar("Valid/" + key, value, epoch+1)

        # Check for learning rate saturation
        lr_stopping(valid_metrics["F1-score"], model, current_lr)
        if lr_stopping.early_stop:
            logging.info(f"Training stopped at epoch {epoch+1} due to learning rate saturation or performance plateau")
            break

        if valid_metrics["F1-score"] > best_f1:
            best_f1 = valid_metrics["F1-score"]

    
    logging.info(f"Iteration {iteration_num+1} completed with best F1: {best_f1:.4f}")
    writer.flush()
    writer.close()
    
    return best_f1, log_dir


def main(args):
    """Main training loop with multiple iterations until learning rate saturation"""
    
    overall_log_dir = args.log_dir
    os.makedirs(overall_log_dir, exist_ok=True)
    
    # Overall logging
    overall_logger = logging.getLogger('overall')
    overall_logger.setLevel(logging.INFO)
    handler = logging.FileHandler(os.path.join(overall_log_dir, "overall_training.log"), mode='w')
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%F %T")
    handler.setFormatter(formatter)
    overall_logger.addHandler(handler)
    
    overall_logger.info("Starting multiple training iterations until learning rate saturation")
    overall_logger.info(f"T6SE data: {args.t6se_data_path}")
    overall_logger.info(f"Non-T6SE data: {args.non_t6se_data_path}")
    overall_logger.info(f"Sampling {args.n_t6se_samples} T6SE and {args.n_non_t6se_samples} non-T6SE per iteration")

    iteration = 0
    all_f1_scores = []
    
    while iteration < args.max_iterations:
        overall_logger.info(f"\n=== Starting Iteration {iteration+1} ===")
        
        try:
            best_f1, iteration_log_dir = run_single_training_iteration(args, iteration)
            all_f1_scores.append(best_f1)
            
            overall_logger.info(f"Iteration {iteration+1} completed - Best F1: {best_f1:.4f}")
            
            # Check if we should continue (you can add more sophisticated stopping criteria here)
            if len(all_f1_scores) >= 3:
                recent_scores = all_f1_scores[-3:]
                score_improvement = max(recent_scores) - min(recent_scores)
                if score_improvement < args.convergence_threshold:
                    overall_logger.info(f"Performance converged (improvement < {args.convergence_threshold}). Stopping.")
                    break
            
            iteration += 1
            
        except Exception as e:
            overall_logger.error(f"Error in iteration {iteration+1}: {str(e)}")
            break
    
    overall_logger.info(f"\nTraining completed after {iteration+1} iterations")
    overall_logger.info(f"All F1 scores: {all_f1_scores}")
    
    # --- PATCH START: Prevent crash when all_f1_scores is empty ---
    if all_f1_scores:
        overall_logger.info(f"Best F1 score: {max(all_f1_scores):.4f}")
        overall_logger.info(f"Mean F1 score: {sum(all_f1_scores)/len(all_f1_scores):.4f}")
    else:
        overall_logger.warning("No F1 scores recorded — all training iterations may have failed.")
    # --- PATCH END ---


if __name__ == "__main__":
    parser = ArgumentParser(description="Train binary classifier for T6SE vs non-T6SE prediction with random sampling.")
    
    # Model and data arguments
    parser.add_argument('--model', type=str, choices=['effectortransformer', 'esm1bmodel'], required=True)
    parser.add_argument('--t6se_data_path', type=str, required=True, help='Path to T6SE FASTA file')
    parser.add_argument('--non_t6se_data_path', type=str, required=True, help='Path to non-T6SE FASTA file (non-secretory + T1SE + T2SE + T3SE + T4SE)')
    parser.add_argument('--log_dir', type=str, default='./logs')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--warm_epochs', type=int, default=1)
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--max_iterations', type=int, default=50, help='Maximum number of training iterations')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--weight_decay', type=float, default=4e-5)
    
    # Sampling arguments
    parser.add_argument('--n_t6se_samples', type=int, default=240, help='Number of T6SE samples per iteration')
    parser.add_argument('--n_non_t6se_samples', type=int, default=300, help='Number of non-T6SE samples per iteration')
    
    # Cross-validation arguments
    parser.add_argument('--fold_num', type=int, default=0)
    parser.add_argument('--kfold', type=int, default=5)
    
    # Learning rate and stopping arguments
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--min_lr_threshold', type=float, default=1e-6, help='Minimum LR threshold for saturation')
    parser.add_argument('--convergence_threshold', type=float, default=0.01, help='F1 improvement threshold for convergence')
    parser.add_argument('--lr_scheduler', type=str, default='cosine', choices=['cosine', 'step', None])
    parser.add_argument('--lr_decay_steps', type=int, default=30)
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--lr_decay_min_lr', type=float, default=1e-6)
    
    # Model architecture arguments
    parser.add_argument('--hid_dim', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dropout_rate', type=float, default=0.4)
    
    # Seed
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    main(args)
