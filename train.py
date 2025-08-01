#!/usr/bin/env python3
"""
Deep Learning Framework for Type VI Secretion Effector Classification

This script implements a comprehensive evaluation framework for binary classification
of Type VI Secretion System Effectors (T6SE) versus other secretory proteins using
transformer-based protein language models.

The evaluation protocol employs stratified k-fold cross-validation with multiple
independent random sampling iterations to ensure robust performance estimates.

Authors: [Your Name]
Institution: [Your Institution]
Date: [Current Date]
"""

import os
import sys
import time
import logging
import argparse
from datetime import datetime
from typing import Tuple, Dict, Any
import numpy as np

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
from DeepSecE.trainer import train, test, set_seed, EarlyStopping


def setup_logging(log_dir: str, process_id: int) -> logging.Logger:
    """
    Configure logging for the training process.
    
    Args:
        log_dir (str): Directory for log files
        process_id (int): Process identifier for this training run
        
    Returns:
        logging.Logger: Configured logger instance
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, f'training_process_{process_id+1}.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    return logger


def validate_computational_environment() -> Dict[str, Any]:
    """
    Validate and report computational environment specifications.
    
    Returns:
        Dict[str, Any]: Environment specifications
    """
    env_info = {
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'python_version': sys.version,
    }
    
    if torch.cuda.is_available():
        env_info['gpu_name'] = torch.cuda.get_device_name(0)
        env_info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
        env_info['compute_capability'] = f"{torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}"
    
    return env_info


def get_current_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    """
    Extract current learning rate from optimizer.
    
    Args:
        optimizer (torch.optim.Optimizer): Training optimizer
        
    Returns:
        float: Current learning rate
    """
    return optimizer.param_groups[0]['lr']


def execute_single_fold_training(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    early_stopping: EarlyStopping,
    device: torch.device,
    config: argparse.Namespace,
    process_id: int,
    fold_id: int,
    writer: SummaryWriter,
    logger: logging.Logger
) -> float:
    """
    Execute training for a single fold of cross-validation.
    
    Args:
        model (nn.Module): Neural network model
        train_loader (DataLoader): Training data loader
        valid_loader (DataLoader): Validation data loader
        criterion (nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimization algorithm
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler
        early_stopping (EarlyStopping): Early stopping callback
        device (torch.device): Computation device
        config (argparse.Namespace): Training configuration
        process_id (int): Process identifier
        fold_id (int): Fold identifier
        writer (SummaryWriter): TensorBoard writer
        logger (logging.Logger): Logger instance
        
    Returns:
        float: Best F1 score achieved during training
    """
    fold_start_time = time.time()
    experiment_id = process_id * config.kfold + fold_id + 1
    
    logger.info(f"Starting Experiment {experiment_id}: Process {process_id+1}, Fold {fold_id+1}")
    logger.info(f"Training samples: {len(train_loader.dataset)}, "
                f"Validation samples: {len(valid_loader.dataset)}")
    
    best_f1_score = 0.0
    training_history = []
    
    for epoch in range(config.max_epochs):
        epoch_start_time = time.time()
        
        # Training phase
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
        
        # Validation phase
        valid_loss, valid_metrics = test(model, valid_loader, criterion, device)
        
        # Learning rate scheduling
        current_lr = get_current_learning_rate(optimizer)
        scheduler.step()
        
        epoch_duration = time.time() - epoch_start_time
        
        # Record training history
        epoch_record = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'valid_loss': valid_loss,
            'valid_metrics': valid_metrics,
            'learning_rate': current_lr,
            'duration_seconds': epoch_duration
        }
        training_history.append(epoch_record)
        
        # Log progress
        if (epoch + 1) % 10 == 0 or valid_metrics['F1-score'] > best_f1_score:
            logger.info(f"Experiment {experiment_id}, Epoch {epoch+1}: "
                       f"F1={valid_metrics['F1-score']:.4f}, "
                       f"Accuracy={valid_metrics['Accuracy']:.4f}, "
                       f"LR={current_lr:.2e}")
        
        # TensorBoard logging
        writer.add_scalar("Training/Loss", train_loss, epoch + 1)
        writer.add_scalar("Training/Accuracy", train_accuracy, epoch + 1)
        writer.add_scalar("Training/LearningRate", current_lr, epoch + 1)
        writer.add_scalar("Validation/Loss", valid_loss, epoch + 1)
        
        for metric_name, metric_value in valid_metrics.items():
            writer.add_scalar(f"Validation/{metric_name}", metric_value, epoch + 1)
        
        # Update best performance
        if valid_metrics['F1-score'] > best_f1_score:
            best_f1_score = valid_metrics['F1-score']
            logger.info(f"New best F1 score achieved: {best_f1_score:.4f}")
        
        # Early stopping check
        early_stopping(valid_metrics['F1-score'], model)
        if early_stopping.early_stop:
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    fold_duration = time.time() - fold_start_time
    logger.info(f"Experiment {experiment_id} completed: "
               f"Best F1={best_f1_score:.4f}, "
               f"Duration={fold_duration/60:.1f} minutes")
    
    return best_f1_score


def execute_cross_validation_protocol(config: argparse.Namespace) -> None:
    """
    Execute the complete cross-validation training protocol.
    
    Args:
        config (argparse.Namespace): Training configuration parameters
    """
    # Initialize logging
    process_log_dir = os.path.join(config.log_dir, f'Process_{config.process_id+1}')
    logger = setup_logging(process_log_dir, config.process_id)
    
    # Validate environment
    env_info = validate_computational_environment()
    logger.info("=" * 80)
    logger.info("COMPUTATIONAL ENVIRONMENT VALIDATION")
    logger.info("=" * 80)
    for key, value in env_info.items():
        logger.info(f"{key}: {value}")
    
    # Set process-specific random seed
    process_seed = config.seed + config.process_id * 100
    set_seed(process_seed)
    logger.info(f"Random seed set to: {process_seed}")
    
    # Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Computation device: {device}")
    
    if device.type == 'cpu':
        logger.warning("CUDA not available - training will proceed on CPU (significantly slower)")
    else:
        logger.info(f"GPU acceleration enabled: {torch.cuda.get_device_name(0)}")
    
    # Initialize TensorBoard logging
    writer = SummaryWriter(process_log_dir)
    
    # Log experimental configuration
    logger.info("=" * 80)
    logger.info("EXPERIMENTAL CONFIGURATION")
    logger.info("=" * 80)
    logger.info(f"Model architecture: {config.model}")
    logger.info(f"Process identifier: {config.process_id + 1}")
    logger.info(f"Cross-validation folds: {config.kfold}")
    logger.info(f"T6SE samples per process: {config.n_t6se_samples}")
    logger.info(f"Non-T6SE samples per process: {config.n_non_t6se_samples}")
    logger.info(f"Maximum epochs per fold: {config.max_epochs}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Learning rate: {config.lr}")
    logger.info(f"Early stopping patience: {config.patience}")
    logger.info("=" * 80)
    
    # Initialize results storage
    fold_performance_scores = []
    alphabet = Alphabet.from_architecture("roberta_large")
    
    # Execute k-fold cross-validation
    for fold_idx in range(config.kfold):
        logger.info(f"Initiating fold {fold_idx+1}/{config.kfold}")
        
        # Dataset preparation
        try:
            train_dataset = TXSESequenceDataSet(
                t6se_fasta_path=config.t6se_data_path,
                non_t6se_fasta_path=config.non_t6se_data_path,
                transform=label2index,
                mode='train',
                kfold=config.kfold,
                fold_num=fold_idx,
                seed=process_seed,
                n_t6se_samples=config.n_t6se_samples,
                n_non_t6se_samples=config.n_non_t6se_samples
            )
            
            valid_dataset = TXSESequenceDataSet(
                t6se_fasta_path=config.t6se_data_path,
                non_t6se_fasta_path=config.non_t6se_data_path,
                transform=label2index,
                mode='valid',
                kfold=config.kfold,
                fold_num=fold_idx,
                seed=process_seed,
                n_t6se_samples=config.n_t6se_samples,
                n_non_t6se_samples=config.n_non_t6se_samples
            )
            
            logger.info(f"Datasets created - Train: {len(train_dataset)}, "
                       f"Validation: {len(valid_dataset)}")
            
        except Exception as e:
            logger.error(f"Dataset creation failed for fold {fold_idx+1}: {e}")
            continue
        
        # Data loader preparation
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            collate_fn=alphabet.get_batch_converter(),
            num_workers=config.num_workers,
            shuffle=True
        )
        
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=config.batch_size,
            collate_fn=alphabet.get_batch_converter(),
            num_workers=config.num_workers,
            shuffle=False
        )
        
        # Model initialization
        if config.model == "effectortransformer":
            model = EffectorTransformer(
                emb_dim=1280,
                repr_layer=33,
                hid_dim=config.hid_dim,
                num_layers=config.num_layers,
                heads=config.num_heads,
                dropout_rate=config.dropout_rate,
                num_classes=1
            )
        elif config.model == "esm1bmodel":
            model = ESM1bModel(
                emb_dim=1280,
                repr_layer=33,
                unfreeze_last=True,
                hid_dim=config.hid_dim,
                dropout_rate=config.dropout_rate,
                num_classes=1
            )
        else:
            raise ValueError(f"Unsupported model architecture: {config.model}")
        
        model.to(device)
        logger.info(f"Model initialized and transferred to {device}")
        
        # Optimization setup
        criterion = nn.BCEWithLogitsLoss()
        
        if config.model == "esm1bmodel" and hasattr(model, 'pretrained_model'):
            optimizer = optim.Adam([
                {'params': filter(lambda p: p.requires_grad, model.pretrained_model.parameters()),
                 'lr': config.lr / 10},
                {'params': model.clf.parameters(), 'lr': config.lr}
            ], weight_decay=config.weight_decay)
        else:
            optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        
        # Learning rate scheduling
        if config.lr_scheduler == 'cosine':
            after_scheduler = lrs.CosineAnnealingLR(
                optimizer,
                T_max=config.lr_decay_steps,
                eta_min=config.lr_decay_min_lr
            )
        elif config.lr_scheduler == 'step':
            after_scheduler = lrs.StepLR(
                optimizer,
                step_size=config.lr_decay_steps,
                gamma=config.lr_decay_rate
            )
        else:
            after_scheduler = None
        
        if after_scheduler is not None:
            scheduler = GradualWarmupScheduler(
                optimizer,
                multiplier=1,
                total_epoch=config.warm_epochs,
                after_scheduler=after_scheduler
            )
        else:
            scheduler = GradualWarmupScheduler(
                optimizer,
                multiplier=1,
                total_epoch=config.warm_epochs
            )
        
        # Early stopping configuration
        fold_checkpoint_dir = os.path.join(process_log_dir, f'Fold_{fold_idx+1}')
        early_stopping = EarlyStopping(patience=config.patience, checkpoint_dir=fold_checkpoint_dir)
        
        # Execute training for current fold
        fold_f1_score = execute_single_fold_training(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            early_stopping=early_stopping,
            device=device,
            config=config,
            process_id=config.process_id,
            fold_id=fold_idx,
            writer=writer,
            logger=logger
        )
        
        fold_performance_scores.append(fold_f1_score)
        
        # Memory cleanup
        if device.type == 'cuda':
            del model
            torch.cuda.empty_cache()
    
    # Close TensorBoard writer
    writer.close()
    
    # Compute and report final statistics
    if fold_performance_scores:
        mean_f1 = np.mean(fold_performance_scores)
        std_f1 = np.std(fold_performance_scores)
        min_f1 = np.min(fold_performance_scores)
        max_f1 = np.max(fold_performance_scores)
        
        logger.info("=" * 80)
        logger.info("CROSS-VALIDATION RESULTS SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Process {config.process_id + 1} Performance Statistics:")
        logger.info(f"Individual fold F1 scores: {[f'{score:.4f}' for score in fold_performance_scores]}")
        logger.info(f"Mean F1 score: {mean_f1:.4f} ± {std_f1:.4f}")
        logger.info(f"Performance range: [{min_f1:.4f}, {max_f1:.4f}]")
        logger.info(f"Coefficient of variation: {(std_f1/mean_f1)*100:.2f}%")
        logger.info("=" * 80)
        
        # Save results to file
        results_file = os.path.join(process_log_dir, "cross_validation_results.txt")
        with open(results_file, 'w') as f:
            f.write(f"Cross-Validation Results - Process {config.process_id + 1}\n")
            f.write(f"{'='*50}\n")
            f.write(f"Random seed: {process_seed}\n")
            f.write(f"Individual fold F1 scores: {fold_performance_scores}\n")
            f.write(f"Mean F1 score: {mean_f1:.4f}\n")
            f.write(f"Standard deviation: {std_f1:.4f}\n")
            f.write(f"Minimum F1 score: {min_f1:.4f}\n")
            f.write(f"Maximum F1 score: {max_f1:.4f}\n")
            f.write(f"Performance range: {max_f1 - min_f1:.4f}\n")
            f.write(f"Coefficient of variation: {(std_f1/mean_f1)*100:.2f}%\n")
        
        logger.info(f"Results saved to: {results_file}")
    else:
        logger.error("No successful fold completions - unable to compute statistics")


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the training script.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Deep Learning Framework for T6SE Classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data and model configuration
    parser.add_argument('--model', type=str, required=True,
                       choices=['effectortransformer', 'esm1bmodel'],
                       help='Neural network architecture for classification')
    parser.add_argument('--t6se_data_path', type=str, required=True,
                       help='Path to T6SE protein sequences (FASTA format)')
    parser.add_argument('--non_t6se_data_path', type=str, required=True,
                       help='Path to non-T6SE protein sequences (FASTA format)')
    parser.add_argument('--log_dir', type=str, default='./experimental_logs',
                       help='Directory for experimental logs and outputs')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--lr', type=float, default=5e-5,
                       help='Initial learning rate')
    parser.add_argument('--warm_epochs', type=int, default=10,
                       help='Learning rate warmup epochs')
    parser.add_argument('--max_epochs', type=int, default=100,
                       help='Maximum training epochs per fold')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--weight_decay', type=float, default=4e-5,
                       help='L2 regularization coefficient')
    
    # Sampling configuration
    parser.add_argument('--n_t6se_samples', type=int, default=240,
                       help='Number of T6SE samples per process')
    parser.add_argument('--n_non_t6se_samples', type=int, default=300,
                       help='Number of non-T6SE samples per process')
    
    # Cross-validation parameters
    parser.add_argument('--process_id', type=int, required=True,
                       help='Process identifier (0-based indexing)')
    parser.add_argument('--kfold', type=int, default=5,
                       help='Number of cross-validation folds')
    
    # Optimization and scheduling
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience (epochs)')
    parser.add_argument('--lr_scheduler', type=str, default='cosine',
                       choices=['cosine', 'step'],
                       help='Learning rate scheduling strategy')
    parser.add_argument('--lr_decay_steps', type=int, default=50,
                       help='Learning rate decay interval')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                       help='Learning rate decay factor')
    parser.add_argument('--lr_decay_min_lr', type=float, default=1e-7,
                       help='Minimum learning rate threshold')
    
    # Model architecture parameters
    parser.add_argument('--hid_dim', type=int, default=512,
                       help='Hidden layer dimensionality')
    parser.add_argument('--num_layers', type=int, default=1,
                       help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=4,
                       help='Number of attention heads')
    parser.add_argument('--dropout_rate', type=float, default=0.4,
                       help='Dropout probability')
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    return parser.parse_args()


def main():
    """Main execution function."""
    # Parse command line arguments
    config = parse_arguments()
    
    # Validate input files
    if not os.path.exists(config.t6se_data_path):
        raise FileNotFoundError(f"T6SE data file not found: {config.t6se_data_path}")
    if not os.path.exists(config.non_t6se_data_path):
        raise FileNotFoundError(f"Non-T6SE data file not found: {config.non_t6se_data_path}")
    
    # Execute cross-validation protocol
    execute_cross_validation_protocol(config)


if __name__ == "__main__":
    main()