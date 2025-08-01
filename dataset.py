#!/usr/bin/env python3
"""
Dataset Module for Type VI Secretion Effector Classification

This module implements a specialized dataset class for loading and preprocessing
protein sequences for T6SE vs non-T6SE binary classification tasks. The dataset
supports stratified k-fold cross-validation with controlled random sampling to
ensure balanced representation across different secretory protein types.

Key Features:
- Stratified sampling to maintain class balance
- Cross-validation support with reproducible data splits  
- Comprehensive logging of data preprocessing steps
- Support for both training and validation modes

Authors: [Your Name]
Institution: [Your Institution]
Date: [Current Date]
"""

import os
import logging
from typing import List, Tuple, Optional, Union
import numpy as np
import random

from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold

from esm import FastaBatchedDataset


class TXSESequenceDataSet(Dataset):
    """
    Dataset class for Type VI Secretion Effector protein sequence classification.
    
    This dataset implements stratified sampling and cross-validation protocols
    for robust evaluation of protein sequence classification models.
    
    Attributes:
        t6se_fasta_path (str): Path to T6SE protein sequences
        non_t6se_fasta_path (str): Path to non-T6SE protein sequences
        transform (callable): Optional transform to be applied to labels
        mode (str): Dataset mode ('train', 'valid', or 'test')
        kfold (int): Number of cross-validation folds
        fold_num (int): Current fold index (0-based)
        seed (int): Random seed for reproducibility
        n_t6se_samples (int): Number of T6SE samples to draw
        n_non_t6se_samples (int): Number of non-T6SE samples to draw
    """
    
    def __init__(
        self,
        t6se_fasta_path: str,
        non_t6se_fasta_path: str,
        transform: Optional[callable] = None,
        mode: str = 'train',
        kfold: int = 5,
        fold_num: int = 0,
        seed: int = 42,
        n_t6se_samples: int = 240,
        n_non_t6se_samples: int = 300
    ):
        """
        Initialize the dataset with specified parameters.
        
        Args:
            t6se_fasta_path (str): Path to T6SE sequences file
            non_t6se_fasta_path (str): Path to non-T6SE sequences file
            transform (callable, optional): Label transformation function
            mode (str): Operating mode ('train', 'valid', 'test')
            kfold (int): Number of cross-validation folds
            fold_num (int): Current fold index
            seed (int): Random seed for reproducibility
            n_t6se_samples (int): T6SE samples to draw per iteration
            n_non_t6se_samples (int): Non-T6SE samples to draw per iteration
        """
        self.t6se_fasta_path = t6se_fasta_path
        self.non_t6se_fasta_path = non_t6se_fasta_path
        self.transform = transform
        self.kfold = kfold
        self.mode = mode
        self.fold_num = fold_num
        self.seed = seed
        self.n_t6se_samples = n_t6se_samples
        self.n_non_t6se_samples = n_non_t6se_samples
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Validate input parameters
        self._validate_parameters()
        
        # Load and process the dataset
        self._load_and_process_sequences()
    
    def _validate_parameters(self) -> None:
        """Validate input parameters and file existence."""
        if not os.path.exists(self.t6se_fasta_path):
            raise FileNotFoundError(f"T6SE FASTA file not found: {self.t6se_fasta_path}")
        
        if not os.path.exists(self.non_t6se_fasta_path):
            raise FileNotFoundError(f"Non-T6SE FASTA file not found: {self.non_t6se_fasta_path}")
        
        if self.mode not in ['train', 'valid', 'test']:
            raise ValueError(f"Invalid mode: {self.mode}. Must be 'train', 'valid', or 'test'")
        
        if self.fold_num >= self.kfold:
            raise ValueError(f"fold_num ({self.fold_num}) must be less than kfold ({self.kfold})")
        
        self.logger.info(f"Dataset parameters validated successfully")
        self.logger.info(f"Mode: {self.mode}, Fold: {self.fold_num+1}/{self.kfold}, Seed: {self.seed}")
    
    def _load_sequences_from_fasta(self, fasta_path: str, label: int) -> Tuple[List[str], List[int]]:
        """
        Load protein sequences from FASTA file.
        
        Args:
            fasta_path (str): Path to FASTA file
            label (int): Class label to assign (0 for non-T6SE, 1 for T6SE)
            
        Returns:
            Tuple[List[str], List[int]]: Sequences and corresponding labels
        """
        try:
            dataset = FastaBatchedDataset.from_file(fasta_path)
            sequences = dataset.sequence_strs
            labels = [label] * len(sequences)
            
            self.logger.info(f"Loaded {len(sequences)} sequences from {os.path.basename(fasta_path)}")
            return sequences, labels
            
        except Exception as e:
            self.logger.error(f"Failed to load sequences from {fasta_path}: {e}")
            raise
    
    def _perform_stratified_sampling(
        self,
        t6se_sequences: List[str],
        non_t6se_sequences: List[str]
    ) -> Tuple[List[str], List[int]]:
        """
        Perform stratified random sampling of protein sequences.
        
        Args:
            t6se_sequences (List[str]): T6SE protein sequences
            non_t6se_sequences (List[str]): Non-T6SE protein sequences
            
        Returns:
            Tuple[List[str], List[int]]: Sampled sequences and labels
        """
        # Set random seeds for reproducibility
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        # Sample T6SE sequences
        t6se_indices = list(range(len(t6se_sequences)))
        n_t6se_to_sample = min(self.n_t6se_samples, len(t6se_indices))
        sampled_t6se_indices = random.sample(t6se_indices, n_t6se_to_sample)
        sampled_t6se_sequences = [t6se_sequences[i] for i in sampled_t6se_indices]
        sampled_t6se_labels = [1] * len(sampled_t6se_sequences)
        
        self.logger.info(f"T6SE sampling: {len(sampled_t6se_sequences)}/{len(t6se_sequences)} "
                        f"({len(sampled_t6se_sequences)/len(t6se_sequences)*100:.1f}%)")
        
        # Sample non-T6SE sequences
        non_t6se_indices = list(range(len(non_t6se_sequences)))
        n_non_t6se_to_sample = min(self.n_non_t6se_samples, len(non_t6se_indices))
        sampled_non_t6se_indices = random.sample(non_t6se_indices, n_non_t6se_to_sample)
        sampled_non_t6se_sequences = [non_t6se_sequences[i] for i in sampled_non_t6se_indices]
        sampled_non_t6se_labels = [0] * len(sampled_non_t6se_sequences)
        
        self.logger.info(f"Non-T6SE sampling: {len(sampled_non_t6se_sequences)}/{len(non_t6se_sequences)} "
                        f"({len(sampled_non_t6se_sequences)/len(non_t6se_sequences)*100:.1f}%)")
        
        # Combine and shuffle samples
        combined_sequences = sampled_t6se_sequences + sampled_non_t6se_sequences
        combined_labels = sampled_t6se_labels + sampled_non_t6se_labels
        
        # Shuffle to eliminate ordering bias
        combined_data = list(zip(combined_sequences, combined_labels))
        random.shuffle(combined_data)
        sequences, labels = zip(*combined_data)
        
        self.logger.info(f"Combined dataset: {len(sequences)} sequences")
        self.logger.info(f"Class distribution: {np.bincount(labels)} (Non-T6SE: {labels.count(0)}, T6SE: {labels.count(1)})")
        
        return list(sequences), list(labels)
    
    def _apply_cross_validation_split(
        self,
        sequences: List[str],
        labels: List[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply stratified k-fold cross-validation split.
        
        Args:
            sequences (List[str]): Protein sequences
            labels (List[int]): Corresponding labels
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Selected sequences and labels for current fold
        """
        if self.kfold <= 1:
            self.logger.info("No cross-validation split applied (kfold <= 1)")
            return np.array(sequences), np.array(labels)
        
        # Perform stratified k-fold split
        skf = StratifiedKFold(n_splits=self.kfold, shuffle=True, random_state=self.seed)
        fold_splits = list(skf.split(sequences, labels))
        
        if self.fold_num >= len(fold_splits):
            raise ValueError(f"Fold number {self.fold_num} exceeds available splits {len(fold_splits)}")
        
        train_indices, valid_indices = fold_splits[self.fold_num]
        
        self.logger.info(f"Cross-validation split {self.fold_num+1}/{self.kfold}")
        self.logger.info(f"Training indices: {len(train_indices)}, Validation indices: {len(valid_indices)}")
        
        # Select appropriate split based on mode
        if self.mode == 'train':
            selected_indices = train_indices
            self.logger.info(f"Using training split: {len(selected_indices)} sequences")
        else:  # validation mode
            selected_indices = valid_indices
            self.logger.info(f"Using validation split: {len(selected_indices)} sequences")
        
        selected_sequences = np.array(sequences)[selected_indices]
        selected_labels = np.array(labels)[selected_indices]
        
        # Log final distribution
        final_distribution = np.bincount(selected_labels)
        self.logger.info(f"Final {self.mode} dataset distribution: {final_distribution}")
        
        return selected_sequences, selected_labels
    
    def _load_and_process_sequences(self) -> None:
        """Load and process protein sequences according to specified protocol."""
        self.logger.info("Initiating sequence loading and processing")
        
        # Load sequences from FASTA files
        t6se_sequences, t6se_labels = self._load_sequences_from_fasta(self.t6se_fasta_path, label=1)
        non_t6se_sequences, non_t6se_labels = self._load_sequences_from_fasta(self.non_t6se_fasta_path, label=0)
        
        if self.mode == 'test':
            # For test mode, use all available sequences
            self.sequence_strs = np.array(t6se_sequences + non_t6se_sequences)
            self.sequence_labels = np.array(t6se_labels + non_t6se_labels)
            self.logger.info(f"Test dataset prepared: {len(self.sequence_labels)} sequences")
        else:
            # Perform stratified sampling
            sampled_sequences, sampled_labels = self._perform_stratified_sampling(
                t6se_sequences, non_t6se_sequences
            )
            
            # Apply cross-validation split
            self.sequence_strs, self.sequence_labels = self._apply_cross_validation_split(
                sampled_sequences, sampled_labels
            )
        
        # Log final dataset statistics
        self._log_dataset_statistics()
    
    def _log_dataset_statistics(self) -> None:
        """Log comprehensive dataset statistics."""
        if len(self.sequence_labels) == 0:
            self.logger.warning("Empty dataset created")
            return
        
        unique_labels, label_counts = np.unique(self.sequence_labels, return_counts=True)
        
        self.logger.info("=" * 60)
        self.logger.info("DATASET STATISTICS SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Dataset mode: {self.mode}")
        self.logger.info(f"Total sequences: {len(self.sequence_labels)}")
        self.logger.info(f"Unique labels: {unique_labels}")
        self.logger.info(f"Label distribution: {dict(zip(unique_labels, label_counts))}")
        
        if len(label_counts) == 2:
            class_balance_ratio = label_counts[1] / label_counts[0] if label_counts[0] > 0 else 0
            self.logger.info(f"Class balance ratio (T6SE:Non-T6SE): 1:{1/class_balance_ratio:.2f}")
        
        # Sequence length statistics
        sequence_lengths = [len(seq) for seq in self.sequence_strs]
        self.logger.info(f"Sequence length statistics:")
        self.logger.info(f"  Mean: {np.mean(sequence_lengths):.1f}")
        self.logger.info(f"  Median: {np.median(sequence_lengths):.1f}")
        self.logger.info(f"  Min: {np.min(sequence_lengths)}")
        self.logger.info(f"  Max: {np.max(sequence_lengths)}")
        self.logger.info("=" * 60)
    
    def __getitem__(self, index: int) -> Tuple[Union[int, any], str]:
        """
        Retrieve a single sequence and its label.
        
        Args:
            index (int): Index of the sequence to retrieve
            
        Returns:
            Tuple[Union[int, any], str]: (label, sequence) pair
        """
        if index >= len(self.sequence_labels):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self.sequence_labels)}")
        
        label = self.sequence_labels[index]
        sequence = self.sequence_strs[index]
        
        # Apply label transformation if specified
        if self.transform is not None:
            label = self.transform(label)
        
        return label, sequence
    
    def __len__(self) -> int:
        """
        Return the total number of sequences in the dataset.
        
        Returns:
            int: Number of sequences
        """
        return len(self.sequence_labels)
    
    def get_class_distribution(self) -> dict:
        """
        Get the class distribution of the current dataset.
        
        Returns:
            dict: Class distribution statistics
        """
        unique_labels, counts = np.unique(self.sequence_labels, return_counts=True)
        return dict(zip(unique_labels, counts))
    
    def get_sequence_length_statistics(self) -> dict:
        """
        Get sequence length statistics for the current dataset.
        
        Returns:
            dict: Sequence length statistics
        """
        lengths = [len(seq) for seq in self.sequence_strs]
        return {
            'mean': np.mean(lengths),
            'median': np.median(lengths),
            'std': np.std(lengths),
            'min': np.min(lengths),
            'max': np.max(lengths)
        }