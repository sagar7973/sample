import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
import random

from esm import FastaBatchedDataset


class TXSESequenceDataSet(Dataset):

    def __init__(self, t6se_fasta_path, non_t6se_fasta_path, transform=None, mode='train', 
                 kfold=5, fold_num=0, seed=42, n_t6se_samples=240, n_non_t6se_samples=300):
        self.t6se_fasta_path = t6se_fasta_path
        self.non_t6se_fasta_path = non_t6se_fasta_path
        self.transform = transform
        self.kfold = kfold
        self.mode = mode
        self.fold_num = fold_num
        self.seed = seed
        self.n_t6se_samples = n_t6se_samples
        self.n_non_t6se_samples = n_non_t6se_samples

        self.check_dataset()

    def check_dataset(self):
        # Load T6SE dataset
        t6se_dataset = FastaBatchedDataset.from_file(self.t6se_fasta_path)
        t6se_labels = [1] * len(t6se_dataset.sequence_labels)  # T6SE = 1
        t6se_strs = t6se_dataset.sequence_strs
        
        # Load non-T6SE dataset (contains non-secretory + T1SE + T2SE + T3SE + T4SE)
        non_t6se_dataset = FastaBatchedDataset.from_file(self.non_t6se_fasta_path)
        non_t6se_labels = [0] * len(non_t6se_dataset.sequence_labels)  # non-T6SE = 0
        non_t6se_strs = non_t6se_dataset.sequence_strs

        if self.mode == 'test':
            # For test mode, use all available data
            self.sequence_labels = np.array(t6se_labels + non_t6se_labels)
            self.sequence_strs = np.array(t6se_strs + non_t6se_strs)
        else:
            # Random sampling for training/validation
            random.seed(self.seed)
            np.random.seed(self.seed)
            
            # Sample T6SE sequences
            t6se_indices = list(range(len(t6se_strs)))
            sampled_t6se_indices = random.sample(t6se_indices, 
                                                min(self.n_t6se_samples, len(t6se_indices)))
            sampled_t6se_strs = [t6se_strs[i] for i in sampled_t6se_indices]
            sampled_t6se_labels = [1] * len(sampled_t6se_strs)
            
            # Sample non-T6SE sequences
            non_t6se_indices = list(range(len(non_t6se_strs)))
            sampled_non_t6se_indices = random.sample(non_t6se_indices, 
                                                    min(self.n_non_t6se_samples, len(non_t6se_indices)))
            sampled_non_t6se_strs = [non_t6se_strs[i] for i in sampled_non_t6se_indices]
            sampled_non_t6se_labels = [0] * len(sampled_non_t6se_strs)
            
            # Combine samples
            all_strs = sampled_t6se_strs + sampled_non_t6se_strs
            all_labels = sampled_t6se_labels + sampled_non_t6se_labels
            
            # Stratified K-fold split
            if self.kfold > 1:
                kf = StratifiedKFold(n_splits=self.kfold, shuffle=True, random_state=self.seed)
                splits = list(kf.split(all_strs, all_labels))
                train_idx, valid_idx = splits[self.fold_num]
                
                if self.mode == 'train':
                    self.sequence_labels = np.array(all_labels)[train_idx]
                    self.sequence_strs = np.array(all_strs)[train_idx]
                else:  # validation
                    self.sequence_labels = np.array(all_labels)[valid_idx]
                    self.sequence_strs = np.array(all_strs)[valid_idx]
            else:
                # No k-fold, use all sampled data
                self.sequence_labels = np.array(all_labels)
                self.sequence_strs = np.array(all_strs)

    def __getitem__(self, idx):
        label = self.sequence_labels[idx]
        seq_str = self.sequence_strs[idx]
        if self.transform is not None:
            label = self.transform(label)
        return label, seq_str

    def __len__(self):
        return len(self.sequence_labels)
