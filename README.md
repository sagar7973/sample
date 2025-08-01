# Professional Training Protocol for T6SE Classification

## Experimental Protocol

### Objective
Evaluate the performance of transformer-based protein language models for binary classification of Type VI Secretion System Effectors (T6SE) versus other secretory proteins using a rigorous cross-validation framework.

### Methodology
- **Model Architecture**: EffectorTransformer with ESM-1b embeddings
- **Evaluation Protocol**: 7-fold independent sampling with 5-fold cross-validation per sample
- **Total Experiments**: 35 independent training-validation cycles
- **Performance Metrics**: F1-score, Accuracy, Precision, Recall, AUC, AUPRC

## Individual Process Execution

### Single Process Training
```bash
python train_t6se_classifier.py \
    --model effectortransformer \
    --t6se_data_path data/t6se_sequences.fasta \
    --non_t6se_data_path data/non_t6se_sequences.fasta \
    --process_id 0 \
    --batch_size 32 \
    --lr 5e-5 \
    --weight_decay 4e-5 \
    --dropout_rate 0.4 \
    --num_layers 1 \
    --num_heads 4 \
    --warm_epochs 10 \
    --patience 15 \
    --lr_scheduler cosine \
    --lr_decay_steps 50 \
    --max_epochs 100 \
    --kfold 5 \
    --n_t6se_samples 240 \
    --n_non_t6se_samples 300 \
    --seed 42 \
    --log_dir experimental_results/process_validation
```

## Complete Experimental Protocol

### Automated Execution Script
```bash
#!/bin/bash
#SBATCH --job-name=t6se_classification
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1

# Professional Training Protocol for T6SE Classification
# Author: [Your Name]
# Institution: [Your Institution]
# Date: $(date)

set -e  # Exit on any error

# Configuration Parameters
readonly EXPERIMENT_NAME="t6se_classification_$(date +%Y%m%d_%H%M%S)"
readonly BASE_LOG_DIR="experimental_results/${EXPERIMENT_NAME}"
readonly T6SE_DATA_PATH="/path/to/processed_data/t6se_dataset.fasta"
readonly NON_T6SE_DATA_PATH="/path/to/processed_data/non_t6se_dataset.fasta"
readonly TOTAL_PROCESSES=7
readonly PYTHON_SCRIPT="train_t6se_classifier.py"

# Create experimental directory structure
mkdir -p "${BASE_LOG_DIR}"
mkdir -p "${BASE_LOG_DIR}/logs"
mkdir -p "${BASE_LOG_DIR}/results"

# Log experimental configuration
cat > "${BASE_LOG_DIR}/experimental_config.txt" << EOF
T6SE Classification Experimental Configuration
============================================
Experiment Name: ${EXPERIMENT_NAME}
Date: $(date)
Host: $(hostname)
User: $(whoami)
Python Version: $(python --version)
PyTorch Version: $(python -c "import torch; print(torch.__version__)")
CUDA Available: $(python -c "import torch; print(torch.cuda.is_available())")

Data Configuration:
- T6SE Data Path: ${T6SE_DATA_PATH}
- Non-T6SE Data Path: ${NON_T6SE_DATA_PATH}
- T6SE Samples per Process: 240
- Non-T6SE Samples per Process: 300

Experimental Design:
- Total Processes: ${TOTAL_PROCESSES}
- Cross-validation Folds: 5
- Total Experiments: $((TOTAL_PROCESSES * 5))
- Random Seed Base: 42

Model Configuration:
- Architecture: EffectorTransformer
- Hidden Dimensions: 512
- Transformer Layers: 1
- Attention Heads: 4
- Dropout Rate: 0.4
- Batch Size: 32
- Learning Rate: 5e-5
- Weight Decay: 4e-5
EOF

echo "======================================================================"
echo "T6SE Classification Experimental Protocol"
echo "======================================================================"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Total Processes: ${TOTAL_PROCESSES}"
echo "Expected Total Experiments: $((TOTAL_PROCESSES * 5))"
echo "Expected Duration: 8-16 hours (GPU) / 40-80 hours (CPU)"
echo "======================================================================"
echo ""

# Validate input files
if [[ ! -f "${T6SE_DATA_PATH}" ]]; then
    echo "ERROR: T6SE data file not found: ${T6SE_DATA_PATH}"
    exit 1
fi

if [[ ! -f "${NON_T6SE_DATA_PATH}" ]]; then
    echo "ERROR: Non-T6SE data file not found: ${NON_T6SE_DATA_PATH}"
    exit 1
fi

if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
    echo "ERROR: Training script not found: ${PYTHON_SCRIPT}"
    exit 1
fi

echo "Input validation completed successfully."
echo ""

# Execute training processes
successful_processes=0
failed_processes=0

for process_id in $(seq 0 $((TOTAL_PROCESSES - 1))); do
    process_number=$((process_id + 1))
    process_start_time=$(date)
    
    echo "======================================================================"
    echo "Executing Process ${process_number}/${TOTAL_PROCESSES}"
    echo "Process ID: ${process_id}"
    echo "Start Time: ${process_start_time}"
    echo "Random Seed: $((42 + process_id * 100))"
    echo "======================================================================"
    
    # Define process-specific log file
    process_log="${BASE_LOG_DIR}/logs/process_${process_number}.log"
    
    # Execute training process
    if python "${PYTHON_SCRIPT}" \
        --model effectortransformer \
        --t6se_data_path "${T6SE_DATA_PATH}" \
        --non_t6se_data_path "${NON_T6SE_DATA_PATH}" \
        --process_id "${process_id}" \
        --batch_size 32 \
        --lr 5e-5 \
        --weight_decay 4e-5 \
        --dropout_rate 0.4 \
        --num_layers 1 \
        --num_heads 4 \
        --warm_epochs 10 \
        --patience 15 \
        --lr_scheduler cosine \
        --lr_decay_steps 50 \
        --max_epochs 100 \
        --kfold 5 \
        --n_t6se_samples 240 \
        --n_non_t6se_samples 300 \
        --seed 42 \
        --log_dir "${BASE_LOG_DIR}" 2>&1 | tee "${process_log}"; then
        
        process_end_time=$(date)
        echo "Process ${process_number} completed successfully at ${process_end_time}"
        ((successful_processes++))
    else
        process_end_time=$(date)
        echo "ERROR: Process ${process_number} failed at ${process_end_time}"
        ((failed_processes++))
    fi
    
    echo ""
done

# Generate comprehensive results summary
echo "======================================================================"
echo "Experimental Protocol Completed"
echo "======================================================================"
echo "Total Processes: ${TOTAL_PROCESSES}"
echo "Successful Processes: ${successful_processes}"
echo "Failed Processes: ${failed_processes}"
echo "Completion Time: $(date)"
echo ""

# Statistical analysis of results
python3 << EOF
import os
import glob
import numpy as np
import json
from pathlib import Path

results_dir = "${BASE_LOG_DIR}"
process_dirs = glob.glob(f"{results_dir}/Process_*")
process_dirs.sort()

print("Statistical Analysis of Experimental Results")
print("=" * 60)

if not process_dirs:
    print("No process results found.")
    exit(1)

all_f1_scores = []
process_summaries = []

for process_dir in process_dirs:
    result_file = f"{process_dir}/cross_validation_results.txt"
    
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            content = f.read()
            
        # Extract statistics
        for line in content.split('\n'):
            if line.startswith('Mean F1 score:'):
                mean_f1 = float(line.split(':')[1].strip())
                all_f1_scores.append(mean_f1)
                
                process_name = os.path.basename(process_dir)
                process_summaries.append({
                    'process': process_name,
                    'mean_f1': mean_f1
                })
                print(f"{process_name}: Mean F1 = {mean_f1:.4f}")

if all_f1_scores:
    overall_mean = np.mean(all_f1_scores)
    overall_std = np.std(all_f1_scores)
    overall_min = np.min(all_f1_scores)
    overall_max = np.max(all_f1_scores)
    
    print("\nOverall Experimental Results:")
    print(f"Number of successful processes: {len(all_f1_scores)}")
    print(f"Mean F1 score: {overall_mean:.4f} ± {overall_std:.4f}")
    print(f"Performance range: [{overall_min:.4f}, {overall_max:.4f}]")
    print(f"Coefficient of variation: {(overall_std/overall_mean)*100:.2f}%")
    
    # Statistical significance assessment
    if overall_std / overall_mean < 0.05:  # CV < 5%
        print("Performance variability: LOW (excellent reproducibility)")
    elif overall_std / overall_mean < 0.10:  # CV < 10%
        print("Performance variability: MODERATE (good reproducibility)")
    else:
        print("Performance variability: HIGH (consider more processes)")
    
    # Save consolidated results
    results_summary = {
        'experiment_name': '${EXPERIMENT_NAME}',
        'total_processes': len(all_f1_scores),
        'total_experiments': len(all_f1_scores) * 5,
        'overall_statistics': {
            'mean_f1': float(overall_mean),
            'std_f1': float(overall_std),
            'min_f1': float(overall_min),
            'max_f1': float(overall_max),
            'coefficient_variation': float((overall_std/overall_mean)*100)
        },
        'process_results': process_summaries
    }
    
    with open(f"{results_dir}/results/experimental_summary.json", 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_dir}/results/experimental_summary.json")
else:
    print("No successful processes found. Check error logs.")

print("=" * 60)
EOF

echo ""
echo "Experimental protocol completed. Results available in: ${BASE_LOG_DIR}"
echo "======================================================================"
```

## Results Analysis and Reporting

### Performance Evaluation Script
```python
#!/usr/bin/env python3
"""
Results Analysis Script for T6SE Classification Experiments

This script analyzes the experimental results and generates publication-ready
performance statistics and visualizations.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

def load_experimental_results(results_path: str) -> dict:
    """Load experimental results from JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)

def generate_performance_report(results: dict) -> str:
    """Generate formatted performance report."""
    stats = results['overall_statistics']
    
    report = f"""
T6SE Classification Performance Report
=====================================

Experimental Configuration:
- Experiment: {results['experiment_name']}
- Total Processes: {results['total_processes']}
- Total Experiments: {results['total_experiments']}

Performance Statistics:
- Mean F1 Score: {stats['mean_f1']:.4f} ± {stats['std_f1']:.4f}
- Performance Range: [{stats['min_f1']:.4f}, {stats['max_f1']:.4f}]
- Coefficient of Variation: {stats['coefficient_variation']:.2f}%

Statistical Assessment:
- Model demonstrates {'excellent' if stats['coefficient_variation'] < 5 else 'good' if stats['coefficient_variation'] < 10 else 'moderate'} reproducibility
- Performance variability is {'within acceptable limits' if stats['coefficient_variation'] < 10 else 'higher than recommended'}

Individual Process Results:"""

    for process in results['process_results']:
        report += f"\n- {process['process']}: F1 = {process['mean_f1']:.4f}"
    
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze T6SE classification results")
    parser.add_argument("--results_path", required=True, help="Path to experimental_summary.json")
    args = parser.parse_args()
    
    results = load_experimental_results(args.results_path)
    report = generate_performance_report(results)
    print(report)
```

## Usage Instructions

1. **Prepare Environment**:
   ```bash
   module load python/3.8
   module load cuda/11.2
   source activate t6se_env
   ```

2. **Execute Protocol**:
   ```bash
   chmod +x run_t6se_classification.sh
   sbatch run_t6se_classification.sh  # For SLURM systems
   # OR
   ./run_t6se_classification.sh      # For direct execution
   ```

3. **Monitor Progress**:
   ```bash
   tail -f experimental_results/*/logs/process_*.log
   ```

4. **Analyze Results**:
   ```bash
   python analyze_results.py --results_path experimental_results/*/results/experimental_summary.json
   ```

This professional framework provides comprehensive logging, error handling, statistical analysis, and reproducible experimental protocols suitable for scientific publication.