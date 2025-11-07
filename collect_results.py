#!/usr/bin/env python3
"""
Grid Search Results Collector
Parses training logs and identifies the best performing model.

Usage: python collect_results.py
"""

import os
import re
import glob
from pathlib import Path
import json


def parse_log_file(log_path):
    """
    Extract key metrics from a training log file.

    Returns dict with:
        - exp_name: experiment name
        - best_val_acc: best validation accuracy achieved
        - final_val_acc: final validation accuracy
        - hyperparameters: dict of lr, epochs, lora_r
        - model_path: path to saved model
        - job_id: SLURM job ID
    """
    with open(log_path, 'r') as f:
        content = f.read()

    result = {
        'log_path': str(log_path),
        'exp_name': None,
        'job_id': None,
        'hyperparameters': {},
        'best_val_acc': None,
        'final_val_acc': None,
        'model_path': None,
        'training_completed': False
    }

    # Extract experiment name from log path (exp1_baseline_12345.log -> exp1_baseline)
    filename = Path(log_path).stem
    if '_' in filename:
        # Remove job ID suffix
        result['exp_name'] = '_'.join(filename.split('_')[:-1])

    # Extract job ID
    job_id_match = re.search(r'Job ID:\s*(\d+)', content)
    if job_id_match:
        result['job_id'] = job_id_match.group(1)

    # Extract hyperparameters
    lr_match = re.search(r'Learning rate:\s*([\d.e-]+)', content)
    epochs_match = re.search(r'Epochs:\s*(\d+)', content)
    lora_r_match = re.search(r'LoRA rank:\s*(\d+)', content)

    if lr_match:
        result['hyperparameters']['learning_rate'] = float(lr_match.group(1))
    if epochs_match:
        result['hyperparameters']['epochs'] = int(epochs_match.group(1))
    if lora_r_match:
        result['hyperparameters']['lora_r'] = int(lora_r_match.group(1))

    # Extract validation accuracies from training progress
    # Look for patterns like: "eval_accuracy': 0.8868"
    val_accs = re.findall(r"'eval_accuracy':\s*([\d.]+)", content)
    if val_accs:
        val_accs = [float(acc) for acc in val_accs]
        result['best_val_acc'] = max(val_accs)
        result['final_val_acc'] = val_accs[-1]

    # Alternative pattern: {'eval_loss': 0.123, 'eval_accuracy': 0.456, ...}
    if not val_accs:
        val_accs_alt = re.findall(r"'eval_accuracy':\s*([\d.]+)", content)
        if val_accs_alt:
            val_accs = [float(acc) for acc in val_accs_alt]
            result['best_val_acc'] = max(val_accs)
            result['final_val_acc'] = val_accs[-1]

    # Check if training completed successfully
    if 'completed at:' in content.lower() or 'training completed' in content.lower():
        result['training_completed'] = True

    # Extract model save path
    if result['exp_name']:
        result['model_path'] = f"data/models/llm-3b_{result['exp_name']}"

    return result


def format_table(results):
    """Format results as a nice ASCII table"""
    if not results:
        return "No results found."

    # Sort by best validation accuracy (descending)
    results = sorted(results, key=lambda x: x['best_val_acc'] if x['best_val_acc'] else 0, reverse=True)

    # Table header
    header = f"{'Rank':<6} {'Experiment':<20} {'LR':<10} {'Epochs':<8} {'LoRA-R':<8} {'Best Val Acc':<14} {'Final Val Acc':<14} {'Status':<10}"
    separator = "=" * len(header)

    lines = [separator, header, separator]

    for rank, result in enumerate(results, 1):
        exp_name = result['exp_name'] or 'Unknown'
        lr = result['hyperparameters'].get('learning_rate', 'N/A')
        epochs = result['hyperparameters'].get('epochs', 'N/A')
        lora_r = result['hyperparameters'].get('lora_r', 'N/A')
        best_acc = f"{result['best_val_acc']:.4f}" if result['best_val_acc'] else "N/A"
        final_acc = f"{result['final_val_acc']:.4f}" if result['final_val_acc'] else "N/A"
        status = "✓ Complete" if result['training_completed'] else "⚠ Running/Failed"

        # Format LR in scientific notation
        if lr != 'N/A':
            lr = f"{lr:.0e}"

        line = f"{rank:<6} {exp_name:<20} {lr:<10} {str(epochs):<8} {str(lora_r):<8} {best_acc:<14} {final_acc:<14} {status:<10}"
        lines.append(line)

    lines.append(separator)
    return '\n'.join(lines)


def main():
    print("="*80)
    print("Grid Search Results Collector")
    print("="*80)
    print()

    # Find all log files
    log_dir = Path("logs/grid_search")
    if not log_dir.exists():
        print(f"Error: Log directory '{log_dir}' not found.")
        return

    log_files = list(log_dir.glob("exp*_*.log"))

    if not log_files:
        print(f"No log files found in '{log_dir}'")
        return

    print(f"Found {len(log_files)} log files. Parsing...")
    print()

    # Parse all logs
    results = []
    for log_file in log_files:
        try:
            result = parse_log_file(log_file)
            results.append(result)
        except Exception as e:
            print(f"Warning: Failed to parse {log_file}: {e}")

    # Filter to only completed experiments for ranking
    completed_results = [r for r in results if r['training_completed'] and r['best_val_acc'] is not None]

    if not completed_results:
        print("No completed experiments with validation accuracy found yet.")
        print("\nAll experiments:")
        print(format_table(results))
        return

    # Display results
    print(format_table(results))
    print()

    # Highlight the winner
    best = max(completed_results, key=lambda x: x['best_val_acc'])
    print("="*80)
    print("🏆 BEST MODEL")
    print("="*80)
    print(f"Experiment:       {best['exp_name']}")
    print(f"Best Val Acc:     {best['best_val_acc']:.4f}")
    print(f"Learning Rate:    {best['hyperparameters'].get('learning_rate', 'N/A')}")
    print(f"Epochs:           {best['hyperparameters'].get('epochs', 'N/A')}")
    print(f"LoRA Rank:        {best['hyperparameters'].get('lora_r', 'N/A')}")
    print(f"Model Path:       {best['model_path']}")
    print("="*80)
    print()

    # Save results to JSON
    results_file = log_dir / "results_summary.json"
    with open(results_file, 'w') as f:
        json.dump({
            'results': results,
            'best_model': best
        }, f, indent=2)

    print(f"Results saved to: {results_file}")
    print()

    # Instructions for using the best model
    print("To use the best model in your pipeline:")
    print(f"  1. Open main.py")
    print(f"  2. Set: CLASSIFIER_PATH = \"{best['model_path']}\"")
    print(f"  3. Set: MODEL_NAME = \"meta-llama/Llama-3.2-3B\"  # (must match training embedder)")
    print()


if __name__ == "__main__":
    main()
