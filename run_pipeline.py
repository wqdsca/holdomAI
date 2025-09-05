#!/usr/bin/env python3
"""
Complete Pipeline for Poker Imitation Learning
End-to-end script to download, process, train, and evaluate
"""

import sys
import argparse
from pathlib import Path
import subprocess
import json

def run_command(cmd: list, description: str):
    """Run a command with error handling"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    
    print(result.stdout)
    return True


def main(args):
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║         POKER IMITATION LEARNING PIPELINE                 ║
    ║                                                           ║
    ║  Training an AI to play poker using the PHH dataset      ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Step 1: Download and preprocess data
    if not args.skip_download:
        cmd = ["python", "scripts/preprocess_data.py", "--download", "--analyze"]
        if args.max_hands:
            cmd.extend(["--max-hands", str(args.max_hands)])
        
        if not run_command(cmd, "Step 1: Downloading and preprocessing PHH dataset"):
            print("Failed to preprocess data. Exiting.")
            return
    else:
        print("Skipping download step (--skip-download flag set)")
    
    # Step 2: Train the model
    if not args.skip_train:
        cmd = ["python", "scripts/train.py", "--evaluate"]
        if args.config:
            cmd.extend(["--config", args.config])
        if args.resume:
            cmd.extend(["--resume", args.resume])
        
        if not run_command(cmd, "Step 2: Training imitation learning model"):
            print("Failed to train model. Exiting.")
            return
    else:
        print("Skipping training step (--skip-train flag set)")
    
    # Step 3: Evaluate the model
    if not args.skip_eval:
        cmd = ["python", "scripts/evaluate.py"]
        if args.model_path:
            cmd.extend(["--model", args.model_path])
        
        if not run_command(cmd, "Step 3: Evaluating trained model"):
            print("Failed to evaluate model. Exiting.")
            return
    else:
        print("Skipping evaluation step (--skip-eval flag set)")
    
    # Step 4: Generate visualizations
    if args.visualize:
        cmd = ["python", "scripts/visualize.py"]
        if not run_command(cmd, "Step 4: Generating visualizations"):
            print("Failed to generate visualizations.")
    
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║                   PIPELINE COMPLETE!                      ║
    ║                                                           ║
    ║  Your poker AI has been successfully trained.            ║
    ║  Check the 'models/' directory for the trained model.    ║
    ║  Check the 'results/' directory for evaluation metrics.  ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Print summary
    print("\nSummary:")
    print("-" * 40)
    
    # Check if model exists
    model_path = Path("models/best_model.pt")
    if model_path.exists():
        print(f"✓ Model saved at: {model_path}")
        print(f"  Size: {model_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Check for results
    results_path = Path("results/evaluation_metrics.json")
    if results_path.exists():
        with open(results_path, 'r') as f:
            metrics = json.load(f)
        print(f"✓ Evaluation metrics:")
        print(f"  Action Accuracy: {metrics.get('action_accuracy', 0):.2%}")
        print(f"  Aggression Factor: {metrics.get('aggression_factor', 0):.2f}")
        print(f"  VPIP: {metrics.get('vpip', 0):.2%}")
    
    # Provide next steps
    print("\nNext steps:")
    print("-" * 40)
    print("1. Fine-tune the model: python scripts/train.py --resume models/best_model.pt")
    print("2. Test on new data: python scripts/evaluate.py --model models/best_model.pt --data [path]")
    print("3. Play against the AI: python scripts/play_against_ai.py")
    print("4. Analyze strategies: jupyter notebook notebooks/strategy_analysis.ipynb")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run complete poker imitation learning pipeline")
    
    # Pipeline control
    parser.add_argument('--skip-download', action='store_true',
                       help='Skip data download and preprocessing')
    parser.add_argument('--skip-train', action='store_true',
                       help='Skip model training')
    parser.add_argument('--skip-eval', action='store_true',
                       help='Skip model evaluation')
    
    # Data options
    parser.add_argument('--max-hands', type=int,
                       help='Maximum number of hands to process')
    
    # Training options
    parser.add_argument('--config', type=str,
                       help='Path to training config file')
    parser.add_argument('--resume', type=str,
                       help='Resume training from checkpoint')
    
    # Evaluation options
    parser.add_argument('--model-path', type=str,
                       help='Path to model for evaluation')
    
    # Visualization
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualizations')
    
    args = parser.parse_args()
    main(args)