import argparse
import subprocess
import os
import re
import statistics

def run_cmd(cmd):
    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running command:\n{result.stderr}")
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return result.stdout

def main():
    parser = argparse.ArgumentParser(description="Run full pipeline across multiple seeds")
    parser.add_argument('--known_classes', type=str, required=True, help='Comma separated list of known classes')
    parser.add_argument('--split_dir', type=str, required=True, help='Base directory to save checkpoints (e.g. research_papers/new_method/rml2016/set1_random)')
    parser.add_argument('--dataset_type', type=str, default='rml2016', choices=['rml2016', 'rml2018'])
    parser.add_argument('--dataset_path', type=str, default='RML2016.10a_dict.pkl')
    parser.add_argument('--seeds', type=str, default='42,43,44', help='Comma separated list of seeds')
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    
    metrics = {'OA': [], 'F1': [], 'CS_ACC': []}

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"  Starting Pipeline for Seed {seed}")
        print(f"{'='*60}")
        
        checkpoint_dir = os.path.join(args.split_dir, f"seed_{seed}")
        
        # 1. Train
        train_cmd = [
            'python', 'train.py',
            '--known_classes', args.known_classes,
            '--checkpoint_dir', checkpoint_dir,
            '--dataset_type', args.dataset_type,
            '--dataset_path', args.dataset_path,
            '--seed', str(seed)
        ]
        run_cmd(train_cmd)
        
        # 2. Incremental Train
        inc_cmd = [
            'python', 'incremental_train.py',
            '--checkpoint_dir', checkpoint_dir,
            '--dataset_path', args.dataset_path,
            '--seed', str(seed)
        ]
        run_cmd(inc_cmd)
        
        # 3. Evaluate
        eval_cmd = [
            'python', 'evaluate.py',
            '--checkpoint_dir', checkpoint_dir,
            '--dataset_path', args.dataset_path,
            '--dataset_type', args.dataset_type,
            '--seed', str(seed)
        ]
        out = run_cmd(eval_cmd)
        
        print(f"\nEvaluation Output for Seed {seed}:")
        print("-" * 40)
        print(out)
        print("-" * 40)
        
        # Parse metrics
        match = re.search(r'METRICS:\s*OA=([\d.]+),\s*F1=([\d.]+),\s*CS_ACC=([\d.]+)', out)
        if match:
            oa = float(match.group(1))
            f1 = float(match.group(2))
            cs_acc = float(match.group(3))
            
            metrics['OA'].append(oa)
            metrics['F1'].append(f1)
            metrics['CS_ACC'].append(cs_acc)
            print(f"Parsed Metrics -> OA: {oa}%, F1: {f1}, CS_ACC: {cs_acc}%")
        else:
            print("ERROR: Could not parse metrics from evaluate.py output.")
            
    print(f"\n\n{'='*60}")
    print("  FINAL AVERAGED RESULTS")
    print(f"{'='*60}")
    
    if len(metrics['OA']) == len(seeds):
        avg_oa = statistics.mean(metrics['OA'])
        avg_f1 = statistics.mean(metrics['F1'])
        avg_cs = statistics.mean(metrics['CS_ACC'])
        
        print(f"Averaged over seeds: {seeds}")
        print(f"Overall Accuracy (OA) : {avg_oa:.2f}%")
        print(f"Open-Set F1-Score     : {avg_f1:.4f}")
        print(f"Closed-Set Accuracy   : {avg_cs:.2f}%")
    else:
        print("Missing metrics for some runs. Cannot compute averages.")

if __name__ == '__main__':
    main()
