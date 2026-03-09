#!/usr/bin/env python3
import os
import json
import re
import statistics
from pathlib import Path
from collections import defaultdict


def parse_output_log(log_path):
    data = {
        'train_acc_list': [],
        'model_val_acc_list': [],
        'model_pgd_acc_list': [],
        'model_fgsm_acc_list': [],
        'wa_val_acc_list': [],
        'wa_pgd_acc_list': [],
        'wa_fgsm_acc_list': [],
        'total_training_time': 0,
        'max_gpu_memory': 0,
        'clean_acc': 0,
        'pgd_10_acc': 0,
        'pgd_20_acc': 0,
        'pgd_50_acc': 0,
        'autoattack_acc': 0,
        'cr_attack_acc': 0
    }
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        train_match = re.search(r'train_acc_list:\s*\n\s*(\[[\d\.,\s\]]+)', content)
        model_val_match = re.search(r'model_val_acc_list:\s*\n\s*(\[[\d\.,\s\]]+)', content)
        model_pgd_match = re.search(r'model_pgd_acc_list:\s*\n\s*(\[[\d\.,\s\]]+)', content)
        model_fgsm_match = re.search(r'model_fgsm_acc_list:\s*\n\s*(\[[\d\.,\s\]]+)', content)
        wa_val_match = re.search(r'wa_val_acc_list:\s*\n\s*(\[[\d\.,\s\]]+)', content)
        wa_pgd_match = re.search(r'wa_pgd_acc_list:\s*\n\s*(\[[\d\.,\s\]]+)', content)
        wa_fgsm_match = re.search(r'wa_fgsm_acc_list:\s*\n\s*(\[[\d\.,\s\]]+)', content)
        time_match = re.search(r'total_training_time:\s*\n\s*([\d.]+)', content)
        memory_match = re.search(r'max_gpu_memory:\s*([\d.]+)\s*GB', content)
        
        clean_match = re.search(r'clean_acc:\s*([\d.]+)', content)
        pgd_10_match = re.search(r'pgd_10_acc:\s*([\d.]+)', content)
        pgd_20_match = re.search(r'pgd_20_acc:\s*([\d.]+)', content)
        pgd_50_match = re.search(r'pgd_50_acc:\s*([\d.]+)', content)
        autoattack_match = re.search(r'autoattack_acc:\s*([\d.]+)', content)
        cr_attack_match = re.search(r'cr_attack_acc:\s*([\d.]+)', content)
        
        if train_match:
            data['train_acc_list'] = json.loads(train_match.group(1))
        if model_val_match:
            data['model_val_acc_list'] = json.loads(model_val_match.group(1))
        if model_pgd_match:
            data['model_pgd_acc_list'] = json.loads(model_pgd_match.group(1))
        if model_fgsm_match:
            data['model_fgsm_acc_list'] = json.loads(model_fgsm_match.group(1))
        if wa_val_match:
            data['wa_val_acc_list'] = json.loads(wa_val_match.group(1))
        if wa_pgd_match:
            data['wa_pgd_acc_list'] = json.loads(wa_pgd_match.group(1))
        if wa_fgsm_match:
            data['wa_fgsm_acc_list'] = json.loads(wa_fgsm_match.group(1))
        if time_match:
            data['total_training_time'] = float(time_match.group(1))
        if memory_match:
            data['max_gpu_memory'] = float(memory_match.group(1))
        if clean_match:
            data['clean_acc'] = float(clean_match.group(1)) * 100
        if pgd_10_match:
            data['pgd_10_acc'] = float(pgd_10_match.group(1)) * 100
        if pgd_20_match:
            data['pgd_20_acc'] = float(pgd_20_match.group(1)) * 100
        if pgd_50_match:
            data['pgd_50_acc'] = float(pgd_50_match.group(1)) * 100
        if autoattack_match:
            data['autoattack_acc'] = float(autoattack_match.group(1)) * 100
        if cr_attack_match:
            data['cr_attack_acc'] = float(cr_attack_match.group(1)) * 100
            
    except Exception as e:
        print(f"Error parsing {log_path}: {e}")
    
    return data


def calculate_mean_std(values):
    if not values or len(values) == 0:
        return {'mean': 0, 'std': 0}
    
    valid_values = [v for v in values if v is not None]
    if not valid_values:
        return {'mean': 0, 'std': 0}
    
    mean_val = statistics.mean(valid_values)
    std_val = statistics.stdev(valid_values) if len(valid_values) > 1 else 0
    
    return {'mean': mean_val, 'std': std_val}


def find_all_seeds(base_path):
    seed_dirs = defaultdict(list)
    for method_dir in os.listdir(base_path):
        method_path = os.path.join(base_path, method_dir)
        if os.path.isdir(method_path):
            for run_dir in os.listdir(method_path):
                run_path = os.path.join(method_path, run_dir)
                if os.path.isdir(run_path):
                    seed_match = re.search(r'seed-(\d+)', run_dir)
                    if seed_match:
                        seed_num = int(seed_match.group(1))
                        seed_dirs[method_dir].append({
                            'seed': seed_num,
                            'path': run_path
                        })
    
    for method in seed_dirs:
        seed_dirs[method].sort(key=lambda x: x['seed'])
    
    return seed_dirs


def main():
    log_base = Path('log')
    datasets = ['cifar10', 'cifar100', 'tiny_imagenet']
    
    all_data = {}
    
    for dataset in datasets:
        dataset_path = log_base / dataset
        if not dataset_path.exists():
            continue
        
        all_data[dataset] = {}
        
        seed_dirs = find_all_seeds(dataset_path)
        
        for method, runs in seed_dirs.items():
            method_data = {
                'method': method,
                'dataset': dataset,
                'seeds': {}
            }
            
            all_seed_data = []
            
            for run_info in runs:
                seed_num = run_info['seed']
                run_path = Path(run_info['path'])
                
                output_log = run_path / 'output.log'
                
                if output_log.exists():
                    seed_data = parse_output_log(output_log)
                    seed_data['seed'] = seed_num
                    all_seed_data.append(seed_data)
                    method_data['seeds'][seed_num] = seed_data
            
            if all_seed_data:
                clean_accs = [d['clean_acc'] for d in all_seed_data]
                pgd_10_accs = [d['pgd_10_acc'] for d in all_seed_data]
                pgd_20_accs = [d['pgd_20_acc'] for d in all_seed_data]
                pgd_50_accs = [d['pgd_50_acc'] for d in all_seed_data]
                autoattack_accs = [d['autoattack_acc'] for d in all_seed_data]
                cr_attack_accs = [d['cr_attack_acc'] for d in all_seed_data]
                training_times = [d['total_training_time'] for d in all_seed_data]
                gpu_memories = [d['max_gpu_memory'] for d in all_seed_data]
                
                method_data['clean_acc'] = calculate_mean_std(clean_accs)
                method_data['pgd_10_acc'] = calculate_mean_std(pgd_10_accs)
                method_data['pgd_20_acc'] = calculate_mean_std(pgd_20_accs)
                method_data['pgd_50_acc'] = calculate_mean_std(pgd_50_accs)
                method_data['autoattack_acc'] = calculate_mean_std(autoattack_accs)
                method_data['cr_attack_acc'] = calculate_mean_std(cr_attack_accs)
                method_data['total_training_time'] = calculate_mean_std(training_times)
                method_data['max_gpu_memory'] = calculate_mean_std(gpu_memories)
            
            all_data[dataset][method] = method_data
    
    output_file = Path('benchmark_data.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2)
    
    print(f"Data parsed and saved to {output_file}")


if __name__ == '__main__':
    main()
