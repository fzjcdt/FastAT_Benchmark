import argparse
import os
import yaml
import numpy as np
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR, OneCycleLR
from torch.utils.data import DataLoader
from torchvision import datasets

from models import ResNet18, PreActResNet18, FeatureResNet18, FeaturePreActResNet18
from utils.dataset import train_transform, train_transform_tiny_imagenet, test_transform, TinyImageNet200, CIFAR10Idx, CIFAR100Idx, TinyImageNet200Idx
from fast_at import (
    FGSMAT, FGSMRS, NFGSM, FGSM_PGI, FGSM_PCO, FGSM_MEP_CS, FGSM_RS_CS, ZeroGrad, AAER, ELLE, GradAlign, PGD_AT,
    Free_AT, Nu_AT, GAT, SSAT, FGSM_UAP, FGSM_CUAP, FGSM_FUAP, LIET
)


METHOD_MAP = {
    'fgsm_at': FGSMAT,
    'fgsm_rs': FGSMRS,
    'n_fgsm': NFGSM,
    'fgsm_pgi': FGSM_PGI,
    'fgsm_pco': FGSM_PCO,
    'fgsm_mep_cs': FGSM_MEP_CS,
    'fgsm_rs_cs': FGSM_RS_CS,
    'zero_grad': ZeroGrad,
    'rs_aaer': AAER,
    'n_aaer': AAER,
    'elle': ELLE,
    'grad_align': GradAlign,
    'free_at': Free_AT,
    'nu_at': Nu_AT,
    'gat': GAT,
    'ssat': SSAT,
    'fgsm_uap': FGSM_UAP,
    'fgsm_cuap': FGSM_CUAP,
    'fgsm_fuap': FGSM_FUAP,
    'liet': LIET,
    'pgd_at': PGD_AT,
    'pgd_at_wa': PGD_AT,
}


def convert_config_values(config):
    def convert_value(value):
        if isinstance(value, dict):
            return {k: convert_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [convert_value(v) for v in value]
        elif isinstance(value, str):
            import re
            if re.search(r'[0-9]\s*[+\-*/]\s*[0-9]|[0-9]+[eE][+\-]?[0-9]+', value):
                try:
                    return eval(value)
                except:
                    return value
            else:
                return value
        else:
            return value
    return convert_value(config)


def load_config(common_config_path, method_config_path):
    with open(common_config_path, 'r') as f:
        common_config = yaml.safe_load(f)
    
    with open(method_config_path, 'r') as f:
        method_config = yaml.safe_load(f)
    
    config = {**common_config, **method_config}
    config = convert_config_values(config)
    return config


def get_data_loaders(config, method_name, seed=0):
    dataset_name = config['dataset']
    data_dir = config['data_dir']
    batch_size = config['batch_size']
    num_workers = config['num_workers']
    train_size = config.get('train_size', None)
    val_size = config.get('val_size', None)
    
    # Use index datasets for FGSM_PGI, FGSM_PCO, and FGSM_MEP_CS
    use_idx_dataset = method_name in ['fgsm_pgi', 'fgsm_pco', 'fgsm_mep_cs']
    
    if dataset_name == 'cifar10':
        if use_idx_dataset:
            train_set = CIFAR10Idx(data_dir, train=True, download=True, transform=train_transform)
            test_set = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_transform)
        else:
            train_set = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_transform)
            test_set = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_transform)
        num_classes = 10
    elif dataset_name == 'cifar100':
        if use_idx_dataset:
            train_set = CIFAR100Idx(data_dir, train=True, download=True, transform=train_transform)
            test_set = datasets.CIFAR100(data_dir, train=False, download=True, transform=test_transform)
        else:
            train_set = datasets.CIFAR100(data_dir, train=True, download=True, transform=train_transform)
            test_set = datasets.CIFAR100(data_dir, train=False, download=True, transform=test_transform)
        num_classes = 100
    elif dataset_name == 'tiny-imagenet':
        if use_idx_dataset:
            train_set = TinyImageNet200Idx(data_dir, train=True, download=True, transform=train_transform_tiny_imagenet)
            test_set = TinyImageNet200(data_dir, train=False, download=True, transform=test_transform)
        else:
            train_set = TinyImageNet200(data_dir, train=True, download=True, transform=train_transform_tiny_imagenet)
            test_set = TinyImageNet200(data_dir, train=False, download=True, transform=test_transform)
        num_classes = 200
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')
    
    if train_size is not None and val_size is not None:
        if dataset_name == 'tiny-imagenet':
            np.random.seed(seed)
            total_samples = len(train_set)
            train_set_idx = np.random.choice(np.arange(0, total_samples), train_size, replace=False)
            train_subset = torch.utils.data.Subset(train_set, train_set_idx)
            
            val_set = TinyImageNet200(data_dir, train=True, download=True, transform=test_transform)
            val_set_idx = np.setdiff1d(np.arange(0, total_samples), train_set_idx)
            val_subset = torch.utils.data.Subset(val_set, val_set_idx)
        else:
            indices = np.arange(len(train_set))
            train_indices = indices[:train_size]
            val_indices = indices[train_size:train_size + val_size]
            train_subset = torch.utils.data.Subset(train_set, train_indices)
            
            if dataset_name == 'cifar10':
                val_set = datasets.CIFAR10(data_dir, train=True, download=True, transform=test_transform)
            elif dataset_name == 'cifar100':
                val_set = datasets.CIFAR100(data_dir, train=True, download=True, transform=test_transform)
            elif dataset_name == 'tiny-imagenet':
                val_set = TinyImageNet200(data_dir, train=True, download=True, transform=test_transform)
            else:
                raise ValueError(f'Unknown dataset: {dataset_name}')
            val_subset = torch.utils.data.Subset(val_set, val_indices)
    else:
        train_subset = train_set
        val_subset = test_set
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader, num_classes


def get_model(config, num_classes, device):
    model_name = config['model']
    if model_name == 'ResNet18':
        model = ResNet18(num_classes=num_classes).to(device)
    elif model_name == 'FeatureResNet18':
        model = FeatureResNet18(num_classes=num_classes).to(device)
    elif model_name == 'PreActResNet18':
        model = PreActResNet18(num_classes=num_classes).to(device)
    elif model_name == 'FeaturePreActResNet18':
        model = FeaturePreActResNet18(num_classes=num_classes).to(device)
    else:
        raise ValueError(f'Unknown model: {model_name}')
    return model


def get_optimizer_and_scheduler(model, config, train_loader):
    lr = config['lr']
    momentum = config['momentum']
    weight_decay = config['weight_decay']
    scheduler_name = config['scheduler']
    total_epoch = config['total_epoch']
    milestones = config.get('scheduler_milestones', [90, 95])
    gamma = config.get('scheduler_gamma', 0.1)
    
    opt = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    
    if scheduler_name == 'multi_step':
        scheduler = MultiStepLR(opt, milestones=milestones, gamma=gamma)
    elif scheduler_name == 'one_cycle':
        scheduler = OneCycleLR(opt, max_lr=lr, total_steps=total_epoch)
    else:
        raise ValueError(f'Unknown scheduler: {scheduler_name}')
    
    return opt, scheduler


def run_experiment(config, method_name, seed, device):
    print(f'Running experiment: method={method_name}, seed={seed}, dataset={config["dataset"]}')
    
    train_loader, val_loader, test_loader, num_classes = get_data_loaders(config, method_name, seed)
    model = get_model(config, num_classes, device)
    opt, scheduler = get_optimizer_and_scheduler(model, config, train_loader)
    
    method_class = METHOD_MAP[method_name]
    trainer = method_class(
        model,
        config=config,
        log_dir=config['log_dir'],
        name=method_name,
        device=device,
        seed=seed
    )
    
    trainer.train(opt, scheduler, train_loader, val_loader, test_loader=test_loader)


def main():
    parser = argparse.ArgumentParser(description='Fast Adversarial Training Benchmark')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'tiny-imagenet'],
                        help='Dataset to use')
    parser.add_argument('--methods', type=str, nargs='+', default=['fgsm_at'],
                        help='Methods to run (default: fgsm_at)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0],
                        help='Random seeds to use (default: [0])')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (default: auto)')
    args = parser.parse_args()
    
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f'Using device: {device}')
    
    for seed in args.seeds:
        for method_name in args.methods:
            common_config_path = f'configs/{args.dataset}/common.yaml'
            method_config_path = f'configs/{args.dataset}/{method_name}.yaml'
            
            if not os.path.exists(common_config_path):
                print(f'Common config not found: {common_config_path}')
                continue
            if not os.path.exists(method_config_path):
                print(f'Method config not found: {method_config_path}')
                continue
            
            config = load_config(common_config_path, method_config_path)
            print(config)
            run_experiment(config, method_name, seed, device)


if __name__ == '__main__':
    main()
