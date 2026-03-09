import os
import torch
from torchattacks import PGD
from autoattack import AutoAttack
from attacks import CRAttack


class AttackTester:
    def __init__(self, model, device, eps=8.0 / 255, log_dir=None):
        self.model = model
        self.device = device
        self.eps = eps
        self.log_dir = log_dir

    def test_pgd(self, test_loader, steps_list=None):
        if steps_list is None:
            steps_list = [10, 20, 50]
        results = {}
        self.model.eval()
        
        for steps in steps_list:
            attacker = PGD(self.model, eps=self.eps, alpha=2.0 / 255, steps=steps)
            clean_correct, adv_correct, total = 0, 0, 0
            
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    clean_correct += (predicted == labels).sum().item()
                
                adv_images = attacker(images, labels)
                
                with torch.no_grad():
                    outputs = self.model(adv_images)
                    _, predicted = torch.max(outputs.data, 1)
                    adv_correct += (predicted == labels).sum().item()
                
                total += labels.size(0)
            
            clean_acc = clean_correct / total
            pgd_acc = adv_correct / total
            results[f'clean_acc'] = clean_acc
            results[f'pgd_{steps}_acc'] = pgd_acc
        
        return results

    def test_autoattack(self, test_loader):
        self.model.eval()
        log_path = os.path.join(self.log_dir, 'autoattack.log') if self.log_dir else None
        attacker = AutoAttack(self.model, eps=self.eps, device=self.device, log_path=log_path)
        
        X, y = [], []
        for images, labels in test_loader:
            X.append(images.to(self.device))
            y.append(labels.to(self.device))
        
        X = torch.cat(X, dim=0)
        y = torch.cat(y, dim=0)
        
        attacker.run_standard_evaluation(X, y, bs=200)
        
        # Read accuracy from log file
        if log_path and os.path.exists(log_path):
            with open(log_path, 'r') as f:
                for line in f:
                    if 'robust accuracy:' in line:
                        acc_str = line.split('robust accuracy:')[1].strip()
                        acc = float(acc_str.rstrip('%')) / 100.0
                        return {'autoattack_acc': acc}
        
        return {'autoattack_acc': -1.0}

    def test_cr_attack(self, test_loader):
        self.model.eval()
        log_path = os.path.join(self.log_dir, 'cr_attack.log') if self.log_dir else None
        attacker = CRAttack(self.model, eps=self.eps, log_path=log_path)
        
        X, y = [], []
        for images, labels in test_loader:
            X.append(images.to(self.device))
            y.append(labels.to(self.device))
        
        X = torch.cat(X, dim=0)
        y = torch.cat(y, dim=0)
        
        attacker.run_standard_evaluation(X, y, bs=200)
        
        # Read accuracy from log file
        if log_path and os.path.exists(log_path):
            with open(log_path, 'r') as f:
                for line in f:
                    if 'Robust accuracy:' in line:
                        acc_str = line.split('Robust accuracy:')[1].strip()
                        acc = float(acc_str.rstrip('%')) / 100.0
                        return {'cr_attack_acc': acc}
        
        return {'cr_attack_acc': -1.0}

    def test_all(self, test_loader):
        results = {}
        
        pgd_results = self.test_pgd(test_loader)
        results.update(pgd_results)
        
        aa_results = self.test_autoattack(test_loader)
        results.update(aa_results)
        
        cr_results = self.test_cr_attack(test_loader)
        results.update(cr_results)
        
        return results
