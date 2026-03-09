import os
import time
import yaml
import csv
import torch
from utils import set_seed, Logger


class ATBase(object):

    def __init__(self, model, config_path=None, config=None, log_dir='./log/', name='', device=None, seed=0):
        set_seed(seed)
        self.model = model
        self.seed = seed
        
        if config_path is not None:
            self.config = self._load_config(config_path)
        elif config is not None:
            self.config = config
        else:
            self.config = {}
        
        self.eps = self.config.get('eps', 8.0 / 255)
        self.name = self.config.get('name', name)
        log_dir = self.config.get('log_dir', log_dir)
        
        output_dir = os.path.join(log_dir, self.name)
        output_dir = os.path.join(output_dir,
                                  time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + '-seed-' + str(seed))

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.output_dir = output_dir
        log_file = os.path.join(output_dir, 'output.log')
        self.logger = Logger(log_file)

        if device is None:
            self.device = next(model.parameters()).device
        else:
            self.device = device
        
        self.max_gpu_memory = 0.0
        self.total_training_time = 0.0

    def _load_config(self, config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def _record_gpu_memory(self):
        if torch.cuda.is_available():
            current_memory = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
            if current_memory > self.max_gpu_memory:
                self.max_gpu_memory = current_memory
            # self.logger.log(f'Current GPU memory: {current_memory:.2f} GB, Max: {self.max_gpu_memory:.2f} GB')
            return current_memory
        return 0.0

    def save_results_to_csv(self, results, csv_path='./results.csv'):
        file_exists = os.path.isfile(csv_path)
        
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(results)

    def _run_final_evaluation(self, test_loader, weight_average, model_to_test):
        from utils import AttackTester
        
        self.logger.log('=' * 50)
        self.logger.log('Starting final evaluation on test set...')
        self.logger.log('=' * 50)
        
        if weight_average:
            import copy
            model = copy.deepcopy(self.model)
            model.load_state_dict(torch.load(os.path.join(self.output_dir, 'best.pth')))
        else:
            model = model_to_test
        
        tester = AttackTester(model, self.device, eps=self.eps, log_dir=self.output_dir)
        
        attack_results = {}
        try:
            pgd_results = tester.test_pgd(test_loader)
            attack_results.update(pgd_results)
            self.logger.log('PGD test results:')
            for k, v in pgd_results.items():
                self.logger.log(f'  {k}: {v:.4f}')
        except Exception as e:
            self.logger.log(f'Error during PGD test: {e}')
        
        try:
            aa_results = tester.test_autoattack(test_loader)
            attack_results.update(aa_results)
            self.logger.log('AutoAttack results:')
            for k, v in aa_results.items():
                self.logger.log(f'  {k}: {v:.4f}')
        except Exception as e:
            self.logger.log(f'Error during AutoAttack test: {e}')
        
        try:
            cr_results = tester.test_cr_attack(test_loader)
            attack_results.update(cr_results)
            self.logger.log('CR-Attack results:')
            for k, v in cr_results.items():
                self.logger.log(f'  {k}: {v:.4f}')
        except Exception as e:
            self.logger.log(f'Error during CR-Attack test: {e}')
        
        csv_results = {
            'method': self.name,
            'seed': self.seed,
            'dataset': self.config.get('dataset', 'unknown'),
            'training_time': self.total_training_time,
            'max_gpu_memory_gb': self.max_gpu_memory,
            **attack_results
        }
        
        self.save_results_to_csv(csv_results, csv_path='./results.csv')

    def _validate_model(self, model, val_loader, model_name='model'):
        from torchattacks import PGD, FGSM
        
        pgd_attacker = PGD(model, eps=self.eps, alpha=2.0 / 255, steps=10)
        fgsm_attacker = FGSM(model, eps=self.eps)
        
        val_acc, fgsm_acc, pgd_acc, val_num = 0, 0, 0, 0
        
        for images, labels in val_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            output = model(images)
            val_acc += (output.max(1)[1] == labels).sum().item()
            val_num += labels.size(0)
            
            adv_images = pgd_attacker(images, labels)
            output = model(adv_images)
            pgd_acc += (output.max(1)[1] == labels).sum().item()
            
            fgsm_images = fgsm_attacker(images, labels)
            output = model(fgsm_images)
            fgsm_acc += (output.max(1)[1] == labels).sum().item()
        
        return {
            'val_acc': val_acc / val_num,
            'fgsm_acc': fgsm_acc / val_num,
            'pgd_acc': pgd_acc / val_num
        }

    def _validate_with_weight_average(self, model, wa_model, val_loader):
        model_results = self._validate_model(model, val_loader, 'model')
        wa_results = self._validate_model(wa_model, val_loader, 'wa_model')
        
        if model_results['pgd_acc'] >= wa_results['pgd_acc']:
            best_pgd_acc = model_results['pgd_acc']
            best_val_acc = model_results['val_acc']
            best_model = 'model'
        else:
            best_pgd_acc = wa_results['pgd_acc']
            best_val_acc = wa_results['val_acc']
            best_model = 'wa_model'
        
        return {
            'model': model_results,
            'wa_model': wa_results,
            'best_pgd_acc': best_pgd_acc,
            'best_val_acc': best_val_acc,
            'best_model': best_model
        }

    def train(self, opt, scheduler, train_loader, val_loader, test_loader=None):
        raise NotImplementedError
