import copy
import os
import time

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torchattacks import PGD, FGSM

from .at_base import ATBase


class AAER(ATBase):

    def __init__(self, model, config_path=None, config=None, log_dir='./log/', name='aaer', device=None, seed=0):
        super(AAER, self).__init__(model, config_path=config_path, config=config, 
                                   log_dir=log_dir, name=name, device=device, seed=seed)
        self.lamda1 = self.config.get('lamda1', 1.0)
        self.lamda2 = self.config.get('lamda2', 2.5)
        self.lamda3 = self.config.get('lamda3', 1.5)
        self.clamp = self.config.get('clamp', 1)
        self.fgsm_alpha = self.config.get('fgsm_alpha', 1.25)
        self.delta_init = self.config.get('delta_init', 'random')

    def _l2_square(self, x, y):
        diff = x - y
        diff = diff * diff
        diff = diff.sum(1).mean(0)
        return diff

    def _aaer_attack(self, X, y, epsilon, alpha, clamp, delta_init):
        batch_size = X.size(0)
        _, _, height, width = X.size()
        
        if delta_init == 'zero':
            delta = torch.zeros(batch_size, 3, height, width).to(self.device)
        elif delta_init == 'random':
            delta = torch.zeros(batch_size, 3, height, width).to(self.device)
            if clamp:
                for j in range(3):
                    delta[:, j, :, :].uniform_(-epsilon, epsilon)
            else:
                for j in range(3):
                    delta[:, j, :, :].uniform_(2 * -epsilon, 2 * epsilon)
        
        delta = torch.clamp(delta, min=-X, max=1 - X)
        delta.requires_grad = True
        
        output = self.model(X + delta)
        output_org = output.detach()
        loss = CrossEntropyLoss(reduce=False)(output, y)
        loss_before = loss.detach()
        loss = loss.mean()
        loss.backward()
        
        grad = delta.grad.detach()
        delta.data = delta + alpha * torch.sign(grad)
        if clamp:
            delta.data = torch.clamp(delta, min=-epsilon, max=epsilon)
        delta.data = torch.clamp(delta, min=-X, max=1 - X)
        delta = delta.detach()
        
        delta.requires_grad = True
        output = self.model(torch.clamp(X + delta, min=0, max=1))
        loss = CrossEntropyLoss(reduce=False)(output, y)
        loss_after = loss.detach()
        loss = loss.mean()
        
        abnormal_example = loss_before > loss_after
        normal_example = loss_before <= loss_after
        abnormal_count = torch.count_nonzero(abnormal_example)
        normal_count = torch.count_nonzero(normal_example)
        
        abnormal_ce = None
        abnormal_variation = None
        normal_variation = None
        
        if abnormal_count != 0:
            abnormal_variation = self._l2_square(output_org[abnormal_example], output[abnormal_example])
            abnormal_ce = abnormal_example * (loss_before - loss_after)
            abnormal_ce = abnormal_ce.sum() / abnormal_count
        
        if normal_count != 0:
            normal_variation = self._l2_square(output_org[normal_example], output[normal_example])
        
        return delta, loss, abnormal_count, normal_count, abnormal_ce, abnormal_variation, normal_variation

    def train(self, opt, scheduler, train_loader, val_loader, test_loader=None):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
            
        total_epoch = self.config.get('total_epoch', 100)
        label_smoothing = self.config.get('label_smoothing', 0.4)
        weight_average = self.config.get('weight_average', True)
        tau = self.config.get('tau', 0.9995)
        
        if label_smoothing is not None:
            criterion = CrossEntropyLoss(label_smoothing=label_smoothing)
        else:
            criterion = CrossEntropyLoss()

        if weight_average:
            wa_model = copy.deepcopy(self.model)
            exp_avg = self.model.state_dict()
            if tau is None:
                raise ValueError('tau should not be None when weight_average is True')

        self.logger.log('Method: {}'.format(self.name))
        self.logger.log('Seed: {}'.format(self.seed))
        self.logger.log('Scheduler: {}'.format(scheduler.__class__.__name__))
        self.logger.log('Label smoothing: {}'.format(label_smoothing))
        self.logger.log('Weight average: {}, tau: {}'.format(weight_average, tau))
        self.logger.log('AAER parameters: lamda1={}, lamda2={}, lamda3={}, clamp={}, fgsm_alpha={}, delta_init={}'.format(
            self.lamda1, self.lamda2, self.lamda3, self.clamp, self.fgsm_alpha, self.delta_init))
        self.logger.new_line()
        self.logger.new_line()

        best_pgd_acc, best_val_acc, total_val_time = 0.0, 0.0, 0.0
        train_acc_list = []
        model_val_acc_list, model_pgd_acc_list, model_fgsm_acc_list = [], [], []
        wa_val_acc_list, wa_pgd_acc_list, wa_fgsm_acc_list = [], [], []

        for epoch in range(total_epoch):
            self.logger.log('============ Epoch {} ============'.format(epoch))
            self.model.train()
            train_loss, train_acc, train_n = 0, 0, 0
            ae_num = 0
            ae_ce_loss = 0
            ae_l2_loss = 0
            start_time = time.time()
            
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                epsilon = self.eps
                alpha = self.fgsm_alpha * epsilon
                
                delta, loss, abnormal_count, normal_count, abnormal_ce, abnormal_variation, normal_variation = self._aaer_attack(
                    images, labels, epsilon, alpha, self.clamp, self.delta_init)
                
                if abnormal_count != 0 and normal_count != 0:
                    # Clip variation difference to prevent numerical instability
                    variation_diff = torch.clamp(abnormal_variation - normal_variation.item(), min=-100, max=100)
                    loss = loss + (self.lamda1 * abnormal_count / labels.size(0)) * (
                        self.lamda2 * abnormal_ce + self.lamda3 * max(variation_diff, 0))
                    ae_num += abnormal_count
                    ae_ce_loss += abnormal_ce.item() * abnormal_count
                    ae_l2_loss += variation_diff.item() * abnormal_count
                
                opt.zero_grad()
                loss.backward()
                opt.step()

                self._record_gpu_memory()

                train_loss += loss.item() * labels.size(0)
                train_acc += (self.model(images + delta).max(1)[1] == labels).sum().item()
                train_n += labels.size(0)

                if weight_average:
                    for key, value in self.model.state_dict().items():
                        exp_avg[key] = (1 - tau) * value + tau * exp_avg[key]

            scheduler.step()

            if weight_average:
                self.model.eval()
                wa_model.load_state_dict(exp_avg)
                wa_model.eval()
            else:
                self.model.eval()

            epoch_training_time = time.time() - start_time
            self.total_training_time += epoch_training_time
            self.logger.log('Training time: {:.2f}s'.format(epoch_training_time))
            self.logger.log('Training loss: {:.4f}'.format(train_loss / train_n))
            self.logger.log('Training accuracy: {:.4f}'.format(train_acc / train_n))
            self.logger.log('Abnormal examples: {}'.format(ae_num))
            if ae_num > 0:
                self.logger.log('Abnormal CE loss: {:.4f}'.format(ae_ce_loss / ae_num))
                self.logger.log('Abnormal L2 loss: {:.4f}'.format(ae_l2_loss / ae_num))
            current_gpu_memory = self._record_gpu_memory()
            self.logger.log('Current GPU memory: {:.2f} GB'.format(current_gpu_memory))
            train_acc_list.append(train_acc / train_n)

            start_time = time.time()
            
            if weight_average:
                validation_results = self._validate_with_weight_average(self.model, wa_model, val_loader)
                
                self.logger.log('Validation time: {:.2f}s'.format(time.time() - start_time))
                self.logger.log('Model - Validation accuracy: {:.4f}'.format(validation_results['model']['val_acc']))
                self.logger.log('Model - FGSM accuracy: {:.4f}'.format(validation_results['model']['fgsm_acc']))
                self.logger.log('Model - PGD accuracy: {:.4f}'.format(validation_results['model']['pgd_acc']))
                self.logger.log('WA Model - Validation accuracy: {:.4f}'.format(validation_results['wa_model']['val_acc']))
                self.logger.log('WA Model - FGSM accuracy: {:.4f}'.format(validation_results['wa_model']['fgsm_acc']))
                self.logger.log('WA Model - PGD accuracy: {:.4f}'.format(validation_results['wa_model']['pgd_acc']))
                
                model_val_acc_list.append(validation_results['model']['val_acc'])
                model_pgd_acc_list.append(validation_results['model']['pgd_acc'])
                model_fgsm_acc_list.append(validation_results['model']['fgsm_acc'])
                wa_val_acc_list.append(validation_results['wa_model']['val_acc'])
                wa_pgd_acc_list.append(validation_results['wa_model']['pgd_acc'])
                wa_fgsm_acc_list.append(validation_results['wa_model']['fgsm_acc'])
                
                current_pgd_acc = validation_results['best_pgd_acc']
                current_val_acc = validation_results['best_val_acc']
                self.logger.log('Best model: {} (PGD acc: {:.4f}, Val acc: {:.4f})'.format(
                    validation_results['best_model'], current_pgd_acc, current_val_acc))
            else:
                validation_results = self._validate_model(self.model, val_loader)
                
                self.logger.log('Validation time: {:.2f}s'.format(time.time() - start_time))
                self.logger.log('Validation accuracy: {:.4f}'.format(validation_results['val_acc']))
                self.logger.log('FGSM accuracy: {:.4f}'.format(validation_results['fgsm_acc']))
                self.logger.log('PGD accuracy: {:.4f}'.format(validation_results['pgd_acc']))
                
                model_val_acc_list.append(validation_results['val_acc'])
                model_pgd_acc_list.append(validation_results['pgd_acc'])
                model_fgsm_acc_list.append(validation_results['fgsm_acc'])
                
                current_pgd_acc = validation_results['pgd_acc']
                current_val_acc = validation_results['val_acc']
            
            total_val_time += time.time() - start_time

            if current_pgd_acc > best_pgd_acc or (
                    current_pgd_acc == best_pgd_acc and current_val_acc > best_val_acc):
                best_pgd_acc = current_pgd_acc
                best_val_acc = current_val_acc
                if weight_average:
                    if validation_results['best_model'] == 'model':
                        torch.save(self.model.state_dict(), os.path.join(self.output_dir, 'best.pth'))
                    else:
                        torch.save(wa_model.state_dict(), os.path.join(self.output_dir, 'best.pth'))
                else:
                    torch.save(self.model.state_dict(), os.path.join(self.output_dir, 'best.pth'))

            self.logger.new_line()

        self.logger.log("train_acc_list: \n" + str(train_acc_list))
        self.logger.new_line()
        self.logger.log("model_val_acc_list: \n" + str(model_val_acc_list))
        self.logger.new_line()
        self.logger.log("model_pgd_acc_list: \n" + str(model_pgd_acc_list))
        self.logger.new_line()
        self.logger.log("model_fgsm_acc_list: \n" + str(model_fgsm_acc_list))
        self.logger.new_line()
        if weight_average:
            self.logger.log("wa_val_acc_list: \n" + str(wa_val_acc_list))
            self.logger.new_line()
            self.logger.log("wa_pgd_acc_list: \n" + str(wa_pgd_acc_list))
            self.logger.new_line()
            self.logger.log("wa_fgsm_acc_list: \n" + str(wa_fgsm_acc_list))
            self.logger.new_line()
        self.logger.log("total_training_time: \n" + str(self.total_training_time))
        self.logger.new_line()
        self.logger.log("total_val_time: \n" + str(total_val_time))
        self.logger.new_line()
        self.logger.log("max_gpu_memory: {:.2f} GB".format(self.max_gpu_memory))
        self.logger.new_line()

        if weight_average:
            torch.save(wa_model.state_dict(), os.path.join(self.output_dir, 'last.pth'))
        else:
            torch.save(self.model.state_dict(), os.path.join(self.output_dir, 'last.pth'))

        if test_loader is not None:
            self._run_final_evaluation(test_loader, weight_average, wa_model if weight_average else self.model)
