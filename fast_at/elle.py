import copy
import os
import time

import torch
import numpy as np
from torch.nn import CrossEntropyLoss, MSELoss
from torchattacks import PGD, FGSM

from .at_base import ATBase


class ELLE(ATBase):

    def __init__(self, model, config_path=None, config=None, log_dir='./log/', name='elle', device=None, seed=0):
        super(ELLE, self).__init__(model, config_path=config_path, config=config, 
                                     log_dir=log_dir, name=name, device=device, seed=seed)
        self.lin_reg = self.config.get('lin_reg', 1.0)
        self.n_triplets = self.config.get('n_triplets', 1)
        self.lambda_schedule = self.config.get('lambda_schedule', 'constant')
        self.decay_rate = self.config.get('decay_rate', 0.99)
        self.sensitivity = self.config.get('sensitivity', 2.0)
        self.lambda_aux = self.config.get('lambda_aux', 10.0)
        self.fgsm_init = self.config.get('fgsm_init', 'random')
        self.fgsm_alpha = self.config.get('fgsm_alpha', 1.25)
        self.attack_iter = self.config.get('attack_iter', 1)
        self.clamp = self.config.get('clamp', False)
        self.input_noise_rate = self.config.get('input_noise_rate', 0)
        
        if self.lambda_schedule == 'onoff':
            self.elle_values = []
            self.current_lin_reg = 0
        else:
            self.current_lin_reg = self.lin_reg

    def _get_triplets(self, x, y, n_triplets):
        bs = x.shape[0]
        x_2 = x.repeat([2, 1, 1, 1]) 
        x_2 = x_2 + self.eps * (2 * torch.rand(x_2.shape, device=x.device) - 1)
        alpha = torch.rand([x.shape[0], 1, 1, 1], device=x.device)
        x_middle = (1 - alpha) * x_2[:bs] + alpha * x_2[bs:]
        alpha = alpha.squeeze()
        x_adv = torch.cat((x_2, x_middle), dim=0)
        
        for i in range(n_triplets - 1):
            x_2 = x.repeat([2, 1, 1, 1]) 
            x_2 = x_2 + self.eps * (2 * torch.rand(x_2.shape, device=x.device) - 1)
            alphai = torch.rand([x.shape[0], 1, 1, 1], device=x.device)
            x_middle = (1 - alphai) * x_2[:bs] + alphai * x_2[bs:]
            alpha = torch.cat((alpha, alphai.squeeze()), dim=0)
            x_adv = torch.cat((x_adv, x_2, x_middle), dim=0)
        
        return x_adv, alpha

    def _fgsm_attack(self, X, y, epsilon, alpha, fgsm_init, attack_iter, clamp):
        delta = torch.zeros_like(X)
        if fgsm_init == 'random':
            delta.uniform_(-epsilon, epsilon)
            delta = torch.clamp(delta, min=-X, max=1 - X)
        
        for i in range(attack_iter):
            delta.requires_grad = True
            output = self.model(X + delta)
            loss = CrossEntropyLoss()(output, y)
            loss.backward()
            
            grad = delta.grad.detach()
            delta = delta + alpha * torch.sign(grad)
            
            if clamp:
                delta = torch.clamp(delta, min=-epsilon, max=epsilon)
                delta = torch.clamp(delta, min=-X, max=1 - X)
            
            delta = delta.detach()
        
        return delta.detach()

    def _compute_lin_err(self, x, y, n_triplets):
        bs = x.shape[0]
        x_ab = x.repeat([2, 1, 1, 1]) 
        x_ab = x_ab + self.eps * (2 * torch.rand(x_ab.shape, device=x.device) - 1)
        alpha = torch.rand([bs, 1, 1, 1], device=x.device)
        x_c = (1 - alpha) * x_ab[:bs] + alpha * x_ab[bs:]
        alpha = alpha.squeeze()

        criterion = CrossEntropyLoss(reduction='none')
        losses = criterion(self.model(torch.cat((x_ab, x_c), dim=0)), y.repeat([3]))

        mse = MSELoss()
        lin_err = mse(losses[2 * bs:], (1 - alpha) * losses[:bs] + alpha * losses[bs:2 * bs])
        
        return lin_err

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
        self.logger.log('lin_reg: {}'.format(self.lin_reg))
        self.logger.log('n_triplets: {}'.format(self.n_triplets))
        self.logger.log('lambda_schedule: {}'.format(self.lambda_schedule))
        self.logger.log('fgsm_init: {}'.format(self.fgsm_init))
        self.logger.log('fgsm_alpha: {}'.format(self.fgsm_alpha))
        self.logger.log('attack_iter: {}'.format(self.attack_iter))
        self.logger.log('clamp: {}'.format(self.clamp))
        self.logger.log('input_noise_rate: {}'.format(self.input_noise_rate))
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
            start_time = time.time()
            
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                delta = self._fgsm_attack(
                    images, labels, self.eps, self.fgsm_alpha * self.eps, 
                    self.fgsm_init, self.attack_iter, self.clamp
                )
                
                if self.input_noise_rate > 0:
                    noise = self.input_noise_rate * self.eps * (2 * torch.rand_like(images) - 1)
                    adv_images = torch.clamp(images + delta + noise, min=0, max=1).detach()
                else:
                    adv_images = torch.clamp(images + delta, min=0, max=1).detach()

                opt.zero_grad()
                output = self.model(adv_images)
                loss = criterion(output, labels)
                
                lin_err = self._compute_lin_err(images, labels, self.n_triplets)
                loss += self.current_lin_reg * lin_err
                
                loss.backward()
                opt.step()

                self._record_gpu_memory()

                if weight_average:
                    for key, value in self.model.state_dict().items():
                        exp_avg[key] = (1 - tau) * value + tau * exp_avg[key]

                train_loss += loss.item() * labels.size(0)
                train_acc += (output.max(1)[1] == labels).sum().item()
                train_n += labels.size(0)
                
                if self.lambda_schedule == 'onoff':
                    self.elle_values.append(lin_err.item())
                    if (len(self.elle_values) > 2) and (lin_err.item() > np.mean(self.elle_values) + self.sensitivity * np.std(self.elle_values)):
                        self.current_lin_reg = self.lambda_aux
                    else:
                        self.current_lin_reg *= self.decay_rate

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
            self.logger.log('Current lin_reg: {:.4f}'.format(self.current_lin_reg))
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
