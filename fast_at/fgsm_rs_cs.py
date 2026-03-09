import copy
import os
import time

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from torchattacks import PGD, FGSM

from .at_base import ATBase


class FGSM_RS_CS(ATBase):

    def __init__(self, model, config_path=None, config=None, log_dir='./log/', name='fgsm_rs_cs', device=None, seed=0):
        super(FGSM_RS_CS, self).__init__(model, config_path=config_path, config=config, 
                                          log_dir=log_dir, name=name, device=device, seed=seed)
        self.factor = self.config.get('factor', 0.6)
        self.lamda = self.config.get('lamda', 7)
        self.gamma = self.config.get('gamma', 0.03)
        self.ratio = self.config.get('ratio', 1)
        self.stride = self.config.get('stride', 1)
        self.w2 = self.config.get('w2', 1.0)
        self.initialize = self.config.get('initialize', False)
        self.reg_single = self.config.get('reg_single', False)
        self.reg_multi = self.config.get('reg_multi', False)
        self.decay_arr = self.config.get('decay_arr', [0, 100, 105, 110])

    def _label_smoothing(self, label, factor, num_classes=10):
        one_hot = torch.eye(num_classes, device=label.device)[label]
        result = one_hot * factor + (one_hot - 1.) * ((factor - 1.) / float(num_classes - 1))
        return result

    def train(self, opt, scheduler, train_loader, val_loader, test_loader=None):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
            
        total_epoch = self.config.get('total_epoch', 100)
        label_smoothing = self.config.get('label_smoothing', 0.4)
        weight_average = self.config.get('weight_average', True)
        tau = self.config.get('tau', 0.9995)
        num_classes = self.config.get('num_classes', 10)
        
        if label_smoothing is not None:
            criterion = CrossEntropyLoss(label_smoothing=label_smoothing)
        else:
            criterion = CrossEntropyLoss()

        loss_fn = MSELoss()

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
        self.logger.log('factor: {}'.format(self.factor))
        self.logger.log('lamda: {}'.format(self.lamda))
        self.logger.log('gamma: {}'.format(self.gamma))
        self.logger.log('ratio: {}'.format(self.ratio))
        self.logger.log('stride: {}'.format(self.stride))
        self.logger.log('w2: {}'.format(self.w2))
        self.logger.log('initialize: {}'.format(self.initialize))
        self.logger.log('reg_single: {}'.format(self.reg_single))
        self.logger.log('reg_multi: {}'.format(self.reg_multi))
        self.logger.new_line()
        self.logger.new_line()

        best_pgd_acc, best_val_acc, total_val_time = 0.0, 0.0, 0.0
        train_acc_list = []
        model_val_acc_list, model_pgd_acc_list, model_fgsm_acc_list = [], [], []
        wa_val_acc_list, wa_pgd_acc_list, wa_fgsm_acc_list = [], [], []

        pre_loss = 0
        beta_loss = 0

        for epoch in range(total_epoch):
            self.logger.log('============ Epoch {} ============'.format(epoch))
            self.model.train()
            train_loss, train_acc, train_n = 0, 0, 0
            train_normal_loss = 0
            start_time = time.time()

            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                rand_perturb = torch.FloatTensor(images.shape).uniform_(-self.eps, self.eps).to(self.device)
                images = images + rand_perturb
                images = torch.clamp(images, 0, 1)

                label_smooth = self._label_smoothing(labels, self.factor, num_classes)
                images = images.detach().requires_grad_(True)

                ori_output = self.model(images)
                ori_loss = criterion(ori_output, labels)

                grad = torch.autograd.grad(ori_loss, [images])[0]
                perturbation = self.eps * torch.sign(grad)
                data_adv = images + perturbation
                data_adv = torch.clamp(data_adv, 0, 1)

                output = self.model(data_adv)

                if self.initialize:
                    loss_normal = ori_loss
                else:
                    benign_output = self.model(images)
                    loss_normal = criterion(benign_output, label_smooth)

                loss_reg1 = 0
                loss_reg2 = 0
                if self.reg_single:
                    loss_reg1 = loss_fn(output.float(), ori_output.float())
                if self.reg_multi:
                    loss_reg2 = torch.mean(torch.sign(output.float() - ori_output.float()) * 
                                          torch.sign(ori_output.float() - benign_output.float()))

                loss = criterion(output, label_smooth) + self.lamda * (loss_reg1 + loss_reg2)

                gamma_flag = False
                for j, _ in enumerate(self.decay_arr):
                    if j < len(self.decay_arr) - 1:
                        start_decay = self.decay_arr[j]
                        end_decay = self.decay_arr[j+1]
                        if epoch > start_decay + self.stride and epoch < end_decay - self.stride:
                            gamma_flag = True
                            break

                if gamma_flag:
                    gamma_max = torch.tensor(self.gamma).to(self.device)
                    gamma_min = gamma_max / self.ratio
                    if torch.abs(loss_normal - pre_loss) > min(max(gamma_min, beta_loss), gamma_max):
                        if loss_normal - pre_loss > 0:
                            loss = loss + self.w2 * loss_normal
                        else:
                            loss = loss - self.w2 * loss_normal

                opt.zero_grad()
                loss.backward()
                opt.step()

                self._record_gpu_memory()

                if weight_average:
                    for key, value in self.model.state_dict().items():
                        exp_avg[key] = (1 - tau) * value + tau * exp_avg[key]

                train_loss += loss.item() * labels.size(0)
                train_normal_loss += loss_normal.item() * labels.size(0)
                train_acc += (output.max(1)[1] == labels).sum().item()
                train_n += labels.size(0)

            scheduler.step()

            pre_loss2 = torch.tensor(train_normal_loss / train_n).to(self.device)
            beta_loss = pre_loss2 - pre_loss
            pre_loss = pre_loss2

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
            self.logger.log('Training normal loss: {:.4f}'.format(train_normal_loss / train_n))
            self.logger.log('Training accuracy: {:.4f}'.format(train_acc / train_n))
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
