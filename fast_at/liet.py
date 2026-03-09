import copy
import os
import time

import torch
import torch.nn.functional as F
from torchattacks import PGD, FGSM

from .at_base import ATBase


class LIET(ATBase):

    def __init__(self, model, config_path=None, config=None, log_dir='./log/', name='liet', device=None, seed=0):
        super(LIET, self).__init__(model, config_path=config_path, config=config, 
                                    log_dir=log_dir, name=name, device=device, seed=seed)

    def train(self, opt, scheduler, train_loader, val_loader, test_loader=None):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
            
        total_epoch = self.config.get('total_epoch', 100)
        label_smoothing = self.config.get('label_smoothing', 0.6)
        weight_average = self.config.get('weight_average', True)
        tau = self.config.get('tau', 0.9995)
        lambda_value = self.config.get('lambda', 100)
        min_mask_ratio = self.config.get('min_mask_ratio', 0.1)
        max_mask_ratio = self.config.get('max_mask_ratio', 0.5)
        li_update = self.config.get('li_update', 20)
        class_num = self.config.get('class_num', 10)
        image_size = self.config.get('image_size', 32)
        
        criterion = self._get_custom_loss(label_smoothing, class_num)

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
        self.logger.log('Lambda: {}'.format(lambda_value))
        self.logger.log('Mask ratio: [{}, {}]'.format(min_mask_ratio, max_mask_ratio))
        self.logger.log('LI update: {}'.format(li_update))
        self.logger.new_line()
        self.logger.new_line()

        best_pgd_acc, best_val_acc, total_val_time = 0.0, 0.0, 0.0
        train_acc_list = []
        model_val_acc_list, model_pgd_acc_list, model_fgsm_acc_list = [], [], []
        wa_val_acc_list, wa_pgd_acc_list, wa_fgsm_acc_list = [], [], []
        cur_lambda, cur_update = 0, lambda_value / total_epoch

        for epoch in range(total_epoch):
            self.logger.log('============ Epoch {} ============'.format(epoch))
            self.model.train()
            train_loss, train_acc, train_n = 0, 0, 0
            start_time = time.time()
            
            cur_lambda += cur_update
            LI = None
            
            for idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                if idx % li_update == 0:
                    self.model.eval()
                    constant_input = torch.full((class_num, 3, image_size, image_size), 0.5).to(self.device)
                    constant_input.requires_grad_(True)
                    sequential_labels = torch.arange(class_num).to(self.device)

                    output = self.model(constant_input)
                    loss = criterion(output, sequential_labels)
                    loss.backward()
                    LI = self.eps * torch.sign(constant_input.grad).detach()

                    self.model.train()

                delta = LI[labels].clone()
                bonulli_idx = torch.rand_like(images)
                if self.eps > 8 / 255:
                    uniform_noise = torch.zeros_like(images).uniform_(-2 * self.eps, 2 * self.eps)
                else:
                    uniform_noise = torch.zeros_like(images).uniform_(-self.eps, self.eps)
                mask_ratio = torch.rand(1).item() * (max_mask_ratio - min_mask_ratio) + min_mask_ratio
                delta[bonulli_idx < mask_ratio] = uniform_noise[bonulli_idx < mask_ratio]

                if torch.rand(1) < 0.5:
                    adv_images = torch.clamp(images.clone() - delta, 0, 1).detach()
                else:
                    adv_images = torch.clamp(images.clone() + delta, 0, 1).detach()

                adv_images = adv_images.detach().requires_grad_(True)

                ori_output = self.model(adv_images)
                ori_loss = criterion(ori_output, labels)
                ori_loss.backward(retain_graph=True)
                delta = self.eps * adv_images.grad.sign().detach()

                adv_images = adv_images + delta
                if self.eps > 8 / 255:
                    adv_images = torch.clamp(adv_images, min=0, max=1).detach()
                else:
                    delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
                    adv_images = torch.clamp(images + delta, min=0, max=1).detach()

                output = self.model(adv_images)
                train_acc += (output.argmax(1) == labels).sum().item()
                train_n += images.shape[0]

                loss1 = criterion(output, labels)
                p = F.softmax(ori_output, dim=1)
                q = F.softmax(output, dim=1)
                m = torch.clamp((p + q) / 2., 0, 1).log()
                loss2 = cur_lambda * (F.kl_div(m, p, reduction='batchmean') + F.kl_div(m, q, reduction='batchmean')) / 2

                loss = loss1 + loss2

                opt.zero_grad()
                loss.backward()
                opt.step()

                self._record_gpu_memory()

                if weight_average:
                    for key, value in self.model.state_dict().items():
                        exp_avg[key] = (1 - tau) * value + tau * exp_avg[key]

                train_loss += loss.item() * labels.size(0)

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

    def _get_custom_loss(self, label_smoothing, class_num):
        import torch.nn.functional as F
        from torch import nn

        class CustomCrossEntropyLoss(nn.Module):
            def __init__(self, label_smoothing=0.0, reduction='mean', class_num=10):
                super(CustomCrossEntropyLoss, self).__init__()
                self.label_smoothing = label_smoothing
                self.reduction = reduction
                self.class_num = class_num

            def forward(self, input, target):
                smooth_labels = torch.zeros_like(input).scatter_(1, target.unsqueeze(1), 1.0)

                if self.label_smoothing > 0:
                    smooth_labels = smooth_labels * (1 - self.label_smoothing)
                    rand_labels = torch.rand_like(smooth_labels) * self.label_smoothing
                    mask = smooth_labels.bool()
                    rand_labels = rand_labels.masked_fill_(mask, 0)
                    rand_labels_sum = rand_labels.sum(dim=1, keepdim=True)
                    rand_labels = rand_labels / rand_labels_sum * self.label_smoothing
                    smooth_labels += rand_labels

                log_probs = F.log_softmax(input, dim=-1)
                loss = (-smooth_labels * log_probs).sum(dim=-1)
                if self.reduction == 'mean':
                    return loss.mean()
                elif self.reduction == 'sum':
                    return loss.sum()
                else:
                    return loss

        return CustomCrossEntropyLoss(label_smoothing=label_smoothing, class_num=class_num)
