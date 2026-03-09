import copy
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from torchattacks import PGD, FGSM

from .at_base import ATBase


class FeatureLayer(nn.Module):
    def __init__(self, uap_num=50, class_num=10):
        super(FeatureLayer, self).__init__()
        self.uap_linear = nn.Linear(512, uap_num)
        self.uap_classifier = nn.Linear(uap_num, class_num)

    def forward(self, x):
        feature = self.uap_linear(x)
        out = self.uap_classifier(F.relu(feature))

        return feature, out


class FGSM_FUAP(ATBase):

    def __init__(self, model, config_path=None, config=None, log_dir='./log/', name='fgsm_fuap', device=None, seed=0, uap_eps=None):
        super(FGSM_FUAP, self).__init__(model, config_path=config_path, config=config, log_dir=log_dir, name=name, device=device, seed=seed)
        self.momentum_decay = self.config.get('momentum_decay', 0.3)
        self.lamda = self.config.get('lamda', 10)
        self.uap_eps = uap_eps or self.config.get('uap_eps', 10/255.0)

    def train(self, opt, scheduler, train_loader, val_loader, test_loader=None):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
            
        total_epoch = self.config.get('total_epoch', 100)
        label_smoothing = self.config.get('label_smoothing', 0.4)
        weight_average = self.config.get('weight_average', True)
        tau = self.config.get('tau', 0.9995)
        uap_num = self.config.get('uap_num', 50)
        class_num = self.config.get('class_num', 10)
        image_shape = self.config.get('image_shape', (3, 32, 32))

        if label_smoothing is not None:
            criterion = CrossEntropyLoss(label_smoothing=label_smoothing)
        else:
            criterion = CrossEntropyLoss()

        loss_fn = MSELoss()
        uaps = torch.zeros((uap_num, *image_shape)).uniform_(-self.uap_eps, self.uap_eps).to(self.device)
        uaps = torch.clamp(self.uap_eps * torch.sign(uaps), -self.uap_eps, self.uap_eps)

        momentum = torch.zeros((uap_num, *image_shape)).to(self.device)
        feature_layer = FeatureLayer(uap_num=uap_num, class_num=class_num).to(self.device)
        opt_feature_layer = torch.optim.Adam(feature_layer.parameters(), lr=0.001)

        if weight_average:
            wa_model = copy.deepcopy(self.model)
            exp_avg = self.model.state_dict()
            if tau is None:
                raise ValueError('tau should not be None when weight_average is True')

        self.logger.log('scheduler: {}'.format(scheduler.__class__.__name__))
        self.logger.log('label smoothing: {}'.format(label_smoothing))
        self.logger.log('weight average: {}, tau: {}'.format(weight_average, tau))
        self.logger.log('uap num: {}'.format(uap_num))
        self.logger.new_line()
        self.logger.new_line()

        best_pgd_acc, best_val_acc, total_val_time = 0.0, 0.0, 0.0
        train_acc_list = []
        model_val_acc_list, model_pgd_acc_list, model_fgsm_acc_list = [], [], []
        wa_val_acc_list, wa_pgd_acc_list, wa_fgsm_acc_list = [], [], []

        for epoch in range(total_epoch):
            self.logger.log('============ Epoch {} ============'.format(epoch))
            self.model.train()
            train_loss, train_correct, train_n = 0, 0, 0
            start_time = time.time()

            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                delta = torch.zeros_like(images).uniform_(-self.eps, self.eps)
                adv_images = torch.clamp(images + delta, 0, 1)
                with torch.no_grad():
                    feature = self.model(adv_images, feature_layer=4)

                feature, out = feature_layer(feature)
                uap_max_idx = feature.max(dim=1)[1]
                uap_noise = uaps[uap_max_idx].clone()

                opt_feature_layer.zero_grad()
                loss_uap = criterion(out, labels)
                loss_uap.backward()
                opt_feature_layer.step()

                adv_images = adv_images + uap_noise
                delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
                adv_images = torch.clamp(images + delta, min=0, max=1).detach()
                adv_images.requires_grad_(True)

                ori_output = self.model(adv_images)
                ori_loss = criterion(ori_output, labels)
                ori_loss.backward(retain_graph=True)

                grad_x = adv_images.grad.detach()
                adv_images = adv_images + self.eps * adv_images.grad.sign()
                delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
                adv_images = torch.clamp(images + delta, min=0, max=1).detach()

                output = self.model(adv_images)
                loss = criterion(output, labels) + self.lamda * loss_fn(output.float(), ori_output.float())

                # train
                opt.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
                opt.step()

                # Record GPU memory
                self._record_gpu_memory()

                grad_norm = torch.norm(grad_x, p=1)
                cur_grad = grad_x / grad_norm
                for uap_idx in set(uap_max_idx.tolist()):
                    momentum[uap_idx] = cur_grad[uap_idx == uap_max_idx].mean(dim=0) + momentum[uap_idx] * 0.3
                    uaps[uap_idx] = torch.clamp(
                        uaps[uap_idx] + self.uap_eps * torch.sign(momentum[uap_idx]), -self.uap_eps, self.uap_eps)

                momentum = momentum.detach()
                uaps = uaps.detach()

                # weight average
                if weight_average:
                    """
                    for p_wa, p_model in zip(wa_model.parameters(), self.model.parameters()):
                        p_wa.data = p_wa.data * tau + p_model.data * (1 - tau)
                    """
                    for key, value in self.model.state_dict().items():
                        exp_avg[key] = (1 - tau) * value + tau * exp_avg[key]

                train_loss += loss.item() * labels.size(0)
                train_correct += (output.max(1)[1] == labels).sum().item()
                train_n += labels.size(0)

            scheduler.step()

            if weight_average:
                self.model.eval()
                wa_model.load_state_dict(exp_avg)
                wa_model.eval()
                # update bn
                """
                for module1, module2 in zip(wa_model.modules(), self.model.modules()):
                    if isinstance(module1, torch.nn.modules.batchnorm._BatchNorm):
                        module1.running_mean = module2.running_mean.clone()
                        module1.running_var = module2.running_var.clone()
                        module1.num_batches_tracked = module2.num_batches_tracked.clone()
                """
            else:
                self.model.eval()

            epoch_training_time = time.time() - start_time
            self.total_training_time += epoch_training_time
            self.logger.log('Training time: {:.2f}s'.format(epoch_training_time))
            self.logger.log('Training loss: {:.4f}'.format(train_loss / train_n))
            self.logger.log('Training accuracy: {:.4f}'.format(train_correct / train_n))
            current_gpu_memory = self._record_gpu_memory()
            self.logger.log('Current GPU memory: {:.2f} GB'.format(current_gpu_memory))
            train_acc_list.append(train_correct / train_n)

            start_time = time.time()
            
            if weight_average:
                validation_results = self._validate_with_weight_average(self.model, wa_model, val_loader)
                
                self.logger.log('Validation time: {:.2f}'.format(time.time() - start_time))
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
                
                self.logger.log('Validation time: {:.2f}'.format(time.time() - start_time))
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
