import copy
import os
import time

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torchattacks import PGD, FGSM

from .at_base import ATBase


class GAT(ATBase):

    def __init__(self, model, config_path=None, config=None, log_dir='./log/', name='gat', device=None, seed=0):
        super(GAT, self).__init__(model, config_path=config_path, config=config, log_dir=log_dir, name=name, device=device, seed=seed)
        self.l2_reg = self.config.get('l2_reg', 10)

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

        self.logger.log('scheduler: {}'.format(scheduler.__class__.__name__))
        self.logger.log('label smoothing: {}'.format(label_smoothing))
        self.logger.log('weight average: {}, tau: {}'.format(weight_average, tau))
        self.logger.new_line()
        self.logger.new_line()

        best_pgd_acc, best_val_acc, total_val_time = 0.0, 0.0, 0.0
        train_acc_list = []
        model_val_acc_list, model_pgd_acc_list, model_fgsm_acc_list = [], [], []
        wa_val_acc_list, wa_pgd_acc_list, wa_fgsm_acc_list = [], [], []

        # self.logger.log('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc')
        l2_reg = self.l2_reg
        for epoch in range(total_epoch):
            self.logger.log('============ Epoch {} ============'.format(epoch))
            train_loss, train_acc, train_n = 0, 0, 0
            start_time = time.time()
            counter = 0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                out = self.model(images)
                p_out = nn.Softmax(dim=1)(out)

                # alpha = 0.5 * eps
                adv_images = images.clone() + self.eps * 0.5 * torch.sign(torch.tensor([0.5]).to(self.device) - torch.rand_like(images))
                adv_images = torch.clamp(adv_images, min=0, max=1)
                adv_images = adv_images.detach().requires_grad_(True)

                self.model.eval()
                opt.zero_grad()
                out = self.model(adv_images)
                r_out = nn.Softmax(dim=1)(out)

                if counter % 2 == 1:
                    loss = criterion(out, labels) + l2_reg * (((p_out - r_out) ** 2.0).sum(1)).mean(0)
                else:
                    loss = criterion(r_out, labels)

                loss.backward()

                adv_images = adv_images.detach() + self.eps * adv_images.grad.detach().sign()
                delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
                adv_images = torch.clamp(images + delta, min=0, max=1).detach()

                self.model.train()

                # train
                opt.zero_grad()
                adv_output = self.model(adv_images)
                output = self.model(images)

                q_out = nn.Softmax(dim=1)(adv_output)
                p_out = nn.Softmax(dim=1)(output)

                closs = criterion(output, labels)
                reg_loss = ((p_out - q_out) ** 2.0).sum(1).mean(0)

                loss = closs + l2_reg * reg_loss
                loss.backward()
                opt.step()
                
                # Record GPU memory
                self._record_gpu_memory()

                counter += 1

                # weight average
                if weight_average:
                    """
                    for p_wa, p_model in zip(wa_model.parameters(), self.model.parameters()):
                        p_wa.data = p_wa.data * tau + p_model.data * (1 - tau)
                    """
                    for key, value in self.model.state_dict().items():
                        exp_avg[key] = (1 - tau) * value + tau * exp_avg[key]

                train_loss += loss.item() * labels.size(0)
                train_acc += (output.max(1)[1] == labels).sum().item()
                train_n += labels.size(0)

            scheduler.step()

            if epoch == 85:
                l2_reg = l2_reg * 4

            if weight_average:
                self.model.eval()
                wa_model.load_state_dict(exp_avg)
                wa_model.eval()
                # update bn
                """
                for module1, module2 in zip(wa_model.modules(), self.model.modules()):
                    if isinstance(module1, torch.nn.modules.batchnorm._BatchNorm):
                        module1.running_mean = module2.running_mean
                        module1.running_var = module2.running_var
                        module1.num_batches_tracked = module2.num_batches_tracked
                """
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
