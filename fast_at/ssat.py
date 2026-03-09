import copy
import os
import time

import torch
from torch.nn import CrossEntropyLoss
from torchattacks import PGD, FGSM

from .at_base import ATBase


class SSAT(ATBase):

    def __init__(self, model, config_path=None, config=None, log_dir='./log/', name='ssat', device=None, seed=0):
        super(SSAT, self).__init__(model, config_path=config_path, config=config,
                                   log_dir=log_dir, name=name, device=device, seed=seed)
        self.c = self.config.get('c', 3)
        self.alpha = self.config.get('alpha', 10.0 / 255)
        self.inf_batch = self.config.get('inf_batch', 1024)

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
        self.logger.log('c: {}'.format(self.c))
        self.logger.log('alpha: {}'.format(self.alpha))
        self.logger.log('inf_batch: {}'.format(self.inf_batch))
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
                batch_size = images.size(0)
                images, labels = images.to(self.device), labels.to(self.device)
                # add uniform noise
                adv_images = images.clone() + torch.zeros_like(images).uniform_(-self.eps, self.eps)
                adv_images = torch.clamp(adv_images, min=0, max=1).detach()
                adv_images = adv_images.detach().requires_grad_(True)

                output = self.model(adv_images)
                loss = criterion(output, labels)
                loss.backward()

                adv_images = adv_images.detach() + self.alpha * adv_images.grad.detach().sign()
                delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
                adv_images = torch.clamp(images + delta, min=0, max=1).detach()

                logit_clean = output
                pert = (adv_images - images).detach()

                _, pre_clean = torch.max(logit_clean.data, 1)
                correct = (pre_clean == labels)
                correct_idx = torch.masked_select(torch.arange(batch_size).to(self.device), correct)
                wrong_idx = torch.masked_select(torch.arange(batch_size).to(self.device), ~correct)

                # Use misclassified images as final images.
                adv_images[wrong_idx] = images[wrong_idx].detach()

                Xs = (torch.cat([images] * (self.c - 1)) + torch.cat(
                    [torch.arange(1, self.c).to(self.device).view(-1, 1)] * batch_size, dim=1).view(-1, 1, 1,
                                                                                                    1) * torch.cat(
                    [pert / self.c] * (self.c - 1)))
                Ys = torch.cat([labels] * (self.c - 1))

                idx = correct_idx
                idxs = []
                self.model.eval()
                with torch.no_grad():
                    for k in range(self.c - 1):
                        # Stop iterations if all checkpoints are correctly classiffied.
                        if len(idx) == 0:
                            break
                        # Stack checkpoints for inference.
                        elif (self.inf_batch >= (len(idxs) + 1) * len(idx)):
                            idxs.append(idx + k * batch_size)
                        else:
                            pass

                        # Do inference.
                        if (self.inf_batch < (len(idxs) + 1) * len(idx)) or (k == self.c - 2):
                            # Inference selected checkpoints.
                            idxs = torch.cat(idxs).to(self.device)
                            pre = self.model(Xs[idxs]).detach()
                            _, pre = torch.max(pre.data, 1)
                            correct = (pre == Ys[idxs]).view(-1, len(idx))

                            # Get index of misclassified images for selected checkpoints.
                            max_idx = idxs.max() + 1
                            wrong_idxs = (idxs.view(-1, len(idx)) * (1 - correct * 1)) + (max_idx * (correct * 1))
                            wrong_idx, _ = wrong_idxs.min(dim=0)

                            wrong_idx = torch.masked_select(wrong_idx, wrong_idx < max_idx)
                            update_idx = wrong_idx % batch_size
                            adv_images[update_idx] = Xs[wrong_idx]

                            # Set new indexes by eliminating updated indexes.
                            idx = torch.tensor(list(
                                set(idx.cpu().data.numpy().tolist()) - set(update_idx.cpu().data.numpy().tolist())))
                            idxs = []

                self.model.train()
                # train
                opt.zero_grad()
                adv_images = adv_images.detach()
                output = self.model(adv_images)
                loss = criterion(output, labels)
                loss.backward()
                opt.step()

                # Record GPU memory
                self._record_gpu_memory()

                # weight average
                if weight_average:
                    for key, value in self.model.state_dict().items():
                        exp_avg[key] = (1 - tau) * value + tau * exp_avg[key]

                train_loss += loss.item() * labels.size(0)
                train_acc += (output.max(1)[1] == labels).sum().item()
                train_n += labels.size(0)

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
