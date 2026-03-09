import torch
from torch.nn import CrossEntropyLoss


def calculate_self_fitting(model, dataloader, num_classes=10, epsilon=8.0 / 255):
    loss_function = CrossEntropyLoss(reduction='none')
    improved_loss_count, self_fitting_count, total_samples = 0, 0, 0

    model.eval()
    device = next(model.parameters()).device

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        for target_class in range(num_classes):
            target_labels = torch.full_like(labels, target_class, device=device)
            perturbed_inputs = inputs.clone().requires_grad_(True)

            original_loss = loss_function(model(perturbed_inputs), target_labels)
            original_loss.mean().backward()

            perturbations = perturbed_inputs.grad.sign()
            perturbed_inputs = torch.clamp(perturbed_inputs + epsilon * perturbations, 0, 1).detach()
            perturbed_outputs = model(perturbed_inputs)

            predictions = perturbed_outputs.argmax(dim=1)
            self_fitting_count += (predictions == target_labels).sum().item()

            new_loss = loss_function(perturbed_outputs, target_labels)
            improved_loss_count += (new_loss < original_loss).sum().item()

            total_samples += labels.size(0)

    improved_loss_ratio = improved_loss_count / total_samples
    correct_predictions_ratio = self_fitting_count / total_samples

    return improved_loss_ratio, correct_predictions_ratio
