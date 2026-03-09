# FastAT Benchmark

## 🚀 Overview

The FastAT Benchmark provides a rigorous and fair evaluation framework for fast adversarial training methods. Unlike public leaderboards that allow diverse combinations of model architectures, data sources, and computational budgets, our benchmark establishes conditions where all methods compete on equal footing.

This platform implements over twenty representative FastAT methods with a unified codebase, ensuring fair and reproducible comparison across different algorithmic innovations. The benchmark systematically removes advantages from massive computational resources and unlimited external data, providing the research community with a transparent baseline for evaluating fast adversarial training techniques.

## ✨ Key Features

- **🎯 Unified Architecture**: All methods are evaluated on identical network structures, eliminating performance differences arising from architectural advantages rather than training procedures.
- **⚙️ Standardized Settings**: Consistent training schedules, optimizers, learning rate policies, and data augmentation strategies prevent the experimental setup from favoring any particular method.
- **🚫 No External Data**: Strict prohibition of using additional or synthetic data beyond the original benchmark training set ensures observed gains stem solely from the learning algorithm.
- **📊 Dual-Metric Framework**: Evaluates both robustness performance (accuracy against strong attacks) and computational cost (GPU hours and memory footprint).
- **🔬 Comprehensive Evaluation**: Includes diverse attack methods: PGD with varying iterations, AutoAttack, and CR Attack for thorough robustness assessment.
- **🔄 Unified Implementation**: Re-implemented over a dozen FastAT methods with a common interface for data loading, model initialization, training loops, and evaluation protocols.

## 📋 Design Principles and Controlled Constraints

Strict experimental control forms the foundation of the FastAT Benchmark. Several key constraints support this central design principle:

- **Unified Architecture**: All methods undergo evaluation on identical network structures.
- **Standardized Training Settings**: Consistent training schedules, optimizers, learning rate policies, and data augmentation strategies.
- **No External Data**: Strictly prohibits using any additional or synthetic data beyond the original benchmark training set.

## 📊 Dual-Metric Evaluation Framework

FastAT methods aim to achieve robustness through efficient means, making it essential to evaluate both effectiveness and computational cost:

- **Robustness Performance**: Measured as final robust accuracy against strong adversarial attacks.
- **Training Cost**: Encompasses critical resource requirements such as total GPU hours and peak memory footprint.

## 🔄 Benchmark Training Flow

1. **Load Configuration** (common.yaml, method.yaml)
2. **Initialization Phase**: Data Loaders, Model Init, Optimizer
3. **Select FastAT Method**
4. **Training Phase**: Training Loop, Validation
5. **Evaluation Phase**: Final Evaluation with PGD Attack, AutoAttack, CR Attack
6. **Log Results & Metrics**

## 📈 Evaluation Setup

### Datasets and Models

- **CIFAR-10**: ResNet-18 architecture
- **CIFAR-100**: ResNet-18 architecture
- **Tiny-ImageNet**: PreActResNet-18 architecture

Each experiment runs three times with different random seeds, and the mean of these three runs constitutes the final reported result.

### Training Configuration

- **Optimizer**: SGD with initial learning rate of 0.1, momentum of 0.9, and weight decay of 5×10⁻⁴
- **Training Duration**: 100 epochs with OneCycleLR scheduler
- **Batch Size**: 128
- **Data Augmentation**: 4-pixel padding followed by random cropping and horizontal flipping
- **Additional Techniques**: Weight averaging (τ = 0.9995) and label smoothing
- **Validation Sets**: 1,000 images for CIFAR-10, 1,000 images for CIFAR-100, 2,000 images for Tiny-ImageNet

## 🛡️ Supported Methods

- [FGSM-RS](https://arxiv.org/pdf/2001.03994)
- [GRAD-ALIGN](https://proceedings.neurips.cc/paper_files/paper/2020/file/b8ce47761ed7b3b6f48b583350b7f9e4-Paper.pdf)
- [FREE-AT](https://proceedings.neurips.cc/paper_files/paper/2019/file/7503cfacd12053d309b6bed5c89de212-Paper.pdf)
- [FGSM-PGI](https://arxiv.org/pdf/2207.08859)
- [FGSM-MEP-CS](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhao_Fast_Adversarial_Training_with_Smooth_Convergence_ICCV_2023_paper.pdf)
- [FGSM-RS-CS](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhao_Fast_Adversarial_Training_with_Smooth_Convergence_ICCV_2023_paper.pdf)
- [FGSM-PCO](https://arxiv.org/pdf/2407.12443)
- [SSAT](https://ojs.aaai.org/index.php/AAAI/article/download/16989/16796)
- [NU-AT](https://proceedings.neurips.cc/paper_files/paper/2021/file/62889e73828c756c961c5a6d6c01a463-Paper.pdf)
- [N-FGSM](https://proceedings.neurips.cc/paper_files/paper/2022/file/5434a6b40f8f65488e722bc33d796c8b-Paper-Conference.pdf)
- [ZERO-GRAD](https://www.sciencedirect.com/science/article/pii/S2667305323000832)
- [GAT](https://proceedings.neurips.cc/paper_files/paper/2020/file/ea3ed20b6b101a09085ef09c97da1597-Paper.pdf)
- [AAER](https://proceedings.neurips.cc/paper_files/paper/2023/file/d65befe6b80ecf7f180b4def503d7776-Paper-Conference.pdf)
- [ELLE](https://arxiv.org/pdf/2401.11618)
- [FGSM-AT](https://arxiv.org/pdf/1412.6572)
- [PGD-AT](https://arxiv.org/pdf/1706.06083)
- [PGD-AT-WA](https://arxiv.org/pdf/1803.05407)
- [FGSM-UAP](https://ojs.aaai.org/index.php/AAAI/article/view/30147/32032)
- [FGSM-CUAP](https://ojs.aaai.org/index.php/AAAI/article/view/30147/32032)
- [FGSM-FUAP](https://ojs.aaai.org/index.php/AAAI/article/view/30147/32032)
- [LIET](https://openaccess.thecvf.com/content/ICCV2025/papers/Pan_Mitigating_Catastrophic_Overfitting_in_Fast_Adversarial_Training_via_Label_Information_ICCV_2025_paper.pdf)

## 📊 Results

The benchmark reveals substantial performance variations among FastAT methods when evaluated under strictly controlled conditions. The results simultaneously capture robustness under multiple attack strengths and computational costs, enabling researchers to identify methods that achieve favorable trade-offs between defensive capability and training efficiency.

## 📈 Training Dynamics

Monitoring training dynamics provides crucial insights into method stability. Training progress curves track training accuracy alongside PGD-10 and FGSM robustness on the validation set throughout training. These curves serve a diagnostic purpose: methods experiencing catastrophic overfitting exhibit a characteristic pattern where training accuracy remains high while robust accuracy suddenly collapses.

## 🚀 Getting Started

To use the FastAT Benchmark, follow these steps:

1. Clone the repository
2. Install the required dependencies
3. Configure the benchmark settings in `common.yaml` and method-specific settings in `method.yaml`
4. Run the benchmark using the provided scripts
5. Analyze the results using the visualization tools

## 🤝 Contributing

Contributions to the FastAT Benchmark are welcome! Please feel free to submit pull requests or open issues to help improve the framework.

## 📚 References

For more details, please refer to the paper:
TODO

