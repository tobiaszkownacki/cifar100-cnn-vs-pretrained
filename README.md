# Image Classification on CIFAR-100: Exploratory Study for GOLEM AI Academic Circle

------------------

## Project Context

This preliminary project was conducted as part of the GOLEM Artificial Intelligence Scientific Circle, focusing on comparing custom and pre-trained CNN architectures for image classification. The study aimed to establish foundational workflows for future research in model optimization and transfer learning.

Collaborators: Adam Szostek, Kacper Ptaszek

------------------

## Key Objectives

- Design a modular CNN architecture with separated feature extraction and classification components.
- Compare the custom model’s performance against pre-trained networks, focusing on ResNet-18.
- Quantify trade-offs between accuracy, training efficiency, and computational complexity.

------------------

## Technologies Used

- **Language**: Python
- **Frameworks**: PyTorch, torchvision
- **Supporting Libraries**: NumPy, Matplotlib
- **Dataset**: CIFAR-100

------------------

## Methodology

1. **Data Preprocessing**:
   - Normalized pixel values and applied augmentations (random crops, horizontal flips).
   - Utilized PyTorch's `DataLoader` for batch processing and dataset splitting.

2. **Model Architectures**:
   - **Custom CNN**: A sequential network with convolutional blocks, max pooling, and dropout layers.
   - **ResNet-18**: Pre-trained weights from ImageNet, with the final layer adapted for 100-class classification.

3. **Training Configuration**:
   - Optimizer: Stochastic Gradient Descent (SGD) with momentum, Adam, AdamW, RMSprop
   - Loss Function: Cross-entropy loss.

4. **Evaluation Metrics**:
   - Tracked validation accuracy, loss function value.

------------------

## Experimental Results

| Model       | Validation Accuracy (%) | Parameters (Millions) |
|-------------|-------------------------|-----------------------|
| Custom CNN  | 66.3                    | 87,6                  |
| ResNet-18   | 78.2                    | 11,6                  |

------------------

## Key Findings

- The pre-trained ResNet-18 achieved **12% higher accuracy** compared to the custom model.
- The custom CNN has significantly more parameters but yields worse accuracy.
- Training one epoch of the ResNet-18 model takes considerably more time.
- Custom model requires more epochs to achieve acceptable accuracy.
