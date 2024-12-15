# Mammal Species Classification Using Convolutional Neural Networks (CNNs)

## Project Description

The increasing threat to biodiversity due to environmental changes and human activities necessitates advanced tools for conservation. Traditional species identification methods are manual, time-consuming, and prone to errors. This project introduces a robust and automated mammal species classification system leveraging **Convolutional Neural Networks (CNNs)** to improve identification accuracy, reduce human effort, and enhance conservation efforts.

Our solution employs three state-of-the-art CNN architectures — **ResNet50**, **DenseNet121**, and **InceptionV3** — to address challenges like dataset imbalance, intra-class variation, inter-class similarity, and environmental variability. Techniques such as **multi-scale feature extraction**, **transfer learning**, and **data augmentation** are integrated to enhance model performance. Additionally, interpretability tools like **Grad-CAM** and **t-SNE** provide insights into model predictions, ensuring transparency and aiding conservationists.

The system is trained and evaluated on three diverse datasets sourced from Kaggle:

1. **Animal Computer Vision Clean Dataset** (4,000 images, 4 classes) — Controlled conditions, minimal variability.
2. **Danger of Extinction Animal Image Set** (6,484 images, 11 classes) — Focuses on endangered species but suffers from class imbalance.
3. **Mammals Image Classification Dataset** (13,800 images, 45 classes) — Diverse class representations with significant intra-class variation and inter-class similarity.

### Key Features

- **Multiple CNN Models**: ResNet50, DenseNet121, and InceptionV3 for robust feature extraction and classification.
- **Transfer Learning**: Fine-tuning pre-trained models to address dataset-specific challenges and improve accuracy.
- **Performance Metrics**: Evaluation based on accuracy, precision, recall, and F1-score.
