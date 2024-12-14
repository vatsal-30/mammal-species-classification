# Instructions on How to Run the Pre-trained Model on the Sample Test Dataset

This guide explains how to load and run the pre-trained model on the sample test dataset and visualize predictions.

---

## 1. **List of Available Models**

You have the following pre-trained models:

1. **DenseNet-121 Models**:

   - `DenseNet121_model1.pth`
   - `DenseNet121_model2.pth`
   - `DenseNet121_model3.pth`

2. **ResNet-50 Models**:

   - `ResNet50_model1.pth`
   - `ResNet50_model2.pth`
   - `ResNet50_model3.pth`

3. **InceptionV3 Models**:

   - `final_model_inceptionv3(Dataset1).pth`
   - `final_model_inceptionv3(Dataset2).pth`
   - `final_model_inceptionv3(Dataset3).pth`

4. **Transfer learning Models**:
   - `Transfer_learning_model1_resnet50.pth`
   - `Transfer_learning_model2_resnet50.pth`

---

## 2. **Install Dependencies**

Make sure you have all the necessary libraries installed. Refer to `Requirement.txt` for installation instructions.

## 3. **Load a Pre-trained Model**

Replace `<model_path>` with the path to the desired model.

For example:

model = torch.load("DenseNet121_model1.pth")
model = torch.load("ResNet50_model2.pth")
model = torch.load("final_model_inceptionv3(Dataset3).pth")
model = torch.load("Transfer_learning_model2_resnet50.pth")

```python
import torch

# Load the pre-trained model
model = torch.load("<model_path>")

# Set the model to evaluation mode
model.eval()
```

## 3. **Example of usage of a Pre-trained Model**

Make sure you replace all the path file correctly. Replace the class_names for the correspondce dataset and model as following:

model1 -> dataset1, model2 -> dataset2, model3 -> dataset3

dataset 1 : class_names=['Buffalo', 'Elephant', 'Rhino', 'Zebra']
dataset 2 : class_name=['African_Elephant', 'Amur_Leopard', 'Arctic_Fox', 'Chimpanzee', 'Jaguars', 'Lion', 'Orangutan', 'Panda', 'Panthers', 'Rhino', 'cheetahs']
dataset 3 : class_names=['african_elephant', 'alpaca', 'american_bison', 'anteater', 'arctic_fox', 'armadillo', 'baboon', 'badger', 'blue_whale' , 'brown_bear' , 'camel', 'dolphin', 'giraffe', 'groundhog', 'highland_cattle', 'horse', 'jackal', 'kangaroo', 'koala', 'manatee', 'mongoose', 'mountain_goat', 'opossum', 'orangutan', 'otter', 'polar_bear', 'porcupine', 'red_panda', 'rhinoceros', 'sea_lion', 'seal', 'snow_leopard', 'squirrel', 'sugar_glider', 'tapir', 'vampire_bat', 'vicuna', 'walrus', 'warthog', 'water_buffalo', 'weasel', 'wildebeest', 'wombat', 'yak', 'zebra']

```python
import torch
from PIL import Image
import os
import random
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms

dataset_path = r'<path to>\Sample_Dataset\Dataset1'

class_names=['Buffalo', 'Elephant', 'Rhino', 'Zebra']

print(class_names)

model = torch.load("<model_path>")

model.eval()


def show_random_test_prediction(model, dataset_path, class_names):
    model.eval()

    image_files = [f for f in os.listdir(dataset_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

    random_idx = random.randint(0, len(image_files) - 1)
    image_file = image_files[random_idx]
    image_path = os.path.join(dataset_path, image_file)

    original_image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image_tensor = transform(original_image).unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor)
        _, predicted_label = torch.max(output, 1)

    # Display the original image with the prediction
    plt.imshow(original_image)
    plt.title(f"Predicted: {class_names[predicted_label.item()]}, Image: {image_file}")
    plt.axis("off")
    plt.show()

show_random_test_prediction(model, dataset_path, class_names)
```
