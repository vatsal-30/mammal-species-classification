# **Instructions on How to Train/Validate Your Models**

## **1. Prerequisites**

Make sure you have all the necessary libraries installed. Refer to `Requirement.txt` for installation instructions.

## **2. Dataset Preparation**

Make sure you have all the necessary datasets installed. Refer to `Datasets.md` for installation instructions.

## **3. Training the Model**

### **3.1 Model Selection**

The repository includes multiple pre-defined models (e.g., ResNet50, DenseNet121, InceptionV3). Choose the model you want to train.

### **3.2 Training Steps**

1. Ensure your dataset path is correctly set in the script.
2. Adjust hyperparameters such as:

   - Number of epochs
   - Batch size
   - Learning rate

3. Run the training script in jupiternotebook:

### **3.3 Monitor Training**

During training, monitor the following:

- **Loss**: Training loss after each epoch.
- **Accuracy**: Validation accuracy after each epoch.

The script will print these values to the console.

### **3.4 Saving the Model**

After training, the model will be automatically saved to a file, such as:

```
trained_model_resnet50.pth
```

### **4.3 Results**

The script will display the **loss** and **accuracy** on the test set and print performacne metrics.

## **5. Notes**

- Ensure your dataset is correctly structured before running any training/validation scripts.
- If running on **GPU**, ensure that CUDA is properly installed and configured.
- Adjust the paths in the scripts as necessary to match your local directory structure.
