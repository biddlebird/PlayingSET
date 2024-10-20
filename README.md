### README: SET Card Game Image Classification

---

#### **Project Overview**

This project focuses on using machine learning models to classify images from the SET card game. The SET card game consists of cards with different attributes—shape, color, number, and shading—where the objective is to identify valid sets of cards. Our goal is to develop a model that can classify these images based on the card attributes using deep learning techniques such as Convolutional Neural Networks (CNN) and transfer learning with MobileNetV2.

---

#### **Table of Contents**
1. [Project Setup](#project-setup)
2. [Data](#data)
3. [Modeling Approach](#modeling-approach)
4. [Results](#results)
5. [Next Steps](#next-steps)
6. [Contributors](#contributors)

---

### **Project Setup**

To run the project, follow these steps:

#### **1. Clone the Repository**

```bash
git clone https://github.com/your-username/SET-Card-Classification.git
cd SET-Card-Classification
```

#### **2. Install Dependencies**

Create a virtual environment (recommended) and install the required dependencies:

```bash
python3 -m venv set-game-env
source set-game-env/bin/activate   # For Linux/Mac
set-game-env\Scripts\activate      # For Windows

pip install -r requirements.txt
```

#### **3. Dataset Structure**

Make sure your dataset of SET card images is organized into folders, each named by the card’s attributes (e.g., `one_blue_diamond_empty`). Images should be in PNG format.

Example directory structure:
```
/flattened_images/
    /one_blue_diamond_empty/
        01.png
        02.png
    /two_green_squiggle_shaded/
        01.png
```

#### **4. Running the Model**

To train and evaluate the model, run the following script:

```bash
python train_model.py
```

This will initialize both the CNN and MobileNetV2 models and output the results, including accuracy and loss plots. The script also includes hyperparameter tuning and early stopping mechanisms to optimize training.

---

### **Data**

The dataset consists of SET card images categorized by four attributes:
1. **Number of symbols** (1, 2, or 3)
2. **Color** (red, green, or blue)
3. **Shape** (diamond, squiggle, or oval)
4. **Shading** (empty, shaded, or filled)

All images are resized to 150x150 pixels, and pixel values are rescaled to be between 0 and 1 using the `ImageDataGenerator` in Keras.

The dataset is split into:
- 80% training data
- 20% validation data

---

### **Modeling Approach**

This project uses two primary models for image classification:

#### **1. Convolutional Neural Network (CNN)**

A custom CNN architecture was built from scratch, consisting of multiple convolutional layers, pooling layers, and dense layers. The model is compiled using the `adam` optimizer, categorical cross-entropy loss, and accuracy as the evaluation metric.

#### **2. MobileNetV2 (Transfer Learning)**

MobileNetV2, a pre-trained model, was fine-tuned for our dataset. The top layers of the network were customized to fit the number of categories in the dataset. Transfer learning allows for faster training and leveraging previously learned features for image classification.

#### **Optimization Techniques**
- **Early Stopping**: Stops training when no further improvement in validation loss is observed.
- **Batch Normalization**: Included to improve training stability and convergence.
- **Hyperparameter Tuning**: Explores different parameters to achieve optimal performance, such as learning rate, batch size, and number of filters in convolutional layers.

---

### **Results**

The models were evaluated based on the following metrics:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**

Performance metrics were plotted and compared over 500 epochs. The confusion matrices and classification reports highlight how well each model performs across different card attributes.

Both the CNN and MobileNetV2 models showed similar performance, and optimization efforts are ongoing to improve accuracy.

---

### **Next Steps**

1. **Dataset Expansion**: Increasing the dataset size and adding more variations in lighting, rotation, and card conditions will help improve model performance.
2. **SQL Integration**: Future work will store image paths and metadata in an SQL database for efficient data retrieval.
3. **Additional Models**: Experiment with other architectures such as ResNet or EfficientNet to further improve classification accuracy.
4. **Hyperparameter Optimization**: Implement a more extensive hyperparameter search using tools like `KerasTuner` to fine-tune model parameters.

---

### **Contributors**

- **Kayla Biddle** - Machine learning engineer and project lead.

Feel free to contribute to this project by submitting pull requests or opening issues for bugs and suggestions.
