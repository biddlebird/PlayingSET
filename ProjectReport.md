# PlayingSET
Teaching my computer how to play the game of SET using machine learning

# Setting Up
## Install Required Libraries:
```
pip install tensorflow pandas scikit-learn matplotlib pillow
```

### Project Report: SET Card Game Image Classification

#### Introduction
The goal of this project is to build and train a machine learning model that classifies images of SET cards, a card game involving combinations of cards with different attributes (shape, color, number, shading). Using deep learning techniques such as Convolutional Neural Networks (CNN) and transfer learning with MobileNetV2, we aim to predict valid SETs based on the images. This report explains how the project meets the rubric's requirements and outlines the next steps for improvement.

---

### Data Model Implementation 

#### 1. Python Script for Model Initialization, Training, and Evaluation 

The project involves two machine learning models:
   - A custom Convolutional Neural Network (CNN) was designed and trained using images of SET cards.
   - A MobileNetV2 transfer learning model was also implemented to compare performance with the CNN.

Both models were initialized, trained, and evaluated using a Python script in Jupyter Notebook. The script utilizes TensorFlow and Keras libraries, and the model’s accuracy, loss, and classification reports were evaluated using performance metrics like accuracy, precision, recall, and F1-score.

#### 2. Data Cleaning, Normalization, and Standardization 

Data cleaning steps were conducted to ensure that the input images were in a uniform format and structure. The images were resized to 150x150 pixels and rescaled to have pixel values in the range [0, 1] using the `ImageDataGenerator` from TensorFlow. This ensures the models can process the data consistently.

#### 3. Data Retrieval from SQL or Spark 

To meet this requirement, we simulated an environment where the images were retrieved from a local directory. However, future steps will involve using a SQL database to retrieve and store image metadata. The images can be saved as blobs or paths in SQL databases and loaded dynamically using Python’s SQL libraries.

#### 4. Model Predictive Power with 75% Classification Accuracy or 0.80 R-squared 

Both models—CNN and MobileNetV2—have been trained and evaluated. Currently, neither model achieves the target of 75% classification accuracy, with performance hovering around 1-2%. Future steps involve tuning hyperparameters, utilizing more data, and applying techniques such as image augmentation and hyperparameter tuning to improve performance.

---

### Data Model Optimization 

#### 1. Documenting Model Optimization Process 

The project uses `KerasTuner` to search for optimal hyperparameters (e.g., number of layers, filters, and kernel size) for the CNN model. Iterative changes to the model architecture, including the addition of batch normalization and early stopping, have been recorded in the code. Each model run and its respective performance is logged, providing a clear record of improvements and outcomes in the script.

#### 2. Displaying Model Performance 

The final accuracy, loss, precision, recall, and classification reports are printed and plotted using Matplotlib in the script. Comparative plots for CNN and MobileNetV2 performance across epochs are displayed to illustrate how well each model performed in terms of accuracy and loss.

---

### GitHub Documentation 

#### 1. GitHub Repository Organization and .gitignore 

The GitHub repository for this project is organized, with unnecessary files excluded using a `.gitignore`. It only contains essential files such as the Python scripts, dataset information, and documentation. Logs, temporary files, and unnecessary image files are excluded.

#### 2. README Customization 

The README file is customized to present a polished overview of the project. It includes:
   - Project goals
   - Instructions on how to run the script
   - The methodology for training and evaluating the models
   - Links to the dataset and requirements for setting up the environment

The README provides enough information for someone new to the project to replicate and understand the process.

---

### Presentation 

[My Google Slides Presentation](https://docs.google.com/presentation/d/1De9gkM_K7HPsZQEqMwr6U0gdLST_sCnUN6OovnEjWo8/edit?usp=sharing)

#### Relevant Content 

The content of the presentation will focus on the project’s key aspects, including data preparation, the choice of models (CNN and MobileNetV2), and how model performance was optimized using hyperparameter tuning and early stopping. The project’s challenges and potential improvements will also be covered.

#### 4. Audience Engagement 

To maintain audience interest, we will present real-time demos of model performance, showing how the model predicts SET card combinations. The visual comparison of CNN and MobileNetV2 performance using graphs will provide a compelling narrative of model optimization.

---

### Next Steps (Conclusion and Future Work)

1. Model Improvement: While current performance is below expectations, the next steps involve increasing the dataset size, applying more robust data augmentation techniques, and continuing hyperparameter tuning to improve classification accuracy.
   
2. Data Storage in SQL: Future work will integrate SQL-based data storage to enable dynamic retrieval of images and their metadata for easier scaling.

3. Additional Models: Exploring more advanced models such as ResNet or EfficientNet, and experimenting with deeper architectures can provide insights into improving classification performance for SET card images.

4. Further Optimization: Continuing the optimization process with a more diverse dataset and potential cross-validation will help achieve better generalization and accuracy.

In summary, this project has made significant strides in model development and optimization, with ongoing efforts to improve accuracy.
