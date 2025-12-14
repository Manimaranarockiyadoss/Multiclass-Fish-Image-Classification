# 🐟 Multiclass Fish Image Classification

## 📌 Project Overview
This project focuses on building a deep learning–based image classification system to identify and classify fish images into multiple species.  
The solution combines a Convolutional Neural Network (CNN) trained from scratch and transfer learning using pre-trained deep learning models to achieve higher accuracy and robustness.

The project also includes saving the trained model and deploying it as a user-friendly web application using Streamlit for real-time predictions.

---

## 🎯 Problem Statement
Manual identification of fish species from images is time-consuming and error-prone.  
This project aims to automate fish species classification using deep learning techniques by training and evaluating multiple models and deploying the best-performing model for practical use.

---

## 🧠 Business Use Cases
- **Enhanced Accuracy**  
  Identify the most suitable deep learning architecture for fish image classification.

- **Deployment-Ready Solution**  
  Provide a real-time prediction system through a web application.

- **Model Comparison**  
  Evaluate and compare CNN and transfer learning models to select the optimal approach.

- **Automation in Fisheries**  
  Support fishery management systems and research applications with AI-driven classification.

---

## 🛠️ Tech Stack
### Programming Language
- Python

### Deep Learning Frameworks
- TensorFlow
- Keras

### Libraries
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

### Deployment
- Streamlit

---

## 📂 Project Structure
Fish-Image-Classification/
│
├── dataset/
│ └── fish_images/
│ ├── class_1/
│ ├── class_2/
│ └── class_n/
│
├── models/
│ └── best_fish_model.h5
│
├── notebooks/
│ └── fish_classification.ipynb
│
├── app.py
├── README.md

yaml
Copy code

---

## 🔄 Project Workflow

### 1. Data Preprocessing & Augmentation
- Rescaled image pixel values to the range [0, 1]
- Applied data augmentation techniques:
  - Rotation
  - Zoom
  - Horizontal flipping
- Improved model generalization and robustness

### 2. Model Training
#### CNN from Scratch
- Designed a custom CNN architecture
- Trained as a baseline model

#### Transfer Learning
- Utilized pre-trained models such as:
  - VGG16
  - ResNet50
  - MobileNet
  - InceptionV3
  - EfficientNetB0
- Fine-tuned models on the fish image dataset

### 3. Model Evaluation
- Compared models using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix
- Visualized training and validation accuracy/loss curves

### 4. Model Selection & Saving
- Selected the model with the highest validation accuracy
- Saved the trained model in `.h5` format for reuse and deployment

### 5. Deployment
- Built a Streamlit web application to:
  - Upload fish images
  - Predict fish species
  - Display confidence scores

---

## 📊 Evaluation Metrics
- **Accuracy** – Overall correctness of predictions  
- **Precision** – Correctness of positive predictions  
- **Recall** – Ability to identify all relevant classes  
- **F1-Score** – Balance between precision and recall  
- **Confusion Matrix** – Class-wise performance visualization  

---

## 📈 Results & Insights
- Transfer learning models significantly outperformed the CNN trained from scratch.
- Data augmentation improved model generalization.
- VGG16 provided strong baseline performance for fish classification.
- The deployed model is capable of real-time predictions with high confidence.

---

## 🧪 Testing & Validation
- Evaluated models on unseen validation data
- Compared training vs validation performance to monitor overfitting
- Selected the most stable and accurate model for deployment

---

## 🎯 Conclusion
This project demonstrates the effective application of deep learning and transfer learning for image classification tasks.  
By combining robust preprocessing, advanced model architectures, and real-time deployment, the solution is suitable for practical use in fisheries, research, and image-based classification systems.

---

## 👤 Author
**Manimaran Arockiyadoss**  
Deep Learning & Data Analytics Enthusiast
