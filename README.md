# Facial Emotion Recognition using CNN

This repository contains an end-to-end implementation of a **Convolutional Neural Network (CNN)** for facial emotion recognition using the **FER-2013 dataset**.  
We trained and evaluated a deep learning model that classifies faces into **7 emotion categories**.

---

## 📌 Project Overview
Facial Emotion Recognition (FER) is an important application of computer vision and deep learning.  
The goal of this project is to automatically detect and classify human emotions from facial images into:

- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

---

## ⚙️ Steps Implemented in the Notebook

### **Step 1. Dataset Setup**
- Used **Kaggle API** authentication (`kaggle.json`) to fetch the FER-2013 dataset.  
- Extracted and organized the dataset into `train/`, `val/`, and `test/` directories.  

---

### **Step 2. Data Preprocessing**
- Loaded dataset images using `ImageDataGenerator`.  
- Created training, validation, and testing generators with **rescaling and batching**.  
- Dataset summary:
  - **Train:** 22,968 images  
  - **Validation:** 5,741 images  
  - **Test:** 7,178 images  
  - **Classes:** 7 (emotions)  

---

### **Step 3. CNN Model**
- Built a **Sequential CNN** with the following layers:
  - Conv2D + BatchNormalization + MaxPooling (×3 blocks)  
  - Flatten layer  
  - Dense (fully connected) layer with Dropout  
  - Final softmax layer for **7-class output**  
- Compiled model with **Adam optimizer** and `categorical_crossentropy` loss.  

---

### **Step 4. Training**
- Applied **data augmentation** (rotation, zoom, horizontal flips).  
- Trained model with:
  - **EarlyStopping** (monitors `val_loss`, patience=5)  
  - **ModelCheckpoint** (saves best model as `best_model.h5`)  
- Achieved stable training while avoiding overfitting.  

---

### **Step 5. Evaluation**
- Evaluated model performance on the **test set**.  
- Reported **test accuracy and loss**.  
- Plotted **training vs validation curves** for accuracy and loss.  

---

### **Step 6. Inference**
- Ran inference on random samples from the test set.  
- Displayed:
  - The **input face image**  
  - The **true label**  
  - The **predicted emotion**  

---

## 📊 Results
- The model successfully classifies images into **7 emotion categories**.  
- Performance varies across emotions (commonly harder: *fear* vs *surprise*).  
- Future improvements may include transfer learning with **pretrained CNNs (VGG, ResNet)** for higher accuracy.  

---

## 🛠️ Tech Stack
- **Python**  
- **TensorFlow / Keras**  
- **NumPy, Pandas, Matplotlib**  
- **Kaggle API**  

---

## 🚀 Future Work
- Fine-tune pretrained CNN architectures (e.g., VGG16, ResNet50).  
- Hyperparameter optimization for improved accuracy.  
- Build a **real-time emotion detection app** with OpenCV.  

---

## 📂 Repository Structure
