# Brain-Tumor-Classification
 # 🧠 Brain Tumor MRI Classification using Deep Learning (MobileNetV2 + Streamlit)

This project is a **Brain Tumor MRI Image Classification system** that predicts the tumor type from MRI images using **Deep Learning** and a **Transfer Learning model (MobileNetV2)**.  
An interactive **Streamlit web application** is included for easy image upload and prediction.

---

## 📌 Tumor Categories
The model classifies MRI images into 4 categories:

- Glioma Tumor
- Meningioma Tumor
- Pituitary Tumor
- No Tumor

---

## 🚀 Project Features
✅ Multi-class MRI tumor classification  
✅ Transfer Learning using **MobileNetV2 (ImageNet weights)**  
✅ Fine-tuning + class-weighted training for better performance  
✅ Displays predicted class + confidence score  
✅ Streamlit Web App for real-time prediction  
✅ Confusion Matrix + Classification Report evaluation  

---

## 🏗️ Project Workflow
1. Dataset Loading (Train / Valid / Test)
2. Image Preprocessing (Resize to 224×224)
3. Data Augmentation (Flip, Rotation, Zoom, Contrast)
4. Model Training (MobileNetV2 Transfer Learning)
5. Fine-tuning for improved accuracy
6. Model Evaluation (Accuracy, Confusion Matrix, Classification Report)
7. Streamlit Deployment

---

## 📊 Model Performance
Final Model: **MobileNetV2 (Transfer Learning + Fine-tuning + Class Weights)**

✅ Test Accuracy: **~84%**  
✅ Improved recall for difficult classes like **Meningioma**

---

## 🖥️ Streamlit App Demo
The Streamlit app allows users to:
- Upload an MRI Image (`.jpg`, `.jpeg`, `.png`)
- Get predicted tumor class
- View confidence score (%) for each class

---

##  Live Demo
http://13.49.44.149:8501

## 📂 Project Structure
```bash
Brain_Tumor_App/
│── app.py
│── brain_tumor_weights.weights.h5        # Trained model weights
│── class_names.json                      # Class labels
│── requirements.txt
│── README.md



