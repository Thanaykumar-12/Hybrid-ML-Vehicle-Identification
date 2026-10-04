[![Python application](https://github.com/Thanaykumar-12/Hybrid-ML-Vehicle-Identification/actions/workflows/python-app.yml/badge.svg)](https://github.com/Thanaykumar-12/Hybrid-ML-Vehicle-Identification/actions/workflows/python-app.yml)
# 🚗 Hybrid ML Vehicle Identification

A hybrid machine learning system designed for fast and accurate vehicle identification in connected network environments. The project combines **Multilayer Perceptron (MLP)** and **Random Forest CART** techniques with data preprocessing, model evaluation, and a web-based interface.

---

## 📌 Project Overview

Vehicle identification and intelligent decision-making are important components of modern **Vehicular Ad Hoc Networks (VANETs)**.

This project proposes a hybrid machine learning approach, **Fusion Mind CART**, to improve vehicle identification and classification using machine learning techniques.

The system processes VANET-related data, prepares the required features, applies machine learning models, and provides predictions through a Flask-based web application.

---

## 🎯 Objectives

- Develop an intelligent vehicle identification system.
- Apply machine learning techniques to VANET data.
- Combine MLP and Random Forest CART approaches.
- Perform data preprocessing and feature preparation.
- Evaluate machine learning model performance.
- Provide an interactive web-based interface.
- Support intelligent decision-making in connected vehicle environments.

---

## 🧠 Proposed Approach

The proposed **Fusion Mind CART** approach combines:

### Multilayer Perceptron (MLP)

MLP is used to learn complex relationships between input features and vehicle-related outcomes.

### Random Forest CART

Random Forest based on Classification and Regression Trees is used for robust decision-making and classification.

### Hybrid Model

The combination of these approaches aims to provide a more reliable machine learning pipeline for vehicle identification in connected network environments.

```text

VANET Dataset
      ↓
Data Preprocessing
      ↓
Feature Preparation
      ↓
Machine Learning Models
      ↓
MLP + Random Forest CART
      ↓
Model Evaluation
      ↓
Vehicle Identification / Prediction
      ↓
Web Application

---

## 🛠️ Technologies Used

- Python
- Flask
- Machine Learning
- Multilayer Perceptron (MLP)
- Random Forest CART
- Pandas
- NumPy
- Scikit-learn
- HTML
- CSS
- JavaScript
- CSV Dataset

## 📂 Project Structure

```text
Hybrid-ML-Vehicle-Identification/
│
├── app.py
├── data_processor.py
├── ml_models.py
├── models/
├── static/
├── templates/
├── testdata.csv
├── vanet_routing_dataset.csv
└── sys.png

```
## 🚀 How to Run

### 1. Clone the Repository

`git clone https://github.com/Thanaykumar-12/Hybrid-ML-Vehicle-Identification.git`

`cd Hybrid-ML-Vehicle-Identification`

### 2. Install Dependencies

`pip install -r requirements.txt`

### 3. Run the Flask Application

`python app.py`

### 4. Open in Browser

Open:

`http://127.0.0.1:5000`

The application provides a web interface for dataset analysis, machine learning prediction, classification, and regression.


## Application Screenshots
## Model Performance

The proposed FusionMind approach was evaluated against multiple machine learning models for both classification and regression tasks.

### Classification Performance

The classification module predicts route optimality using multiple machine learning algorithms.

| Model | Accuracy |
|---|---:|
| SGD Classifier | ~81% |
| GP Classifier | ~78% |
| KNN Classifier | ~78% |
| **FusionMind Classifier** | **100%** |

**Best Classification Model:** FusionMind Classifier

### Regression Performance

The regression module predicts average vehicle spacing.

| Model | MAE | RMSE | R² Score |
|---|---:|---:|---:|
| SGD Regressor | 1.5335 | 1.8840 | 0.239 |
| GP Regressor | 3.6166 | 3.9289 | -2.310 |
| KNN Regressor | 1.8205 | 2.1190 | 0.037 |
| **FusionMind Regressor** | **0.0382** | **0.0637** | **0.999** |

**Best Regression Model:** FusionMind Regressor

> Note: The reported metrics are based on the project's evaluation dataset and should not be interpreted as real-world deployment performance.

### Prediction Input

The prediction module accepts vehicle, network, and road parameters to generate route and vehicle-spacing predictions.

![Prediction Input](screenshots/prediction-input.png)

### Prediction Output

The system displays classification results from multiple machine learning models and the predicted vehicle spacing.

![Prediction Output](screenshots/prediction-output.png)

---

---

## 👨‍💻 Author

**Thanu (Ande Thanay Kumar)**

- 💼 [LinkedIn](https://www.linkedin.com/in/a-thanay-3b5880293/)
- 🐙 [GitHub](https://github.com/Thanaykumar-12)
- 🌐 [Portfolio](https://thanay11.netlify.app/)

---

## 📄 License

This project is licensed under the MIT License.
