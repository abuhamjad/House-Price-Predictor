# House Price Predictor

A Machine Learning project that predicts residential property prices using **Linear Regression** and property-related features such as area, bedrooms, bathrooms, furnishing status, building type, and location.

---

## Project Overview

This project uses a supervised machine learning approach to estimate house prices based on property characteristics. The model is trained on a housing dataset containing information about residential properties in Delhi NCR regions including Delhi, Noida, Greater Noida, Gurgaon, Ghaziabad, and Faridabad.

The project demonstrates:

* Data preprocessing
* Feature engineering
* Categorical encoding
* Model training and evaluation
* Cross-validation
* Real-time price prediction through a command-line interface

---

## Features

* Predict house prices based on:

  * Area (sq ft)
  * Number of Bedrooms
  * Number of Bathrooms
  * Balcony Count
  * Parking Spaces
  * Lift Availability
  * City
  * Property Status
  * Property Age
  * Furnishing Status
  * Building Type

* Automated preprocessing using:

  * Missing value imputation
  * One-Hot Encoding
  * Feature scaling

* Feature engineering:

  * Total Rooms
  * Location categorization

* Model persistence using Pickle

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Pickle

---

## Project Structure

```text
House-Price-Predictor/
│
├── data/
│   └── dataset.csv
│
├── screenshots/
│   └── model-metrics.png
│   └──prediction-demo.png
│
├── src/
│   └── app.py
│
├── house_price_model.pkl
│
├── requirements.txt
│
├── README.md
│
└── .gitignore
```

---

## Machine Learning Pipeline

### Data Preprocessing

* Removed irrelevant columns
* Handled missing values using median and most-frequent imputation
* Converted categorical features using One-Hot Encoding
* Scaled numerical features using StandardScaler

### Feature Engineering

Created additional features such as:

* Total_Rooms = Bedrooms + Bathrooms

Location information was categorized into broader city regions:

* Delhi
* Noida
* Greater Noida
* Gurgaon
* Ghaziabad
* Faridabad

---

## Model Performance

### Evaluation Metrics

| Metric   | Value                    |
| -------- | ------------------------ |
| R² Score | 0.82+                    |
| MAE      | Generated during runtime |
| RMSE     | Generated during runtime |

### Cross Validation

5-Fold Cross Validation was performed to evaluate model stability and generalization.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/House-Price-Predictor.git

cd House-Price-Predictor
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

```bash
python src/app.py
```

---
## 📸 Screenshots

### Model Performance

![Model Performance](screenshots/model_metrics.png)

### Prediction Demo

![Prediction Demo](screenshots/prediction_demo.png)

## 💾 Saving the Trained Model

The trained model is automatically saved using Pickle:

```python
with open("house_price_model.pkl", "wb") as f:
    pickle.dump(model, f)
```

This allows future predictions without retraining the model.

---

## Author

Abu Huraira

Python Developer | Machine Learning Enthusiast

GitHub: https://github.com/abuhamjad
LinkedIn: https://www.linkedin.com/in/abuhamjad/

```
```
