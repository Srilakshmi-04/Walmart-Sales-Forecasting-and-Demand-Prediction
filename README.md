# Walmart Sales Forecasting and Demand Prediction

**A Time-Series Forecasting Project Using Machine Learning, Deep Learning and Statistical Models**

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Notebook-Google%20Colab-orange?logo=googlecolab)](https://colab.research.google.com/)
[![Machine Learning](https://img.shields.io/badge/Domain-Machine%20Learning-green)]()
[![Status](https://img.shields.io/badge/Status-Completed-success)]()

## Overview

Accurate demand forecasting is essential for retail businesses to maintain optimal inventory levels, reduce operational costs and improve customer satisfaction.

This project focuses on analysing historical Walmart sales data and predicting weekly sales at the department level. It explores different forecasting techniques and compares five models to identify the most suitable approach for retail demand prediction.

The project combines statistical time-series forecasting, traditional machine learning and deep learning techniques, evaluating their performance using MAE, RMSE and Weighted Mean Absolute Error (WMAE).

**Key Result:** SARIMA achieved the lowest WMAE of **29,130.27**, while Random Forest achieved the lowest RMSE of **68,866.77** on the common validation dataset.

## Project Objectives

- Analyse historical Walmart sales trends and patterns.
- Explore the relationship between sales, holidays and economic factors.
- Perform data cleaning, preprocessing and feature engineering.
- Implement five different forecasting models.
- Evaluate model performance using multiple error metrics.
- Identify the best-performing model based on WMAE.
- Understand how demand forecasting supports inventory planning.

## Dataset

**Dataset Source:** [Walmart Recruiting – Store Sales Forecasting (Kaggle)](https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting)

The project uses three datasets:

| Dataset | Description |
|---|---|
| `train.csv` | Historical weekly sales information |
| `stores.csv` | Store information, including type and size |
| `features.csv` | External factors such as temperature, fuel price, CPI, unemployment and markdowns |

### Dataset Information

| Metric | Value |
|---|---:|
| Total sales records | 421,570 |
| Number of stores | 45 |
| Number of departments | 81 |
| Historical weeks | 143 |
| Department-week records | 11,090 |
| Historical period | February 2010 – October 2012 |

**Target Variable:** `Weekly_Sales`

Sales were aggregated across stores at the department-week level for forecasting.

## Technology Stack

| Category | Tools |
|---|---|
| Programming Language | Python |
| Development Environment | Google Colab |
| Data Manipulation | Pandas, NumPy |
| Data Visualization | Matplotlib |
| Machine Learning | Scikit-learn |
| Gradient Boosting | XGBoost |
| Deep Learning | PyTorch |
| Statistical Forecasting | Statsmodels |

## Methodology

The project follows an end-to-end forecasting workflow:

1. **Data Collection:** Load the Walmart sales, stores and features datasets.
2. **Data Preprocessing:** Merge datasets, handle missing values and check data quality.
3. **Exploratory Data Analysis:** Analyse weekly sales trends, holiday effects, store types and department performance.
4. **Feature Engineering:** Create calendar, lag and rolling-average features.
5. **Data Splitting:** Divide the dataset chronologically into 80% training and 20% validation.
6. **Model Development:** Implement five forecasting models.
7. **Model Evaluation:** Compare model predictions using MAE, RMSE and WMAE.
8. **Result Analysis:** Interpret forecasting errors and identify the best-performing model.

## Feature Engineering

A total of 17 features were used in the modelling workflow.

**Base Features**
- Department
- Temperature
- Fuel Price
- CPI
- Unemployment
- Store Size
- Year
- Month
- Week
- Holiday Indicator

**Lag Features**
- Lag 1
- Lag 2
- Lag 4
- Lag 13
- Lag 52

**Rolling Features**
- 4-week Rolling Mean
- 8-week Rolling Mean

Rolling features were shifted by one week to prevent target leakage and ensure that only historical information was used.

## Forecasting Models

Five different forecasting approaches were implemented.

| Model | Approach |
|---|---|
| Seasonal Naive | Uses sales from the corresponding week 52 weeks earlier |
| Random Forest | Ensemble learning using multiple decision trees |
| XGBoost | Sequential gradient boosting using decision trees |
| LSTM | Deep learning model for sequential pattern learning |
| SARIMA | Statistical time-series forecasting with seasonal components |

### Model Configurations

| Model | Configuration |
|---|---|
| Random Forest | 150 estimators, max depth 18 |
| XGBoost | 400 estimators, learning rate 0.04, max depth 7 |
| LSTM | 8-week input sequence, 48 hidden units, 8 epochs |
| SARIMA | Order (1,1,1), seasonal order (1,0,0,52) |

SARIMA models were fitted separately for each department.

## Validation Strategy

An 80:20 chronological split was used to preserve the time-series order.

| Dataset | Weeks | Records |
|---|---:|---:|
| Training | 114 | 8,831 |
| Validation | 29 | 2,259 |

The validation process followed a rolling one-week-ahead approach, where earlier actual validation observations were used as historical information for subsequent predictions.

For a consistent comparison, all five models were evaluated on the same **2,240 common department-week observations**.

## Evaluation Metrics

Three evaluation metrics were used:

**1. Mean Absolute Error (MAE)**

Measures the average absolute difference between actual and predicted sales.

**2. Root Mean Squared Error (RMSE)**

Penalises larger prediction errors more strongly by squaring the errors.

**3. Weighted Mean Absolute Error (WMAE)**

Measures weighted prediction errors, giving different importance to different observations.

The project follows the original Walmart competition's weighting convention:

- Holiday weeks: Weight = 5
- Normal weeks: Weight = 1

WMAE was selected as the primary evaluation metric.

## Results and Model Comparison

The following results were obtained on the common validation dataset:

| Model | MAE | RMSE | WMAE |
|---|---:|---:|---:|
| **SARIMA** | **28,633.07** | 80,286.18 | **29,130.27** |
| Random Forest | 30,530.48 | **68,866.77** | 30,271.42 |
| Seasonal Naive | 33,117.95 | 75,744.70 | 32,519.00 |
| XGBoost | 44,241.06 | 79,536.31 | 43,432.78 |
| LSTM | 62,185.64 | 99,371.74 | 63,027.73 |

### Key Findings

- SARIMA achieved the lowest WMAE of **29,130.27**, making it the best model according to our primary evaluation metric.
- Random Forest achieved the lowest RMSE of **68,866.77**.
- Seasonal Naive provided a baseline for comparing advanced models.
- XGBoost performed less effectively than SARIMA and Random Forest in this experiment.
- LSTM recorded the highest WMAE in this implementation, indicating scope for further tuning.

### Best Performing Model

**SARIMA – Selected based on the lowest WMAE.**

The results demonstrate that statistical time-series models can remain competitive with machine learning and deep learning approaches for retail sales forecasting.

## Business Applications

The forecasting approach can support retail businesses in:

- Planning inventory based on estimated demand.
- Reducing overstocking and understocking risks.
- Identifying seasonal and holiday sales patterns.
- Improving purchasing and replenishment decisions.
- Supporting data-driven inventory management.

Forecasted sales are demand estimates and should be combined with current inventory, supplier lead times and safety-stock requirements before determining actual order quantities.

## Limitations

- Historical data covers February 2010 to October 2012.
- Forecasting was performed at department-week level across stores, not individual store-department level.
- Evaluation was conducted using a single chronological validation window.
- LSTM training was limited to eight epochs without extensive hyperparameter tuning.
- Additional multi-week forecasting and rolling-origin backtesting are needed to assess longer-term forecasting performance.

## Future Enhancements

- Implement store-department-level forecasting.
- Perform hyperparameter optimization.
- Improve LSTM training and architecture.
- Conduct rolling-origin validation.
- Evaluate multi-week forecasting performance.
- Integrate predictions with inventory planning systems.

## Repository Structure

```text
Walmart-Sales-Forecasting/
│
├── Walmart_Sales_Forecasting_Complete_Guide.ipynb
├── README.md
├── requirements.txt
│
├── results/
│   ├── model_comparison.csv
│   └── visualizations/
│
└── docs/
    └── project_report.pdf
```

*This is a suggested structure. Keep only files and folders that are actually included in the repository.*

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/walmart-sales-forecasting.git
```

### 2. Navigate to the project directory

```bash
cd walmart-sales-forecasting
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download the Walmart Recruiting Store Sales Forecasting dataset from Kaggle and provide the required CSV files to the notebook.

### 5. Run the notebook

Open `Walmart_Sales_Forecasting_Complete_Guide.ipynb` using Google Colab or Jupyter Notebook and execute the cells in sequence.

## References

- [Walmart Recruiting – Store Sales Forecasting](https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [NumPy Documentation](https://numpy.org/doc/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Statsmodels Documentation](https://www.statsmodels.org/)

## Contributors

| Team Member | Responsibility |
|---|---|
| Sanjana | Dataset and Preprocessing |
| Praveen R | Exploratory Data Analysis |
| **Srilakshmi** | **Model Implementation** |
| Vismaya H M | Evaluation and Results |

**Institution:** RV University  
**Programme:** B.Tech (Hons.) Computer Science and Engineering  
**Major:** Data Science and Engineering  
**Semester:** 5th Semester

---

⭐ If you find this project useful, feel free to star the repository!
