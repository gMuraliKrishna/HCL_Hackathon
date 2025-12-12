HCL Hackathon

This project implements an end-to-end data engineering and machine learning pipeline for the Retail Store Inventory Forecasting .
It includes database integration, data preprocessing, exploratory data analysis, feature engineering, model development, and result evaluation.

Dataset Overview

This project uses the Retail Store Inventory Forecasting Dataset, a synthetic but realistic dataset designed for time series forecasting, demand analysis, and inventory optimization tasks. The dataset contains approximately 73,100 daily records capturing sales and operational metrics across multiple retail stores and product groups. It is clean, complete, and well-structured, making it suitable for end-to-end forecasting model development.

Source: Kaggle Dataset

Project Overview

The goal of this project is to build a complete workflow starting from database connectivity to generating predictive insights.
The pipeline is implemented in Python (Jupyter Notebook) with MySQL as the data source.

 1. Data Integration Layer (MySQL → Jupyter Notebook)

    The project begins by establishing a connection between a local MySQL database and Jupyter Notebook to retrieve and work with the dataset.
    
    Retrieved data directly using SQL queries.
    
    Ensured smooth data flow for preprocessing and modeling.

 2. Data Cleaning & Preprocessing

    After importing the dataset, multiple preprocessing steps were performed:

    Steps: 

     Handled missing values.

     Corrected inconsistent date formats using pd.to_datetime.

     Removed duplicate entries.

     Standardized column names.

     Converted categorical variables into numerical form.

     Normalized or scaled numeric features where needed.

3. Exploratory Data Analysis (EDA)

   EDA was performed to understand patterns, trends, and anomalies in the dataset.

4. EDA Tasks

   Visualized distributions of major features.

   Identified correlations using heatmaps.

   Detected outliers.

   Time-series trend analysis.

   Seasonal/periodic pattern detection.

4. Feature Engineering

   To improve model performance, new features were created:
   
   Engineered Features
   
   Date-based features (Year, Month, Day, Week, Quarter).

   Lag features for time-series prediction.
   
   Moving averages and rolling statistics.
   
   Encoded categorical variables.
   
   Removed irrelevant or redundant columns.

5. Model Development

   Different models were trained to evaluate performance and select the best fit.

   Models Used: Random Forest Regressor , XGBoost Regressor, LSTM, XGBoost , SARIMA
  
   Time-series forecasting models
   
   Steps Used:

     Train-test split
     
     Model training
     
     Hyperparameter tuning
     
     Prediction generation

6. Model Evaluation

   Multiple metrics were calculated to determine accuracy and consistency.

   Metrics Used

      Mean Squared Error (MSE)
      
      Root Mean Squared Error (RMSE)
      
      Mean Absolute Error (MAE)

      MAPE
   
7. Forecasting / Final Output

The final model was used to generate predictions for future values based on test or unseen data.

Conclusion: 

This project demonstrates a complete, production-ready data analytics pipeline—starting from database integration and ending with meaningful machine learning predictions. It covers all essential data science steps:

✔ Data Extraction, Preprocessing, EDA, Feature Engineering, Model Building, Evaluation , Forecasting
