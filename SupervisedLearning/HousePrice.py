from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
from src.model_trainer import ModelTrainer
from src.evaluator import Evaluator
from src.visualization import Visualization

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def main():
    #Load Data
    path = r"dataset\HousePrice.csv"
    loader = DataLoader(path)

    df = loader.load_data()
    print("Loaded dataset from:", path)

    
    #============================================= Initial EDA =============================================
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])

    #View the first few rows (default is 5 rows)
    print(df.head())

    #Check structure, data types, missing values
    print(df.info())

    #Summary statistics
    print(df.describe())

    

    #=========================================== DATA CLEANING ============================================
    #Check missing values
    print("\n","Missing values:")
    print(df.isna().sum())
    print()

    #Check data price
    print("Rows with price <= 0:", (df["price"] <= 0).sum())

    #Count duplicates
    print('Number of duplicate data :', df.duplicated().sum())

    # #Remove duplicates
    # df = df.drop_duplicates()


    #==================================== Further EDA & Visualization ====================================
    #Target Variable Analysis (Data distribution, Skewness, Outlier)    
    fig, axes = plt.subplots(1, 2, figsize=(12,4))
    sns.histplot(df['price'], kde=True, ax=axes[0])
    axes[0].set_title('Price Distribution')

    #Outlier Detection
    sns.boxplot(y=df['price'], ax=axes[1])
    axes[1].set_title('Price Outliers')
    fig.suptitle("Exploratory Data Analysis (Target Variable Analysis)")
    fig.tight_layout()
    plt.show()

    #Filter data ($0 > price  < $2.000.000)
    df = df[(df['price'] > 0) & (df['price'] < 2000000)]

    
    #Correlation Analysis
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, cmap='coolwarm')
    plt.title("Correlation Analysis")
    plt.show()


    #===================================== Feature Engineering Insight =====================================
    #effective age of the house
    df['date'] = pd.to_datetime(df['date'])

    df['effective_year'] = np.where(
        df['yr_renovated'] > 0,
        df['yr_renovated'],
        df['yr_built']
    )

    df['effective_age'] = (
        df['date'].dt.year - df['effective_year']
    )

    #Building to land area ratio
    df['BuildingLotRatio'] = (df['sqft_living'] / df['sqft_lot'])



    #=================== PREPROCESSING (Selecting Target&feature variables + Splitting) ===================
    preprocessor = Preprocessor(df)
    #Selecting the target variable (Y)
    y ='price'
    
    #Selecting the feature variables (X)
    x =['sqft_living']
    x_multi = [
        'sqft_living',
        'sqft_above',
        'bathrooms',
        'bedrooms',
        'view',
        'floors',
        'effective_age',
        'BuildingLotRatio',
        'sqft_lot',
        'sqft_basement',
        'condition',
        'statezip',
        'city'
    ]

    #split data
    data = preprocessor.preprocess(y,x)
    x_train, x_test, y_train, y_test = data

    #split data + convert nominal data ('city','statezip') to One Hot Encoding
    data_multi = preprocessor.preprocess_with_dummies(y,x_multi,['city','statezip'])
    x_multi_train, x_multi_test, y_multi_train, y_multi_test = data_multi



    #=========================================== MODEL TRAINING ============================================
    trainer = ModelTrainer()

    models_single = trainer.train_regression(
        x_train,
        y_train
    )
    models_multiple = trainer.train_regression(
        x_multi_train,
        y_multi_train
    )




    #============================================== Evaluation ==============================================
    evaluator = Evaluator()

    results_single = evaluator.evaluate_regression(
        models_single,
        x_test,
        y_test,
    )
    for x, y in results_single.items():
        print(x,"- Single ('sqft_living')")
        print(f"MSE (Mean Squared Error)        : {y['MSE']:,.2f}")
        print(f"RMSE (Root Mean Squared Error)  : {y['RMSE']:,.2f}")
        print(f"MAE (Mean Absolute Error)       : ${y['MAE']:,.0f}")
        print(f"R²(Coefficient of Determination): {y['R2']:,.2f}\n")
    

    results_multiple = evaluator.evaluate_regression(
        models_multiple,
        x_multi_test,
        y_multi_test,
    )
    for x, y in results_multiple.items():
        print(x,"- Multiple")
        print(f"MSE (Mean Squared Error)        : {y['MSE']:,.2f}")
        print(f"RMSE (Root Mean Squared Error)  : {y['RMSE']:,.2f}")
        print(f"MAE (Mean Absolute Error)       : ${y['MAE']:,.0f}")
        print(f"R²(Coefficient of Determination): {y['R2']:,.2f}\n")



    #============================================== Visualization ==============================================
    viz = Visualization()

    # Compare model
    viz.compare_models(results_single, "Single ('sqft_living')")

    viz.compare_models(results_multiple, "Multiple")


    
    

if __name__ == "__main__":
    main()
