from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
from src.model_trainer import ModelTrainer
from src.evaluator import Evaluator


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def main():
    # Load Data
    path = r"dataset\heart.csv"
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



    #Count duplicates
    print('Number of duplicate data :', df.duplicated().sum())

    # #Remove duplicates
    # df = df.drop_duplicates()


    #==================================== Further EDA & Visualization ====================================
    #Target Variable Analysis (Data distribution)    
    df['HeartDisease'].value_counts()
    sns.countplot(x='HeartDisease', data=df)

    #Univariate Analysis
    df.hist(figsize=(12,8))
    plt.show()

    #Outlier Detection
    numerical_columns = ['Age','RestingBP','Cholesterol','MaxHR']
    fig, axes = plt.subplots(2, 2, figsize=(12,4))
    axes = axes.flatten()
    
    for i, col in enumerate(numerical_columns):
        sns.boxplot(x=df[col], ax=axes[i])
        axes[i].set_title(col)
    
    fig.tight_layout()
    plt.show()

    
    #Correlation Analysis
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, cmap='coolwarm')
    plt.title("Correlation Analysis")
    plt.show()


    # Filter data (menghapus data 'Cholesterol' bernilai 0)
    df = df[df['Cholesterol'] > 0]




    #=================== PREPROCESSING (Selecting Target&feature variables + Splitting) ===================
    preprocessor = Preprocessor(df)
    #Selecting the target variable (Y)
    y ='HeartDisease'
    
    #Selecting the feature variables (X)
    x_single =['Age']

    x_str =['Sex','ChestPainType','RestingECG','ExerciseAngina','ST_Slope']
    x_int = ['Age', 'Cholesterol','FastingBS', 'MaxHR', 'Oldpeak']
    x_multi = x_str + x_int

    #split data + convert nominal data ('city','statezip') to One Hot Encoding
    data_multi = preprocessor.preprocess_with_dummies(y,x_multi,x_str)
    x_multi_train, x_multi_test, y_multi_train, y_multi_test = data_multi

    

    #=========================================== MODEL TRAINING ============================================
    trainer = ModelTrainer()
    # models_single = trainer.train_classification(
    #     X_train,
    #     y_train
    # )
    models_multiple = trainer.train_classification(
        x_multi_train,
        y_multi_train
    )


    #============================================== Evaluation ==============================================
    evaluator = Evaluator()
    # results_single = evaluator.evaluate(
    #     models_single,
    #     X_test,
    #     y_test,
    #     "Single ('sqft_living')"

    # )
    
    results_multiple = evaluator.evaluate_classification(
        models_multiple,
        x_multi_test,
        y_multi_test,
    )

    for x, y in results_multiple.items():
        print(x,"- Multiple")
        print(f"accuracy_score         : {y['accuracy_score']:,.2f}")
        print(f"classification_report  : \n{y['classification_report']}")
        print(f"confusion_matrix       : \n{y['confusion_matrix']}\n")
      


if __name__ == "__main__":
    main()
