from src.data_loader import DataLoader
from src.model_trainer import ModelTrainer
from src.evaluator import Evaluator

from sklearn.preprocessing import StandardScaler

import seaborn as sns
import matplotlib.pyplot as plt


def main():
    path = r"dataset/Mall_Customers.csv"
    loader = DataLoader(path)
  
    df = loader.load_data()

    print("Loaded dataset from:", path)

    #Check number of rows and columns
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])
    print()

    #========================================== Initial EDA ==========================================
    # View the first few rows (default is 5 rows)
    print(df.head())

    # Check structure, data types, missing values
    print(df.info())

    # Summary statistics
    print(df.describe())


    #======================================== DATA CLEANING =========================================
    #Check missing values
    print("\n","Missing values:")
    print(df.isna().sum())
    print()

    #Count duplicates
    print('Number of duplicate data :', df.duplicated().sum())
    print()

    # # Remove duplicates
    # df = df.drop_duplicates()


    
    #================================= Further EDA & Visualization =================================
    #Check data Gender
    print(df['Gender'].value_counts())

    # Hapus ID
    df.drop(columns=["CustomerID"], inplace=True)

    # Encode gender
    df["Gender"] = df["Gender"].map({
        "Male": 0,
        "Female": 1
    })

    #Outlier Detection (Age, Annual Income (k$), Spending Score (1-100))
    fig, axes = plt.subplots(3, 2, figsize=(12,10))
    sns.boxplot(y=df['Age'], ax=axes[0,0])
    sns.histplot(df['Age'], kde=True, ax=axes[0,1])
    sns.boxplot(y=df['Annual Income (k$)'], ax=axes[1,0])
    sns.histplot(df['Annual Income (k$)'], kde=True, ax=axes[1,1])
    sns.boxplot(y=df['Spending Score (1-100)'], ax=axes[2,0])
    sns.histplot(df['Spending Score (1-100)'], kde=True, ax=axes[2,1])
    plt.show()

    #Filter data Annual Income (k$) < k$130
    df = df[df['Annual Income (k$)'] < 130]

    df.boxplot(column='Annual Income (k$)')
    plt.show()


    #===================================== Feature Selection (X) =====================================
    X = df[['Annual Income (k$)','Spending Score (1-100)']]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(x=X['Annual Income (k$)'], y=X['Spending Score (1-100)'])
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.show()

    #SCALING THE DATA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)


    #========================================= MODEL TRAINING ==========================================
    trainer = ModelTrainer()
    
    #ELBOW METHOD
    inertias = trainer.elbow_method(X_scaled)

    plt.plot(range(1, len(inertias)+1), inertias, marker='x')
    plt.xlabel('Number of Cluster')
    plt.ylabel('Inertia')
    plt.title("Elbow Method")
    plt.show()

    models = trainer.train_clustering(X_scaled, n=5)



    #============================================ Evaluation ============================================
    evaluator = Evaluator()
    results = evaluator.evaluate_models(
        X_scaled,
        models
    )

    for model_name, metrics in results.items():
        print(f"\n{model_name}  :")

        for metric, value in metrics.items():
            print(f"{metric}: {value}")

    




  

if __name__ == "__main__":
    main()
