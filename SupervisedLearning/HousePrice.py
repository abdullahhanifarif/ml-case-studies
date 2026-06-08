from src.data_loader import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    #Load Data
    path = r"dataset\HousePrice.csv"
    loader = DataLoader(path)

    df = loader.load_data()
    print("Loaded dataset from:", path)

    
    #======================= Initial EDA =======================
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])

    #View the first few rows (default is 5 rows)
    print(df.head())

    #Check structure, data types, missing values
    print(df.info())

    #Summary statistics
    print(df.describe())

    

    #===================== DATA CLEANING ======================
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


    #============== Further EDA & Visualization ==============
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


    #================= Feature Engineering Insight =================
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
    
    

if __name__ == "__main__":
    main()
