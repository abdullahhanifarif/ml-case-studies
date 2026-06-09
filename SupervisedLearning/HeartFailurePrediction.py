from src.data_loader import DataLoader


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

    #Check data price
    print("Rows with price <= 0:", (df["price"] <= 0).sum())

    #Count duplicates
    print('Number of duplicate data :', df.duplicated().sum())

    # #Remove duplicates
    # df = df.drop_duplicates()


    



if __name__ == "__main__":
    main()
