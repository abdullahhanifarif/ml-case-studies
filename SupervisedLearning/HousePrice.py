from src.data_loader import DataLoader


def main():
    # Load Data
    path = r"dataset\HousePrice.csv"
    loader = DataLoader(path)

    df = loader.load_data()

    print("Loaded dataset from:", path)
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])

if __name__ == "__main__":
    main()
