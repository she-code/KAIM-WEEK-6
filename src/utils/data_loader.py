import pandas as pd
def load_data(file_path):
    """
    Load dataset from a CSV file.
    
    Parameters:
    file_path (str): Path to the CSV file.
    
    Returns:
    pd.DataFrame: Loaded dataset.
    """
    try:
        data = pd.read_csv(file_path)
        if "Unnamed: 0" in data.columns:
            data.drop(columns="Unnamed: 0", inplace=True)
        print(f"Data loaded successfully from {file_path}")
        return data
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None
    except pd.errors.EmptyDataError:
        print("Error: The file is empty.")
        return None
    except pd.errors.ParserError:
        print("Error: There was a parsing error while reading the file.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
