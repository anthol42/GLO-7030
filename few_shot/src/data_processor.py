import pandas as pd
from kagglehub import KaggleDatasetAdapter
import kagglehub

class RudditDataProcessor:
    """
    A class to handle loading, cleaning, and saving of Ruddit comments data from Kaggle.
    
    Attributes:
        df (pd.DataFrame): Main dataframe storing the comments and scores
        original_columns (list): List of original columns in the dataset
        seed (int): Random seed for reproducibility
    """
    
    def __init__(self, seed=None):
        """
        Initialize the data processor with optional seed for reproducibility.
        
        Args:
            seed (int, optional): Random seed for reproducible operations
        """
        self.df = None
        self.original_columns = None
        self.seed = seed

    def load_from_kaggle(self):
        """
        Load the dataset from Kaggle using the kagglehub library.
        
        Raises:
            Exception: If loading fails
        """
        try:
            file_path = "ruddit_comments_score.csv"
            self.df = kagglehub.load_dataset(
                KaggleDatasetAdapter.PANDAS,
                "estebanmarcelloni/ruddit-papers-comments-scored",
                file_path,
            )
            self.original_columns = self.df.columns.tolist()
            print("Dataset loaded successfully. Shape:", self.df.shape)
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset: {str(e)}")

    def clean_data(self):
        """
        Clean the loaded data by:
        1. Keeping only relevant columns (body and score)
        2. Removing deleted/removed comments
        3. Removing missing values
        4. Removing duplicate comments
        5. Trimming whitespace
        6. Removing multiline skips
        """
        if self.df is None:
            raise ValueError("No data to clean. Load data first using load_from_kaggle()")
            
        self.df = self.df[['body', 'score']].copy()
        
        self.df = self.df[~self.df['body'].str.contains(r'\[deleted\]|\[removed\]', na=False)]
        
        self.df = self.df.dropna()
        
        self.df = self.df.drop_duplicates(subset=['body'])
        
        self.df['body'] = self.df['body'].str.strip()
        self.df['body'] = self.df['body'].str.replace(r'\n+', ' ', regex=True)
        
        self.df = self.df.reset_index(drop=True)
        print("Data cleaned. New shape:", self.df.shape)

    def save_to_csv(self, filename, subset=None, index=False, df=None):
        """
        Save the dataframe to a CSV file.
        
        Args:
            filename (str): Path to save the CSV file
            subset (list, optional): List of columns to save. If None, saves all columns.
            index (bool): Whether to save the DataFrame index. Default is False.
            df (pd.DataFrame, optional): DataFrame to save. If None, uses the current dataframe.
            
        Raises:
            ValueError: If no data exists
        """
        if self.df is None:
            raise ValueError("No data to save. Load data first")
            
        df_to_save = self.df if df is None else df
        if subset is not None:
            if not all(col in self.df.columns for col in subset):
                raise ValueError("One or more columns in subset do not exist in the dataframe")
            df_to_save = self.df[subset]
            
        df_to_save.to_csv(filename, index=index)
        print(f"Data saved to {filename}")
        
    def pprint(self, max_rows=5, max_body_length=150):
        """
        Pretty-print the dataframe with controlled formatting.
        
        Args:
            max_rows (int): Maximum number of rows to display
            max_body_length (int): Maximum length for body text preview
        """
        if self.df is None:
            raise ValueError("No data to display. Load data first using load_from_kaggle()")
        
        display_df = self.df.copy()
        
        display_df['body'] = display_df['body'].str.slice(0, max_body_length) + '...'
        
        with pd.option_context(
            'display.max_columns', 2,
            'display.width', 120,
            'display.colheader_justify', 'left'
        ):
            print(f"\nCurrent DataFrame (First {max_rows} of {len(self.df)} rows):")
            print(display_df.head(max_rows).to_string(index=False, max_colwidth=max_body_length+3))
