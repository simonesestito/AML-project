from torch.utils.data import Dataset
import pandas as pd
import torch
import os


class StatementDataset(Dataset):
    """
    PyTorch Dataset for statements and their truth values.
    """

    def __init__(self, dataframe: pd.DataFrame):
        """
        Args:
            dataframe (pd.DataFrame): The combined dataset from all CSV files.
                                      Expects columns ['statement', 'label', 'topic'].
        """
        self.data = dataframe


    @staticmethod
    def load_from_directory(drive_path: str) -> 'StatementDataset':
        """
        Create a StatementDataset from CSV files in a specified Google Drive folder,
        adding a 'topic' column to indicate the source file of each row.

        Args:
            drive_path (str): Path to the folder containing the CSV files.

        Returns:
            StatementDataset: PyTorch Dataset for the combined dataset.
        """
        # Ensure the path exists
        if not os.path.exists(drive_path):
            raise ValueError(f"Path '{drive_path}' does not exist.")

        all_dataframes = []
        for file_name in os.listdir(drive_path):
            file_path = os.path.join(drive_path, file_name)
            if file_name.endswith(".csv"):
                print(f"Loading file: {file_name}")
                # Read the CSV and add a 'topic' column with the file name (without extension)
                df = pd.read_csv(file_path)
                df['topic'] = os.path.splitext(file_name)[0]
                all_dataframes.append(df)

        if not all_dataframes:
            raise ValueError(f'No CSV files found in the directory "{drive_path}".')
    
        combined_dataset = pd.concat(all_dataframes, ignore_index=True)
        return StatementDataset(combined_dataset)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[str, torch.Tensor, str]:
        """
        Args:
            idx (int): Index of the row to retrieve.

        Returns:
            tuple: (statement, label, topic), where statement is the text,
                   label is the binary target, and topic is the source file name.
        """
        row = self.data.iloc[idx]
        statement = row['statement']
        label = torch.tensor(row['label'])
        topic = row['topic']
        return statement, label, topic
    

    def split_by_topic(self, topic: str) -> tuple['StatementDataset', 'StatementDataset']:
        """
        Split the dataset into two parts based on the 'topic' column.

        Args:
            topic (str): The value in the 'topic' column to split on.

        Returns:
            tuple: Two StatementDatasets, one for all but the specified topic (TRAIN+VAL), and one for the specified topic (TEST).
        """
        mask = self.data['topic'] == topic
        topic_data = self.data[mask]
        other_data = self.data[~mask]
        return StatementDataset(other_data), StatementDataset(topic_data)

