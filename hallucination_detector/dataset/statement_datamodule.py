from torch.utils.data import Dataset, DataLoader, random_split
import lightning as pl
from .statement_dataset import StatementDataset

class StatementDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, drive_path: str, validation_set_percentage: float = 0.2):
        super().__init__()
        self.batch_size = batch_size
        self.drive_path = drive_path
        self.validation_set_percentage = validation_set_percentage

        # Full dataset that contains all statements, both train and test
        self.full_dataset: StatementDataset = None

        # Subset for train + validation
        self.train_val_split: StatementDataset = None

        # Subset for test
        self.test_split: StatementDataset = None

        # Subset (random) for train
        self.train_dataset: Dataset = None

        # Subset (random) for validation
        self.val_dataset: Dataset = None


    def set_test_topic(self, test_topic: str):
        self.prepare_data()
        self.train_val_split, self.test_split = self.full_dataset.split_by_topic(test_topic)


    def prepare_data(self):
        if not self.full_dataset:
            # Load the dataset
            self.full_dataset = StatementDataset.load_from_directory(self.drive_path)


    def setup(self, stage=None):
        if not self.train_val_split:
            raise ValueError("Call set_test_topic() before calling setup()")

        if stage == "fit" or stage is None:
            # Create random train/val splits from the train_val subset
            val_size = int(self.validation_set_percentage * len(self.dataset))
            train_size = len(self.dataset) - val_size
            self.train_dataset, self.val_dataset = random_split(self.train_val_split, [train_size, val_size])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
    
    def test_dataloader(self) -> DataLoader:
        if not self.test_split:
            raise ValueError("Call set_test_topic() before calling test_dataloader()")
        return DataLoader(self.test_split, batch_size=self.batch_size)
