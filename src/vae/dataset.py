import lightning as L
from torch.utils.data import DataLoader
from utils.dataset import SingleClassDataset
from utils.transform import create_transformation

class VAEDataModule(L.LightningDataModule):
    def __init__(self, train_dir, val_dir, batch_size, image_size, num_workers=4):
        super().__init__()
        self.save_hyperparameters()
        
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        transform = create_transformation(self.image_size)
        if stage == "fit" or stage is None:
            self.train_dataset = SingleClassDataset(self.train_dir, transform)
            self.val_dataset = SingleClassDataset(self.val_dir, transform) 

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers
        )