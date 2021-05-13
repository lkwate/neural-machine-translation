from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from utils import *
import torch 


class LangDataset(Dataset):
    
    def __init__(self, args):
        self.args = args
        self.input_lang, self.output_lang, self.pairs = prepareData('eng', 'fra', True)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        pair = self.pairs[idx]
        (input_ids, output_ids) = tensorsFromPair(self.input_lang, self.output_lang, pair)
        return (input_ids, output_ids)


class LangDataModule(pl.LightningDataModule):

    def __init__(self, args):
        self.args = args
        
    def prepare_data(self):
        self.dataset = LangDataset(self.args)

    def setup(self, stage):
        train_len = int(len(self.dataset) * self.args.train_val_split_factor)
        val_len = len(self.dataset) - train_len
        self.train_dataset, self.val_dataset = random_split(self.dataset, [train_len, val_len])

    def train_dataloader(self):
        dataloader = DataLoader(self.train_dataset, batch_size = self.args.batch_size)
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(self.val_dataset, batch_size=self.args.batch_size)
        return dataloader
