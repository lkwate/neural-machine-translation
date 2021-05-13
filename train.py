import torch
from utils import *
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from model.Model import *
from Dataset import *
import json
from collections import namedtuple

def train(args):
    nmt_translator = NMTTranslator(args)
    dm = LangDataModule(args)

    early_stop_callback = EarlyStopping(monitor='val_loss', patience=args.patience, mode='min', strict=False, verbose=True)
    trainer_args = {
        'gpus' : -1,
        'max_epochs' : args.max_epochs, 
        'val_check_interval':args.val_check_interval, 'callbacks' : [early_stop_callback]
    }
    trainer = pl.Trainer(**trainer_args)
    trainer.fit(nmt_translator, dm)


if __name__ == '__main__':

   args = open('config.json').read()
   args = json.loads(args)
   args = namedtuple("args", args.kesy())(*args.values())

   train(args)