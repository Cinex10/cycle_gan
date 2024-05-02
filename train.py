import os
import torch
from models.cycle_gan import CycleGan
from data.unpaired_dataset import UnpairedDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import hydra

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):

    data = UnpairedDataset(cfg)
    data = DataLoader(data, 
                      batch_size=cfg.train.batch_size,
                      shuffle=True,
                      num_workers=cfg.train.num_workers,
                      )
    
    model = CycleGan(cfg)

    wandb_logger = WandbLogger(project=cfg.project.name,name=cfg.project.run)
    checkpoint_path = os.path.join(cfg.folders.checkpoint,cfg.project.run)
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_path,monitor='train/gen_loss')
    trainer = L.Trainer(accelerator=cfg.train.accelerator,
                        max_epochs=cfg.train.nb_epochs,
                        logger=wandb_logger,
                        callbacks=[checkpoint_callback],
                        strategy=cfg.train.strategy,
                        )

    trainer.fit(model, data)    

if __name__ == '__main__':
    main()