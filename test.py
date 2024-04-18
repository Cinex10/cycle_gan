import torch
from models.cycle_gan import CycleGan
from data.unpaired_dataset import UnpairedDataset
from torch.utils.data import DataLoader
from PIL import Image
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

def main():
    expirement_name = 'example'

    data = UnpairedDataset('datasets/edges2shoes', 'test')
    data = DataLoader(data, batch_size=1, shuffle=False)
    model = CycleGan()

    #device = 'cuda' if torch.cuda.is_available() else 'cpu'

    wandb_logger = WandbLogger(project="CycleGAN",log_model=True,name=expirement_name+'_test')
    trainer = L.Trainer(accelerator='auto',max_epochs=3,logger=wandb_logger,strategy='ddp_find_unused_parameters_true')

    trainer.test(model, data,ckpt_path='checkpoints/example/epoch=2-train/gen_loss=4.51.ckpt')
    
    

if __name__ == '__main__':
    main()