import torch
from models.cycle_gan import CycleGan
from data.unpaired_dataset import UnpairedDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

def main():
    expirement_name = 'example'

    data = UnpairedDataset('datasets/edges2shoes', 'train')
    data = DataLoader(data, batch_size=1, shuffle=True)
    model = CycleGan()

    #device = 'cuda' if torch.cuda.is_available() else 'cpu'

    wandb_logger = WandbLogger(project="CycleGAN",log_model=True)
    checkpoint_callback = ModelCheckpoint(dirpath=f'checkpoints/{expirement_name}',monitor='train/gen_loss')
    trainer = L.Trainer(accelerator='auto',max_epochs=3,logger=wandb_logger,callbacks=[checkpoint_callback],strategy='ddp_find_unused_parameters_true')

    trainer.fit(model, data)

if __name__ == '__main__':
    main()