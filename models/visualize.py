import os
import numpy as np
from PIL import Image
import wandb

class Visualizer:
    def __init__(self,result_folder,visuals) -> None:
        self.result_folder = result_folder
        self.visuals = visuals
        train_path = os.path.join(result_folder,'train')
        test_path = os.path.join(result_folder,'test')
        os.makedirs(train_path,exist_ok=True)
        os.makedirs(test_path,exist_ok=True)
        for name in visuals:
            path = os.path.join(train_path,name)
            os.makedirs(path,exist_ok=True)
            path = os.path.join(test_path,name)
            os.makedirs(path,exist_ok=True)
    
    
    def tensor2pil(self,x):
        # converts tensor in range [-1,1] to a pil image
        x = x.squeeze(0)
        x = x.cpu().numpy()
        x = np.transpose(x, (1,2,0))
        x = (x + 1)*127.5
        x = x.astype('uint8')
        return Image.fromarray(x)


    def save_visuals(self,imgs,epoch,logger,phase):
        ims_dict = {}
        results_path = os.path.join(self.result_folder,phase)        
        for name,image in imgs.items():
            a = self.tensor2pil(image)
            visual_path = os.path.join(results_path,name)
            path = os.path.join(visual_path,f'{epoch}.png')
            a.save(path)
            ims_dict[name] = wandb.Image(a)
        # log image to wandb
        logger.log_image(key='summary',images=list(ims_dict.values()),caption=list(ims_dict.keys()))
  