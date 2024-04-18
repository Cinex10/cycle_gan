import os
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from tqdm.notebook import tqdm

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(286, Image.BICUBIC),
        transforms.RandomCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([.5, .5, .5], [.5, .5, .5])
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.5, .5, .5], [.5, .5, .5])
    ]),
}

def get_dataroot_path(root):
    return os.path.join(Path(__file__).resolve().parent.parent,root)

class ImageFolder:
    def __init__(self, root, transforms):
        self.root = root
        self.paths = os.listdir(root)
        self.images = []
        for child in tqdm(self.paths):
            full_path = os.path.join(self.root, child)
            image = Image.open(full_path)
            self.images.append( transforms(image) )
            
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx): #doesnt support slices, we dont want them
        idx = idx % len(self)
        return os.path.join(self.root, self.paths[idx]), self.images[idx]
    

class UnpairedDataset(Dataset):
    def __init__(self, root, mode):
        """
        root must have trainA trainB testA testB as its subfolders
        mode must be either 'train' or 'test'
        """
        assert mode in 'train test'.split(), 'mode should be either train or test'
        
        super().__init__()
        self.transforms = data_transforms[mode]
        root = get_dataroot_path(root)
        pathA = os.path.join(root, mode+"A")
        self.dirA = ImageFolder(pathA, self.transforms)
        
        pathB = os.path.join(root, mode+"B")
        self.dirB = ImageFolder(pathB, self.transforms)
    
        

        print(f'Found {len(self.dirA)} images of {mode}A and {len(self.dirB)} images of {mode}B')
        
    
        
    def __len__(self):
        return max(len(self.dirA), len(self.dirB))
    
    def load_image(self, path):
        image = Image.open(path)
        if self.transforms:
            image = self.transforms(image)
        return path, image
    
    def __getitem__(self, idx): #doesnt support slices, we dont want them
        # we use serial batching
        pathA, imgA = self.dirA[idx]
        pathB, imgB = self.dirB[idx]
        return {
            'A': imgA, 'pathA': pathA,
            'B': imgB, 'pathB': pathB
        }