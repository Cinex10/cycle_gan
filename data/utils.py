
from torchvision import transforms
from PIL import Image

def get_data_transform(cfg):
    size = (cfg.preprocess.input_width,cfg.preprocess.input_height)
    data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(size, Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([.5, .5, .5], [.5, .5, .5])
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.5, .5, .5], [.5, .5, .5])
    ]),
    }
    return data_transforms