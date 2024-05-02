from typing import Any, Mapping
import torch
from torch import nn
from torch.nn import init
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import functional as F
from torchsummary import summary
import itertools
import lightning as L
from .networks import get_patchgan_model, get_resnet_generator
from .utils import ImagePool, init_weights, set_requires_grad
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from .visualize import Visualizer

class CycleGan(L.LightningModule):
    def __init__(self,cfg):
        super().__init__()
        self.automatic_optimization=False
        # generator pair
        self.genX = get_resnet_generator()
        self.genY = get_resnet_generator()
        
        # discriminator pair
        self.disX = get_patchgan_model()
        self.disY = get_patchgan_model()
        
        self.lm = 10.0
        self.fakePoolA = ImagePool()
        self.fakePoolB = ImagePool()
        self.genLoss = None
        self.disLoss = None
        
        self.psnr = PeakSignalNoiseRatio()
        self.ssim = StructuralSimilarityIndexMeasure()
        
        
        self.visualizer = Visualizer(result_folder=cfg.folders.results,
                                     visuals=['real_A','fake_A','real_B','fake_B'],
                                     )

        self.psnr = PeakSignalNoiseRatio()
        self.ssim = StructuralSimilarityIndexMeasure()

        for m in [self.genX, self.genY, self.disX, self.disY]:
            init_weights(m)
    
    def configure_optimizers(self):
        optG = Adam(
            itertools.chain(self.genX.parameters(), self.genY.parameters()),
            lr=2e-4, betas=(0.5, 0.999))
        
        optD = Adam(
            itertools.chain(self.disX.parameters(), self.disY.parameters()),
            lr=2e-4, betas=(0.5, 0.999))
        
        gamma = lambda epoch: 1 - max(0, epoch + 1 - 100) / 101
        schG = LambdaLR(optG, lr_lambda=gamma)
        schD = LambdaLR(optD, lr_lambda=gamma)
        return [optG, optD], [schG, schD]

    def get_mse_loss(self, predictions, label):
        """
            According to the CycleGan paper, label for
            real is one and fake is zero.
        """
        if label.lower() == 'real':
            target = torch.ones_like(predictions)
        else:
            target = torch.zeros_like(predictions)
        
        return F.mse_loss(predictions, target)
            
    def generator_training_step(self, imgA, imgB):        
        """cycle images - using only generator nets"""
        fakeB = self.genX(imgA)
        cycledA = self.genY(fakeB)
        
        fakeA = self.genY(imgB)
        cycledB = self.genX(fakeA)
        
        sameB = self.genX(imgB)
        sameA = self.genY(imgA)
        
        # generator genX must fool discrim disY so label is real = 1
        predFakeB = self.disY(fakeB)
        mseGenB = self.get_mse_loss(predFakeB, 'real')
        
        # generator genY must fool discrim disX so label is real
        predFakeA = self.disX(fakeA)
        mseGenA = self.get_mse_loss(predFakeA, 'real')
        
        # compute extra losses
        identityLoss = F.l1_loss(sameA, imgA) + F.l1_loss(sameB, imgB)
        
        # compute cycleLosses
        cycleLoss = F.l1_loss(cycledA, imgA) + F.l1_loss(cycledB, imgB)
        
        # gather all losses
        extraLoss = cycleLoss + 0.5 * identityLoss
        self.genLoss = mseGenA + mseGenB + self.lm * extraLoss
        
        
        # store detached generated images
        self.fakeA = fakeA.detach()
        self.fakeB = fakeB.detach()
        
        return self.genLoss
    
    def discriminator_training_step(self, imgA, imgB):
        """Update Discriminator"""        
        fakeA = self.fakePoolA.query(self.fakeA)
        fakeB = self.fakePoolB.query(self.fakeB)
        
        # disX checks for domain A photos
        predRealA = self.disX(imgA)
        mseRealA = self.get_mse_loss(predRealA, 'real')
        
        predFakeA = self.disX(fakeA)
        mseFakeA = self.get_mse_loss(predFakeA, 'fake')
        
        # disY checks for domain B photos
        predRealB = self.disY(imgB)
        mseRealB = self.get_mse_loss(predRealB, 'real')
        
        predFakeB = self.disY(fakeB)
        mseFakeB = self.get_mse_loss(predFakeA, 'fake')
        
        # gather all losses
        self.disLoss = 0.5 * (mseFakeA + mseRealA + mseFakeB + mseRealB)
        return self.disLoss
    
    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()
        
        self.imgA, self.imgB = batch['A'], batch['B']
        #discriminator_requires_grad = (optimizer_idx==1)
        set_requires_grad([self.disX, self.disY], True)
        
        self.toggle_optimizer(opt_g.optimizer)
        self.generator_training_step(self.imgA, self.imgB)
        opt_g.optimizer.zero_grad()
        self.manual_backward(self.genLoss)
        opt_g.step()
        self.untoggle_optimizer(opt_g.optimizer)
        
        
        self.toggle_optimizer(opt_d.optimizer)
        self.discriminator_training_step(self.imgA, self.imgB)        
        opt_d.optimizer.zero_grad()
        self.manual_backward(self.disLoss)
        opt_d.optimizer.step()
        self.untoggle_optimizer(opt_d.optimizer)
        psnr_ab_score = self.psnr(self.fakeB,self.imgB) 
        psnr_ba_score = self.psnr(self.fakeA,self.imgA)
        
        scores = {
            'psnr_ab':psnr_ab_score,
            'psnr_ba':psnr_ba_score,
            'train/dis_loss' : self.disLoss.item(),
            'train/gen_loss' : self.genLoss.item(),
        }
        self.log_dict(scores, on_step=True, on_epoch=True, prog_bar=True, logger=True,sync_dist=True,batch_size=1)
        
    def test_step(self, batch) -> torch.Tensor | Mapping[str, Any] | None:
        print('start test')
        imgA, imgB = batch['A'], batch['B']
        
        fakeB = self.genY(imgA)
        fakeA = self.genX(imgB)
        print('generate fake images')
        
        psnr_ab_score = self.psnr(fakeB,imgB)
        psnr_ba_score = self.psnr(fakeA,imgA)
        
        self.log('psnr_ab',psnr_ab_score,batch_size=1)
        self.log('psnr_ba',psnr_ba_score,batch_size=1)
        
        #self.log_dict({
        #    'psnr_ab' : psnr_ab_score,
        #    'psnr_ba' : psnr_ba_score,
        #    
        #})
    
        #self.logger.log_image(key="visuals", images=[fakeA, fakeB], caption=["fakeA", "fakeB"])
        
    def on_train_epoch_end(self):
        visuals = {
            'real_A' : self.imgA,
            'fake_A' : self.fakeA,
            'real_B' : self.imgB,
            'fake_B' : self.fakeB,
        }
        self.visualizer.save_visuals(visuals,self.current_epoch,self.logger)
        
        
        
        
        

if __name__ == '__main__':
    model = CycleGan()