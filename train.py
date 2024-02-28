import os
import random

import lightning as L
import wandb
from torch import Tensor, nn, optim, utils
from torchvision.transforms import ToTensor
from model import *
from data_loader import *
from lightning.pytorch.callbacks import ModelCheckpoint
from augmentation import *


def compute_psnr(img1: Tensor, img2: Tensor, saturation=1) -> float:
    mse = nn.functional.mse_loss(img1 * saturation, img2 * saturation)
    return 10 * torch.log10(saturation**2 / mse)


class LightningTrainer(L.LightningModule):
    def __init__(self, data_transform=None):
        super().__init__()
        self.model = UNetSeeInDark()
        self.data_transform = data_transform
        self.best_psnr = 0.0
        self.validation_psnrs = []

    def training_step(self, batch, batch_idx):
        if self.data_transform is not None:
            batch = self.data_transform(batch)
        inputs = batch['inputs']
        targets = batch['labels']

        out = self.model(inputs)
        loss = nn.functional.l1_loss(out, targets)
        wandb.log({"Train_loss": loss})
        self.log("Train_loss", loss)

        return loss

    def on_train_epoch_end(self):
        self.lr_schedulers().step()

    def validation_step(self, batch, batch_idx):
        inputs = batch['inputs']
        targets = batch['labels']

        out = self.model(inputs)
        psnr = compute_psnr(out, targets)
        self.validation_psnrs.append(psnr)
        return psnr

    def on_validation_epoch_end(self):
        avg_psnr = torch.stack(self.validation_psnrs).mean()
        wandb.log({"Val_psnr": avg_psnr})
        self.log("Val_psnr", avg_psnr)

        if avg_psnr > self.best_psnr:
            self.best_psnr = avg_psnr
            self.trainer.save_checkpoint('./checkpoints/sid_best.ckpt')
        self.validation_psnrs.clear()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        inputs = batch['inputs']
        y_hat = self.model(inputs)
        return y_hat

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=3e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.trainer.max_epochs, eta_min=1e-5)
        return [optimizer], [scheduler]


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    epoch = 500
    batch_size = 16

    data_transform = Compose([Random90Rotate(), RandomFlip()])
    denoise_traniner = LightningTrainer(data_transform)
    dataset = NoiseDataset('./data/sony_sid_torch/train/', duplicate=5)
    valset = NoiseDataset('./data/sony_sid_torch/val/', duplicate=10)
    train_loader = utils.data.DataLoader(dataset,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         num_workers=4,
                                         persistent_workers=True)
    val_loader = utils.data.DataLoader(valset,
                                       batch_size=batch_size,
                                       num_workers=4,
                                       shuffle=False,
                                       persistent_workers=True,
                                       drop_last=True)

    checkpoint_callback = ModelCheckpoint(dirpath='./checkpoints/',
                                          filename='sid_{epoch:02d}',
                                          every_n_epochs=10,
                                          save_top_k=-1)
    trainer = L.Trainer(max_epochs=epoch,
                        precision="bf16",
                        accelerator='gpu',
                        enable_checkpointing=True,
                        gradient_clip_val=1.0,
                        callbacks=[checkpoint_callback])

    wandb.init(
        # set the wandb project where this run will be logged
        project="night_challenge_2024",

        # track hyperparameters and run metadata
        config={
            "learning_rate": 3e-4,
            "architecture": "U-Net",
            "dataset": "SonySID",
            "batch_size": batch_size,
            "epochs": epoch,
        })
    trainer.fit(model=denoise_traniner,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader)

    wandb.finish()
