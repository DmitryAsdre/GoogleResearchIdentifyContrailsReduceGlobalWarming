import os
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast

from torch.utils.tensorboard import SummaryWriter

import torchvision.transforms as T

import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss
from segmentation_models_pytorch.losses import SoftBCEWithLogitsLoss

from transformers import SegformerForSemanticSegmentation

from utils.metrics import dice_avg

class CFG:
    exp_name = 'baseline_segformer_256_nvidia_b3'
    
    DATATRAIN_PATH = '../data/ssd_data/vanilla_data/train'
    DATAVALID_PATH = '../data/ssd_data/vanilla_data/validation'
    
    PATH_TO_SAVE = f'../models/{exp_name}'
    
    write_to_tensorboard = True
    
    n_epoch = 45
    use_amp = False
    
    device = 'cuda:0'
    
    
    smooth_loss = 1.0
    train_loss = SoftBCEWithLogitsLoss(weight=torch.tensor(torch.Tensor([7.31]).to(device)))
    #DiceLoss(mode='binary', smooth=smooth_loss)
    valid_loss = DiceLoss(mode='binary')
    
    valid_loss_th = DiceLoss(mode='binary', from_logits=False)
    
    backbone = 'nvidia/mit-b3'
    
    
    n_workers = 2
    batch_size = 20
    lr = 5e-4
    
    resize = True
    resize_value = 256

    decoder_attention_type = None
    decoder_use_batchnorm = False
    
    flip_augs = False
    hflip_p = 0.5
    vflip_p = 0.5
    
    rotate_augs = False
    rotate_p = 0.5
    
    

class ContrailDataset(Dataset):
    
    def __init__(self, dataset_path, data_type):        
        self.dataset_path = dataset_path
        self.imgs = os.listdir(self.dataset_path)
        self.data_type = data_type
       
        self.resize_image = T.Resize(CFG.resize_value,interpolation=T.InterpolationMode.BILINEAR,antialias=True)
        self.resize_mask = T.Resize(CFG.resize_value,interpolation=T.InterpolationMode.NEAREST,antialias=True)
   
    def __len__(self):
        return len(self.imgs)
    
    @staticmethod
    def normalize_range(data, bounds):
        """Maps data to the range [0, 1]."""
        return (data - bounds[0]) / (bounds[1] - bounds[0])
    
    @staticmethod
    def get_false_color(record_data):
        __T11_BOUNDS = (243, 303)
        _CLOUD_TOP_TDIFF_BOUNDS = (-4, 5)
        _TDIFF_BOUNDS = (-4, 2)
    
        N_TIMES_BEFORE = 4

        r = ContrailDataset.normalize_range(record_data["band_15"] - record_data["band_14"], _TDIFF_BOUNDS)
        g = ContrailDataset.normalize_range(record_data["band_14"] - record_data["band_11"], _CLOUD_TOP_TDIFF_BOUNDS)
        b = ContrailDataset.normalize_range(record_data["band_14"], __T11_BOUNDS)
        false_color = np.clip(np.stack([r, g, b], axis=2), 0, 1)
        img = false_color[..., N_TIMES_BEFORE]
        return img

    def __getitem__(self, idx):
        
        record_id = self.imgs[idx]
        record_dir = os.path.join(self.dataset_path, record_id)
    
        record_data = {}
        record_data['band_11'] = np.load(os.path.join(record_dir, 'band_11.npy'))
        record_data['band_14'] = np.load(os.path.join(record_dir, 'band_14.npy'))
        record_data['band_15'] = np.load(os.path.join(record_dir, 'band_15.npy'))

        false_color = ContrailDataset.get_false_color(record_data)        
        
        human_pixel_mask = np.load(os.path.join(record_dir,'human_pixel_masks.npy')) 
        
        false_color = torch.from_numpy(false_color)
        human_pixel_mask = torch.from_numpy(human_pixel_mask)
        
        false_color = torch.moveaxis(false_color,-1,0)
        human_pixel_mask = torch.moveaxis(human_pixel_mask,-1,0).squeeze(0)
        
        if CFG.resize:
            false_color = self.resize_image(false_color)
        
        
            
        if self.data_type == 'train':                
            if CFG.flip_augs:
                random_hflip_p = np.random.uniform(0, 1)
                random_vflip_p = np.random.uniform(0, 1)
            
            
                if random_hflip_p <= CFG.hflip_p:
                    false_color = torch.flip(false_color, dims=(1,))
                    human_pixel_mask = torch.flip(human_pixel_mask, dims=(0,))
                    
                if random_vflip_p <= CFG.vflip_p:
                    false_color = torch.flip(false_color, dims=(2,))
                    human_pixel_mask = torch.flip(human_pixel_mask, dims=(1,))
                    
            if CFG.rotate_augs:
                angle = np.random.choice([1, 2, 3])
                
                if np.random.uniform(0, 1) < CFG.rotate_p:
                    false_color = torch.rot90(false_color, k=angle, dims=[1, 2])
                    human_pixel_mask = torch.rot90(human_pixel_mask, k=angle, dims=[0, 1])
            
        normalize_image = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        false_color = normalize_image(false_color)
                  
        return false_color, human_pixel_mask.float()
    
    
def train_one_epoch(model, optimizer, dataloader, scheduler=None):
    model = model.train()
    losses = []
    scaler = GradScaler(enabled=CFG.use_amp)
    
    for imgs, masks in (pbar := tqdm(dataloader, total=len(dataloader))):
        imgs, masks = imgs.to(CFG.device), masks.to(CFG.device).to(torch.float)
        
        #with autocast(CFG.use_amp, dtype=torch.float16):
        preds = model(imgs)[0].squeeze(1)
        preds, masks = preds.squeeze(), masks.squeeze()
        
        if CFG.resize:
            preds = T.Resize(masks.shape[1],interpolation=T.InterpolationMode.BILINEAR,antialias=True)(preds)
        
        pbar.set_description(f'mean dice - {np.mean(losses)}, last dice = {losses[-1:]}')
        
        loss = CFG.train_loss(preds, masks)
        
        #scaler.scale(loss).backward()
        loss.backward()
        optimizer.step()
        #scaler.update()
        if scheduler:
            scheduler.step()
            
        losses.append(loss.item())
        optimizer.zero_grad()
        
    return np.mean(losses)

def get_ths(all_preds, all_masks):
    all_preds_sigmoid = torch.nn.Sigmoid()(all_preds)
    
    ths, losses = [], []
    for th in np.linspace(0.05, 1.01, 25):
        all_preds_cur = (all_preds_sigmoid > th).to(int)
        cur_loss = CFG.valid_loss_th(all_preds_cur, all_masks)
        
        ths.append(th)
        losses.append(cur_loss.item())
        
    return ths, losses
        

def valid_one_epoch(model, dataloader):
    model = model.eval()
    losses = []
    all_preds = []
    all_masks = []
    for imgs, masks in tqdm(dataloader, total=len(dataloader)):
        imgs, masks = imgs.to(CFG.device), masks.to(CFG.device)
        
        with torch.no_grad():
            preds = model(imgs)[0]
            if CFG.resize:
                preds = T.Resize(masks.shape[1],interpolation=T.InterpolationMode.BILINEAR,antialias=True)(preds)
            loss = CFG.valid_loss(preds, masks)
            
        masks = masks.to('cpu')
        preds = preds.to('cpu')
        
        all_preds.append(preds)
        all_masks.append(masks)
            
        losses.append(loss.item())
    all_preds = torch.stack(all_preds, dim=1).contiguous()
    all_masks = torch.stack(all_masks, dim=1).contiguous()
    
    print(all_preds.shape, all_masks.shape)
    
    global_loss = CFG.valid_loss(all_preds, all_masks)
    
    ths, losses_ = get_ths(all_preds, all_masks)
        
    return np.mean(losses), global_loss, (ths, losses_), (all_preds, all_masks)
        
    
if __name__ == '__main__':    
    os.makedirs(CFG.PATH_TO_SAVE, exist_ok=True)
    
    if CFG.write_to_tensorboard:
        writer = SummaryWriter(comment=CFG.exp_name)
    
    train_ds = ContrailDataset(CFG.DATATRAIN_PATH, 'train')
    valid_ds = ContrailDataset(CFG.DATAVALID_PATH, 'validation')
    
    train_dl = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.n_workers, drop_last=True)
    valid_dl = DataLoader(valid_ds, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.n_workers, drop_last=True)

    model = SegformerForSemanticSegmentation.from_pretrained(
            CFG.backbone, 
            return_dict=False, 
            num_labels=1,
            ignore_mismatched_sizes=True,
        ).to(CFG.device)
    #smp.Unet(encoder_name=CFG.backbone, decoder_attention_type=CFG.decoder_attention_type, decoder_use_batchnorm=True).to(CFG.device)
    
    optimizer = AdamW(model.parameters(), lr=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=80, eta_min=1e-5, last_epoch=-1)
    
    for i in range(CFG.n_epoch):
        loss_train = train_one_epoch(model, optimizer, train_dl, scheduler)
        loss_valid, global_loss_valid, (ths, losses_), (all_preds, all_masks) = valid_one_epoch(model, valid_dl)
        print(f'EPOCH - {i} : loss_train - {loss_train}, loss_valid - {loss_valid}, global loss valid - {global_loss_valid}')
        best_loss_idx = np.argmin(losses_)        
        print(f'Best th : {ths[best_loss_idx]}, Best dice loss : {losses_[best_loss_idx]}')
        
        
        if CFG.write_to_tensorboard:
            writer.add_scalar('Train BCE', 1. - loss_train, i)
            writer.add_scalar('Valid DICE', 1. - loss_valid, i)
            writer.add_scalar('Valid TH', ths[best_loss_idx], i)
            writer.add_scalar('Valid DICE TH', 1. - losses_[best_loss_idx], i)
            
        
        
        torch.save({'model' : model.state_dict(),
                    'best_th' : ths[best_loss_idx],
                    'best_dice' : losses_[best_loss_idx], 
                    'resize' : CFG.resize,
                    'size' : CFG.resize_value},                    
                    os.path.join(CFG.PATH_TO_SAVE, f'model_{i}.pth'))
        torch.save({'all_preds' : all_preds,
                    'all_masks' : all_masks,
                    'best_th' : ths[best_loss_idx],
                    'best_loss' : losses_[best_loss_idx]},
                    os.path.join(CFG.PATH_TO_SAVE, f'predictions_{i}.pth'))