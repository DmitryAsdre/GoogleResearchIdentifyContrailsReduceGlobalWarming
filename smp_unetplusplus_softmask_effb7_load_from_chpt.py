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

from utils.metrics import dice_avg

#def Loss(preds, masks):
#    dl = DiceLoss(mode='binary', smooth=1.0)
#    bcel = SoftBCEWithLogitsLoss(weight=torch.tensor(torch.Tensor([8.31]).to('cuda:1')))
#    
#    return dl(preds, masks) + 2.0*bcel(preds, masks)

class CFG:
    exp_name = 'softmask_efficientnet_b8_256_unetplusplus_bce_8_31_05_retrain_v2_focall'
    
    DATATRAIN_PATH = '../data/ssd_data/vanilla_data/train'
    DATAVALID_PATH = '../data/ssd_data/vanilla_data/validation'
    
    PATH_TO_SAVE = f'../models/{exp_name}'
    
    write_to_tensorboard = True
    
    n_epoch = 150
    use_amp = False
    device = 'cuda:0'
    
    #device = SoftBCEWithLogitsLoss(weight=torch.tensor([8.31]).to(device))
    
    
    smooth_loss = 1.0
    train_loss = smp.losses.FocalLoss(mode='binary')
    #SoftBCEWithLogitsLoss(weight=torch.tensor([8.31]).to(device))
    #smp.losses.FocalLoss(mode='binary')
    #DiceLoss(mode='binary', smooth=1.0)
    #SoftBCEWithLogitsLoss(weight=torch.tensor([8.31]).to(device))
    #smp.losses.FocalLoss(mode='binary')
    #SoftBCEWithLogitsLoss(weight=torch.tensor([8.31]).to(device))
    #SoftBCEWithLogitsLoss(weight=torch.tensor([8.31]).to(device))
    #smp.losses.TverskyLoss(mode='binary')
    #smp.losses.FocalLoss(mode='binary')
    #SoftBCEWithLogitsLoss(weight=torch.tensor([7.0]).to(device))
    #DiceLoss(mode='binary', smooth=smooth_loss)
    valid_loss = DiceLoss(mode='binary')
    
    valid_loss_th = DiceLoss(mode='binary', from_logits=False)
    
    backbone = 'timm-efficientnet-b8'
    
    CHECKPOINT_PATH = '../models/softmask_efficientnet_b8_256_unetplusplus_bce_8_31_05_retrain_v2/model_23.pth'
    #'../models/softmask_effb7_manet_plus_augs_05_bce_8_31/model_149.pth'
    
    
    n_workers = 2
    batch_size = 6
    lr = 5e-4
    #max_grad_norm = 1e6
    
    resize = True
    resize_value = 256

    decoder_attention_type = 'scse'#None
    decoder_use_batchnorm = True
    
    train_augmentations = A.Compose([A.RandomRotate90(p=0.5),
                                     A.Flip(p=0.5),
                                    #A.Cutout(max_h_size=int(resize_value * 0.03),
                                     #         max_w_size=int(resize_value * 0.03), num_holes=3, p=0.3)
                                     #A.Affine(scale=(0.87, 1.0), translate_percent=(0.0, 0.07), rotate=(-10, 10), p=0.2),
                                     #A.Rotate((1, 25), p=0.5)
                                     #A.ColorJitter(p=0.2),
                                     #A.HueSaturationValue(p=0.2, hue_shift_limit=10, sat_shift_limit=14, val_shift_limit=10)
                                     ]
                                    )
    p_mixup = 0.0
    beta_a = 0.2
    beta_b = 0.2
    
    

class ContrailDataset(Dataset):
    
    def __init__(self, dataset_path, data_type, train_augmentations=None):        
        self.dataset_path = dataset_path
        self.imgs = os.listdir(self.dataset_path)
        self.data_type = data_type
       
        self.resize_image = T.Resize(CFG.resize_value,interpolation=T.InterpolationMode.BILINEAR,antialias=True)
        self.resize_mask = T.Resize(256,interpolation=T.InterpolationMode.NEAREST,antialias=True)
        self.train_augmentations = train_augmentations
   
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

    @staticmethod
    def load_img(record_dir, with_mask=True):
        record_data = {}
        record_data['band_11'] = np.load(os.path.join(record_dir, 'band_11.npy'))
        record_data['band_14'] = np.load(os.path.join(record_dir, 'band_14.npy'))
        record_data['band_15'] = np.load(os.path.join(record_dir, 'band_15.npy'))

        false_color = ContrailDataset.get_false_color(record_data) 
        
        human_pixel_mask = None
        if with_mask:
            human_pixel_mask = np.load(os.path.join(record_dir, 'human_individual_masks.npy')).mean(axis=3) 
        
        return false_color, human_pixel_mask
            
    
    def __getitem__(self, idx):
        
        record_id = self.imgs[idx]
        record_dir = os.path.join(self.dataset_path, record_id)
    
        #record_data = {}
        #record_data['band_11'] = np.load(os.path.join(record_dir, 'band_11.npy'))
        #record_data['band_14'] = np.load(os.path.join(record_dir, 'band_14.npy'))
        #record_data['band_15'] = np.load(os.path.join(record_dir, 'band_15.npy'))

        #false_color = ContrailDataset.get_false_color(record_data)        
        
        #human_pixel_mask = np.load(os.path.join(record_dir,'human_pixel_masks.npy')) 
        
            
        if self.data_type == 'train':   
            false_color, human_pixel_mask = ContrailDataset.load_img(record_dir, True)  
            if np.random.uniform(0, 1) < CFG.p_mixup:
                idx = np.random.randint(0, len(self.imgs))
                record_id_mixup = self.imgs[idx]
                record_dir_mixup = os.path.join(self.dataset_path, record_id_mixup)
                false_color_mixup, human_pixel_mask_mixup = ContrailDataset.load_img(record_dir_mixup, True)
                
                alpha = np.random.beta(CFG.beta_a, CFG.beta_b)  
                false_color = alpha*false_color + (1. - alpha)*false_color_mixup
                human_pixel_mask = alpha*human_pixel_mask + (1. - alpha)*human_pixel_mask_mixup 
                      
            if self.train_augmentations:
                res = self.train_augmentations(image=false_color, mask=human_pixel_mask)
                false_color = res['image']
                human_pixel_mask = res['mask']
        else:
            false_color, _ = ContrailDataset.load_img(record_dir, with_mask=False)
            human_pixel_mask = np.load(os.path.join(record_dir,'human_pixel_masks.npy')) 
                        
        false_color = torch.from_numpy(false_color)
        human_pixel_mask = torch.from_numpy(human_pixel_mask)
        
        false_color = torch.moveaxis(false_color,-1,0).contiguous()
        human_pixel_mask = torch.moveaxis(human_pixel_mask,-1,0).squeeze(0).contiguous()
                        
        if CFG.resize:
            false_color = self.resize_image(false_color)
            human_pixel_mask = self.resize_mask(human_pixel_mask.unsqueeze(0)).squeeze(0)
            
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
        preds = model(imgs).squeeze(1)
        preds, masks = preds.squeeze(), masks.squeeze()
        
        if CFG.resize:
            preds = T.Resize(masks.shape[1],interpolation=T.InterpolationMode.BILINEAR,antialias=True)(preds)
        
        pbar.set_description(f'mean dice - {np.mean(losses)}, last dice = {losses[-1:]}')
        
        loss = CFG.train_loss(preds, masks)
        
        #scaler.scale(loss).backward()
        loss.backward()
        
        #grad_norm = torch.nn.utils.clip_grad_norm_(
        #        model.parameters(), CFG.max_grad_norm)
        
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
            preds = model(imgs)
            if CFG.resize:
                preds = T.Resize(masks.shape[1],interpolation=T.InterpolationMode.BILINEAR,antialias=True)(preds)
            loss = CFG.valid_loss(preds, masks)
            
        preds = preds.to('cpu')
        masks = masks.to('cpu')
        
        all_preds.append(preds)
        all_masks.append(masks)
            
        losses.append(loss.item())
    all_preds = torch.stack(all_preds, dim=1)
    all_masks = torch.stack(all_masks, dim=1)
    
    global_loss = CFG.valid_loss(all_preds, all_masks)
    all_preds = all_preds.contiguous()
    all_masks = all_masks.contiguous()
    
    ths, losses_ = get_ths(all_preds, all_masks)
        
    return np.mean(losses), global_loss, (ths, losses_), (all_preds, all_masks)


def load_model():
    state_dict = torch.load(CFG.CHECKPOINT_PATH, map_location=torch.device(CFG.device))
    model = smp.UnetPlusPlus(encoder_name=CFG.backbone,
                             decoder_attention_type=CFG.decoder_attention_type,
                             decoder_use_batchnorm=CFG.decoder_use_batchnorm).to(CFG.device)
    
    #for param in model.encoder.parameters():
    #    param.requires_grad = False
    #model = smp.MAnet(encoder_name='efficientnet-b7', 
    #                      encoder_weights=None).to(CFG.device)
    model.load_state_dict(state_dict['model'])
    
    
    optimizer = AdamW(model.parameters(), lr=CFG.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=80, eta_min=1e-6, last_epoch=-1)
    
    optimizer.load_state_dict(state_dict['optimizer'])
    scheduler.load_state_dict(state_dict['scheduler'])
    
    return model, optimizer, scheduler
    
        
    
if __name__ == '__main__':    
    os.makedirs(CFG.PATH_TO_SAVE, exist_ok=True)
    
    if CFG.write_to_tensorboard:
        writer = SummaryWriter(comment=CFG.exp_name)
    
    train_ds = ContrailDataset(CFG.DATATRAIN_PATH, 'train', train_augmentations=CFG.train_augmentations)
    valid_ds = ContrailDataset(CFG.DATAVALID_PATH, 'validation')
    
    train_dl = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.n_workers, drop_last=True)
    valid_dl = DataLoader(valid_ds, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.n_workers, drop_last=True)

    #model = 
    
    #optimizer = AdamW(model.parameters(), lr=CFG.lr)
    #scheduler = CosineAnnealingLR(optimizer, T_max=80, eta_min=1e-6, last_epoch=-1)
    model, optimizer, scheduler = load_model()
    model.to(CFG.device)
    #optimizer.to(CFG.device)
    #scheduler.to(CFG.device)
    
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
                    'optimizer' : optimizer.state_dict(),
                    'scheduler' : scheduler.state_dict(),
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