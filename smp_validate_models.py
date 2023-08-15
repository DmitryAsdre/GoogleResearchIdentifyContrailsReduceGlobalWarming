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

import timm

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss
from segmentation_models_pytorch.losses import SoftBCEWithLogitsLoss
from torchvision.transforms.functional import hflip, vflip, rotate



from utils.metrics import dice_avg

#def Loss(preds, masks):
#    dl = DiceLoss(mode='binary', smooth=1.0)
#    bcel = SoftBCEWithLogitsLoss(weight=torch.tensor(torch.Tensor([8.31]).to('cuda:1')))
#    
#    return dl(preds, masks) + 2.0*bcel(preds, masks)

class CFG:
    exp_name = 'softmask_efficientnet_b7_512_unet++_bce_831_augs_5'
    
    DATATRAIN_PATH = '../data/ssd_data/vanilla_data/train'
    DATAVALID_PATH = '../data/ssd_data/vanilla_data/validation'
    
    PATH_TO_SAVE = f'../models/{exp_name}'
    
    write_to_tensorboard = False
    
    n_epoch = 150
    use_amp = False
    device = 'cuda:0'
    
    #device = SoftBCEWithLogitsLoss(weight=torch.tensor([8.31]).to(device))
    
    
    smooth_loss = 1.0
    train_loss = SoftBCEWithLogitsLoss(weight=torch.tensor([8.31]).to(device))
    #smp.losses.TverskyLoss(mode='binary')
    #smp.losses.FocalLoss(mode='binary')
    #SoftBCEWithLogitsLoss(weight=torch.tensor([7.0]).to(device))
    #DiceLoss(mode='binary', smooth=smooth_loss)
    valid_loss = DiceLoss(mode='binary')
    
    valid_loss_th = DiceLoss(mode='binary', from_logits=False)
    
    backbone = 'efficientnet-b7'
    
    CHECKPOINT_PATH = '../models/softmask_efficientnet_b7_256_unet++_bce_8_31_augs_5/model_145.pth'
    
    
    n_workers = 2
    batch_size = 4
    lr = 5e-4
    #max_grad_norm = 1e6
    
    resize = True
    resize_value = 256

    decoder_attention_type = 'scse'#None
    decoder_use_batchnorm = True
    
    train_augmentations = A.Compose([A.RandomRotate90(p=0.5),
                                     A.Flip(p=0.5),
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
    
def get_ths(all_preds, all_masks):
    all_preds_sigmoid = all_preds#torch.nn.Sigmoid()(all_preds)
    
    ths, losses = [], []
    for th in np.linspace(0.05, 1.01, 55):
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
    
    ths, losses_ = get_ths(all_preds, all_masks)
        
    return np.mean(losses), global_loss, (ths, losses_), (all_preds, all_masks)

def valid_one_epoch_tta(model, dataloader):
    model = model.eval()
    all_preds = []
    all_masks = []
    for images, masks in tqdm(dataloader, total=len(dataloader)):
        with torch.no_grad():
            images = images.to(CFG.device)
            images_hflip = hflip(images)
            images_vflip = vflip(images)
            images_90 = rotate(images, 90)
            images_180 = rotate(images, 180)
            images_270 = rotate(images, 270)
            
            predicted_mask = model.forward(images[:, :, :, :])
            predicted_mask_hflip = model.forward(images_hflip[:, :, :, :])
            predicted_mask_vflip = model.forward(images_vflip[:, :, :, :])
            predicted_mask_90 = model.forward(images_90[:, :, :, :])
            predicted_mask_180 = model.forward(images_180[:, :, :, :])
            predicted_mask_270 = model.forward(images_270[:, :, :, :])
            
            
            predicted_mask_vflip = vflip(predicted_mask_vflip)
            predicted_mask_hflip = hflip(predicted_mask_hflip)
            predicted_mask_90 = rotate(predicted_mask_90, -90)
            predicted_mask_180 = rotate(predicted_mask_180, -180)
            predicted_mask_270 = rotate(predicted_mask_270, -270)
            
            #predicted_mask_sigmoid = (2.0*predicted_mask + predicted_mask_hflip + \
            #                predicted_mask_vflip + predicted_mask_90 + predicted_mask_180 + predicted_mask_270) / 7.
            #predicted_mask_sigmoid = torch.sigmoid(predicted_mask_sigmoid).cpu()
                
            #predicted_mask = torch.sigmoid(predicted_mask).cpu()#.detach().numpy()
            #predicted_mask_hflip = torch.sigmoid(predicted_mask_hflip).cpu()#.detach().numpy()
            #predicted_mask_vflip = torch.sigmoid(predicted_mask_vflip).cpu()#.detach().numpy()
            #predicted_mask_90 = torch.sigmoid(predicted_mask_90).cpu()#.detach().numpy()
            #predicted_mask_180 = torch.sigmoid(predicted_mask_180).cpu()#.detach().numpy()
            #predicted_mask_270 = torch.sigmoid(predicted_mask_270).cpu()#.detach().numpy()
            
            predicted_mask = (3.0*predicted_mask + predicted_mask_hflip + \
                            predicted_mask_vflip + predicted_mask_90 + predicted_mask_180 + predicted_mask_270) / 8.
            #predicted_mask = predicted_mask#(predicted_mask + predicted_mask_sigmoid) / 2.0
            predicted_mask = predicted_mask.squeeze(1)
            
            #print(masks.shape, predicted_mask.shape)
            all_masks.append(masks)
            all_preds.append(predicted_mask)
    
    with torch.no_grad():
        all_preds = torch.concat(all_preds, dim=0)
        all_masks = torch.concat(all_masks, dim=0)
        
    return all_preds, all_masks


def valid_denoising_model(model, dataloader):
    model = model.eval()
    
    all_preds = []
    
    for images, _ in tqdm(dataloader, total=len(dataloader)):
        with torch.no_grad():
            images = images.to(CFG.device)
            
            prediction = model(images)
            #prediction = torch.sigmoid(prediction)
            
        all_preds.append(prediction.detach().cpu())
        
    with torch.no_grad():
        all_preds = torch.concat(all_preds, dim=0)
        
    return all_preds.to('cpu')
            
            




def load_model(model_params):
    state_dict = torch.load(model_params['model_path'], map_location=torch.device('cpu'))
    if model_params['model_name'] == 'unetplusplus_scse_batchnorm_efficientnetb7':
        model = smp.UnetPlusPlus(encoder_name='efficientnet-b7',
                                decoder_attention_type='scse',
                                decoder_use_batchnorm=True, encoder_weights=None)
    elif model_params['model_name'] == 'manet_efficientnetb7':
        model = smp.MAnet(encoder_name='efficientnet-b7', 
                          encoder_weights=None)
    elif model_params['model_name'] == 'unetplusplus_scse_batchnorm_efficientnetb8':
        model = smp.UnetPlusPlus(encoder_name='timm-efficientnet-b8',
                                decoder_attention_type='scse',
                                decoder_use_batchnorm=True, encoder_weights=None)
    model.load_state_dict(state_dict['model'])
    
    return model

def load_model_denoising(model_params):
    state_dict = torch.load(model_params['model_path'], map_location=torch.device('cpu'))
    if model_params['model_name'] == 'nfnet_l0':
        model = timm.create_model('nfnet_l0', pretrained=False, num_classes=1)
    model.load_state_dict(state_dict['model'])
    
    return model
        

models_params  = [{'model_name' : 'unetplusplus_scse_batchnorm_efficientnetb7',
                   'weight' : 1.0,
                  'model_path' : '../models/baseline_efficientnet_b6/efficientnet_b7_unetplusplus_softmax_augs_05_epch_105_256.pth'},
                  {'model_name' : 'unetplusplus_scse_batchnorm_efficientnetb7',
                   'weight' : 1.0,
                   'model_path' : '../models/baseline_efficientnet_b6/efficientnet_b7_unetplusplus_softmax_augs_05_epch_145_256.pth'},
                  {'model_name' : 'unetplusplus_scse_batchnorm_efficientnetb7',
                   'weight' : 1.0,
                   'model_path' : '../models/baseline_efficientnet_b6/efficientnet_b7_unetplusplus_softmax_augs_05_epch_0_focall_loss.pth'},
                  {'model_name' : 'unetplusplus_scse_batchnorm_efficientnetb8',
                   'weight' : 1.0,
                   'model_path' : '../models/baseline_efficientnet_b6/efficientnet_b8_unetplusplus_softmax_augs_05_epch_0_focall_loss.pth'},
                  {'model_name' : 'unetplusplus_scse_batchnorm_efficientnetb8',
                   'weight' : 1.0,
                   'model_path' : '../models/baseline_efficientnet_b6/efficientnet_b8_unetplusplus_softmax_auugs_05_epch_23.pth'}
                ]

model_params_denoise = [{'model_name' : 'nfnet_l0',
                         'model_path' : '../models/nfnet_l0_denoising_model_full_augs/model_14.pth'}]
        
    
if __name__ == '__main__':    
    valid_ds = ContrailDataset(CFG.DATAVALID_PATH, 'validation')
    valid_dl = DataLoader(valid_ds, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.n_workers, drop_last=True)
    
    all_preds_mean = None
    weight_sum = 0
    
    #model_denoise = load_model_denoising(model_params_denoise[0])
    #model_denoise.to(torch.device(CFG.device))
    #all_preds_denoise = valid_denoising_model(model_denoise, valid_dl).squeeze()
    #print(all_preds_denoise.shape)
    
    #del model_denoise
    #torch.cuda.empty_cache()
    full_preds = []
    
    for model_params in models_params:
        cur_weigth = model_params['weight']
        weight_sum += cur_weigth
        
        model = load_model(model_params)
        model = model.to(CFG.device)

    
    
        all_preds, all_masks = valid_one_epoch_tta(model, valid_dl)
        if all_preds_mean is None:
            all_preds_mean = cur_weigth*all_preds
        else:
            all_preds_mean += cur_weigth*all_preds  
            
        full_preds.append(all_preds)
         
            
    all_preds_mean = all_preds_mean / weight_sum  
    full_preds = torch.stack(full_preds, axis=1)
    print(full_preds.shape)
    
    #with torch.no_grad():
    #    ths, losses_ = get_ths(all_preds_mean, all_masks)
        
    torch.save({'segmentation_pred' : all_preds_mean,
                'full_preds' : full_preds,
                'mask' : all_masks},
               'all_preds_3.pth')
    print(all_preds_mean.shape, all_masks.shape)
    #print(np.min(losses_), ths[np.argmin(losses_)])