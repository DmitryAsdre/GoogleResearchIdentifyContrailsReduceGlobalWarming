import os
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T

import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss
from segmentation_models_pytorch.losses import SoftBCEWithLogitsLoss

from utils.metrics import dice_avg

class CFG:
    exp_name = 'baseline_mit_b1_unet'
    
    DATATRAIN_PATH = '../data/ssd_data/vanilla_data'
    DATAVALID_PATH = '../data/ssd_data/vanilla_data'
    
    PATH_TO_SAVE = f'../models/{exp_name}'
    
    loss = DiceLoss(mode='binary')
    device = 'cuda:1'
    bce_loss = SoftBCEWithLogitsLoss(weight=torch.tensor([5.0]).to(device))
    
    backbone = 'timm-resnest26d'
    
    
    n_workers = 12
    batch_size = 12
    in_chans = 3
    
    resize = False
    resize_value = 256
    
    size = 256
    decoder_attention_type = None#'scse'
    
    transformations = {
        "train" : [
        A.RandomResizedCrop(
             size, size, scale=(0.5, 1.0)),
        #A.Resize(size, size),
        #A.HorizontalFlip(p=0.5),
        #A.VerticalFlip(p=0.5),
        #A.RandomBrightnessContrast(p=0.75),
        #A.ShiftScaleRotate(p=0.75),
        
        #A.OneOf([
        #        A.GaussNoise(var_limit=[10, 50]),
        #        A.GaussianBlur(),
        #        A.MotionBlur(),
        #        ], p=0.4),
        #A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
        #A.CoarseDropout(max_holes=1, max_width=int(size * 0.3), max_height=int(size * 0.3), 
        #                mask_fill_value=0, p=0.5),
        # A.Cutout(max_h_size=int(size * 0.6),
        #          max_w_size=int(size * 0.6), num_holes=1, p=1.0),
        #A.Normalize(
        #    mean= [0] * in_chans,
        #    std= [1] * in_chans
        #),
        ToTensorV2(transpose_mask=True),
    ],
    "valid" : [
        #A.Resize(size, size),
        #A.Normalize(
        #    mean= [0] * in_chans,
        #    std= [1] * in_chans
        #),
        ToTensorV2(transpose_mask=True),
    ],
    "test" :[
        A.Resize(size, size),
        A.Normalize(
            mean= [0] * in_chans,
            std= [1] * in_chans
        ),
        ToTensorV2(transpose_mask=True),
    ]}
    
#A custom Dataset class must implement three functions: __init__, __len__, and __getitem__
class ContrailDataset(Dataset):
    
    def __init__(self, base_dir, data_type='train', bands=['band_11', 'band_12', 'band_13', 'band_14', 'band_15']):        
        self.base_dir = base_dir
        self.data_type = data_type
        self.record = os.listdir(self.base_dir +'/'+ self.data_type)
       
        self.resize_image = T.Resize(CFG.resize_value,interpolation=T.InterpolationMode.BILINEAR,antialias=True)
        self.resize_mask = T.Resize(CFG.resize_value,interpolation=T.InterpolationMode.NEAREST,antialias=True)
        self.bands = bands
   
    def __len__(self):
        return len(self.record)
    
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
        img = img*255
        img = img.astype(np.int32)
        img = img.astype(np.float32)
        #img_prev = false_color[..., N_TIMES_BEFORE - 1]
        #img_post = false_color[..., N_TIMES_BEFORE + 1]
        #img = np.concatenate([img_prev, img0, img_post], axis=-1)

        return img

    def __getitem__(self, idx):
        
        record_id = self.record[idx]
        record_dir = os.path.join(self.base_dir, self.data_type, record_id)
        
        #bands_ = []
        #for bands_name in self.bands:
        #    cur_band = np.load(os.path.join(record_dir, bands_name + '.npy'))
        #    bands_.append(cur_band[:,:, 4])
            
        #false_color = []
        #for i in range(len(bands_)):
        #    for j in range(i):
        #        false_color.append(bands_[i] - bands_[j])
        
        band_11 = np.load(os.path.join(record_dir, 'band_11.npy'))
        band_14 = np.load(os.path.join(record_dir, 'band_14.npy'))
        band_15 = np.load(os.path.join(record_dir, 'band_15.npy'))
        
        #band_10 = np.load(os.path.join(record_dir, 'band_10.npy'))
        #band_16 = np.load(os.path.join(record_dir, 'band_16.npy'))
        #band_13 = np.load(os.path.join(record_dir, 'band_13.npy'))
        
        #band_12 = np.load(os.path.join(record_dir, 'band_12.npy'))
        #band_8 = np.load(os.path.join(record_dir, 'band_08.npy'))
        #band_9 = np.load(os.path.join(record_dir, 'band_09.npy'))
        
        record_data = {}
        record_data['band_11'] = band_11
        record_data['band_14'] = band_14
        record_data['band_15'] = band_15
        false_color = ContrailDataset.get_false_color(record_data)
        #false_color = np.stack(false_color, axis=2)
        
        
        human_pixel_mask = np.load(os.path.join(record_dir,'human_pixel_masks.npy')) 
        
        false_color = torch.from_numpy(false_color)#.clone().detach()
        human_pixel_mask = torch.from_numpy(human_pixel_mask)#.clone().detach()
        
        false_color = torch.moveaxis(false_color,-1,0)
        human_pixel_mask = torch.moveaxis(human_pixel_mask,-1,0)
            
        if self.data_type == 'train':
            
            random_crop_factor = torch.rand(1)
            crop_min, crop_max = 0.5 , 1
            crop_factor = crop_min + random_crop_factor * (crop_max-crop_min) 
            crop_size = int(crop_factor * 256)
            self.crop = T.CenterCrop(size=crop_size)
            
            false_color = self.crop(false_color)
            human_pixel_mask =  self.crop(human_pixel_mask)
            
            false_color = self.resize_image(false_color)
            human_pixel_mask =  self.resize_mask(human_pixel_mask)
            
            #random_hflip_p = np.random.uniform(0, 1)
            #srandom_vflip_p = np.random.uniform(0, 1)
            
            #if random_hflip_p <= 0.5:
            #    false_color = torch.flip(false_color, dims=(1,))
            #    human_pixel_mask = torch.flip(human_pixel_mask, dims=(0,))
                
            #if random_vflip_p <= 0.5:
            #    false_color = torch.flip(false_color, dims=(2,))
            #    human_pixel_mask = torch.flip(human_pixel_mask, dims=(1,))
                
            
                

        
        #if CFG.resize and self.data_type=='validation':
            #false_color = self.resize_image(false_color)
            #human_pixel_mask =  self.resize_mask(human_pixel_mask)
            
        normalize_image = T.Normalize((0)*false_color.shape[0], (1)*false_color.shape[0])
        false_color = normalize_image(false_color)
                  
        # false color is scaled between 0 and 1!
        return false_color, human_pixel_mask.float()
    
    
import torch
from segmentation_models_pytorch.base import initialization as init
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import (
    #SegmentationModel,
    SegmentationHead,
    ClassificationHead,
)
from einops import rearrange, reduce, repeat



class SegmentationModelMeanPooled(torch.nn.Module):
    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)
    
    def check_input_shape(self, x):

        h, w = x.shape[-2:]
        output_stride = self.encoder.output_stride
        if h % output_stride != 0 or w % output_stride != 0:
            new_h = (h // output_stride + 1) * output_stride if h % output_stride != 0 else h
            new_w = (w // output_stride + 1) * output_stride if w % output_stride != 0 else w
            raise RuntimeError(
                f"Wrong input shape height={h}, width={w}. Expected image height and width "
                f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
            )

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        n_ch = x.shape[1]
        if self.extend_chs:
            idxs =  [(i, i + 1, i+2) for i in range(0, n_ch - 3 + 1, 2)]
            idxs.extend([i, i + 5, i + 10] for i in range(0, n_ch - 11 + 1, 4))
            idxs.extend([i, i + 7, i + 14] for i in range(0, n_ch - 15 + 1, 4))
            x_prepared = [x[:, torch.LongTensor(idx), :, :] for idx in idxs]
        else:
            x_prepared = [x[:, i : i + 3, :, :] for i in range(0, n_ch - self.in_channels + 1, self.crop_stride)]
        K = len(x_prepared)
        B = x.shape[0]
        
        
        x_prepared = torch.cat(x_prepared, axis=0)

        features = self.encoder(x_prepared)
        
        
        for i in range(len(features)):
            e = features[i]
            _, c, h, w = e.shape
            e = rearrange(e, '(K B) c h w -> B K c h w', K=K, B=B, h=h, w=w)
            features[i] = self.conv3d(e).squeeze(1)
            #if self.pooling_type == 'mean':
            #    features[i] = e.mean(0)
            #elif self.pooling_type == 'max':
            #    features[i], _ = torch.max(e, 0)
            #elif self.pooling_type == 'min':
            #    features[i], _ = torch.min(e, 0)
                
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks
    
from typing import Optional, Union, List

from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import (
    SegmentationModel,
    SegmentationHead,
    ClassificationHead,
)
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder


class UnetMeanPooled(SegmentationModelMeanPooled):

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        in_strides : int = 3,
        extend_chs = False,
        crop_stride : int = 2,
        classes: int = 1,
        pooling_type = 'mean',
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()
        
        self.extend_chs = extend_chs
        self.crop_stride = crop_stride
        self.in_channels = in_channels
        self.pooling_type = pooling_type
        self.in_strides = in_strides
        
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )
        
        self.conv3d = torch.nn.Conv3d(self.in_strides, 1, kernel_size=3, stride=1, padding=1)


        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()

    
#class ContrailDataset(Dataset):
#    def __init__(self, base_dir, imgs, transformations=None, train=False):
#        self.imgs = imgs
#        self.train = train
#        self.base_dir = base_dir
#        self.transformations = transformations
#    
#    def __len__(self):
#        return len(self.imgs)
    
#    def __getitem__(self, idx):
#        img_name = self.imgs[idx]
#        img_path = os.path.join(self.base_dir, img_name)
#        img = np.load(os.path.join(img_path, 'band.npy'))
#        mask = np.load(os.path.join(img_path, 'human_pixel_masks.npy')).astype(np.int32)
        
#        img, mask = torch.tensor(img).squeeze(-1), torch.tensor(mask).squeeze(-1)
#        if self.transformations:
#            res = self.transformations(image = img.numpy(), mask=mask.numpy())
#            
#            img = res['image']
#            mask = res['mask']
#        
#        return img, mask
    
def train_one_epoch(model, optimizer, dataloader, scheduler=None):
    model = model.train()
    losses = []
    
    for imgs, masks in (pbar := tqdm(dataloader, total=len(dataloader))):
        imgs, masks = imgs.to(CFG.device), masks.to(CFG.device).to(torch.float)
        
        preds = model(imgs).squeeze(1)
        preds, masks = preds.squeeze(), masks.squeeze()
        pbar.set_description(f'mean dice - {np.mean(losses)}, last dice = {losses[-1:]}')
        
        #print(preds.shape, masks.shape)
        loss = CFG.loss(preds, masks)
        
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        losses.append(loss.item())
        optimizer.zero_grad()
        
    return np.mean(losses)

def valid_one_epoch(model, dataloader):
    model = model.eval()
    losses = []
    all_preds = []
    all_masks = []
    for imgs, masks in tqdm(dataloader, total=len(dataloader)):
        imgs, masks = imgs.to(CFG.device), masks.to(CFG.device)
        
        with torch.no_grad():
            preds = model(imgs)
            loss = CFG.loss(preds, masks)
            
        preds = preds.to('cpu')
        masks = masks.to('cpu')
        
        all_preds.append(preds)
        all_masks.append(masks)
            
        losses.append(loss.item())
    all_preds = torch.stack(all_preds, dim=1)
    all_masks = torch.stack(all_masks, dim=1)
    
    global_loss = CFG.loss(all_preds, all_masks)
        
    return np.mean(losses), global_loss
        
    
if __name__ == '__main__':
    #train_base_dir = os.path.join(CFG.DATATRAIN_PATH, 'train')
    #valid_base_dir = os.path.join(CFG.DATAVALID_PATH, 'validation')
    
    #train_imgs = os.listdir(train_base_dir)
    #valid_imgs = os.listdir(valid_base_dir)
    
    train_ds = ContrailDataset(CFG.DATATRAIN_PATH, 'train')
    valid_ds = ContrailDataset(CFG.DATAVALID_PATH, 'validation')
    #train_ds = ContrailDataset(train_base_dir, train_imgs, transformations=A.Compose(CFG.transformations['train']), train=True)
    #valid_ds = ContrailDataset(valid_base_dir, valid_imgs, transformations=A.Compose(CFG.transformations['valid']), train=True)
    
    train_dl = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.n_workers, drop_last=True)
    valid_dl = DataLoader(valid_ds, batch_size=CFG.batch_size*2, shuffle=False, num_workers=CFG.n_workers, drop_last=True)

    model = smp.Unet(encoder_name=CFG.backbone, decoder_attention_type=CFG.decoder_attention_type, in_channels=CFG.in_chans, decoder_use_batchnorm=True).to(CFG.device)
    #model = UnetMeanPooled(encoder_name=CFG.backbone, decoder_attention_type=CFG.decoder_attention_type, pooling_type='mean', crop_stride=3).to(CFG.device)
    #smp.Unet(encoder_name=CFG.backbone, decoder_attention_type=CFG.decoder_attention_type, in_channels=CFG.in_chans).to(CFG.device)
    optimizer = AdamW(model.parameters(), lr=5e-4)
    
    for i in range(30):
        loss_train = train_one_epoch(model, optimizer, train_dl)
        loss_valid, global_loss_valid = valid_one_epoch(model, valid_dl)
        print(f'EPOCH - {i} : loss_train - {loss_train}, loss_valid - {loss_valid}, global loss valid - {global_loss_valid}')