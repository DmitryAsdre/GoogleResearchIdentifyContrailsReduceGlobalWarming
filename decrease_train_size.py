import os
from tqdm import tqdm

import numpy as np


class CFG:
    DATA_PATH = '../data/'
    OUTPUT_DATA_PATH = '../data/ssd_data'
    input_dataset_name = 'vanilla_data'
    output_dataset_name = 'bands_all_bands_ch4'
    bands = ['band_08.npy', 'band_09.npy', 'band_10.npy', 'band_11.npy', 'band_12.npy', 'band_13.npy', 'band_14.npy', 'band_15.npy', 'band_16.npy']
    chns = [4]
    human_msks_train = ['human_individual_masks.npy', 'human_pixel_masks.npy']
    human_msks_valid = ['human_pixel_masks.npy']
    
    
def preprocess_dataset(folder='train'):
    input_dataset_path = os.path.join(CFG.DATA_PATH, CFG.input_dataset_name)
    output_dataset_path = os.path.join(CFG.OUTPUT_DATA_PATH, CFG.output_dataset_name)
    
    for inst_name in tqdm(os.listdir(os.path.join(input_dataset_path, folder))):
        inst_path = os.path.join(input_dataset_path, folder, inst_name)
        
        cur_output_path = os.path.join(output_dataset_path, folder, inst_name)            
        os.makedirs(cur_output_path, exist_ok=True)
        
        bands_np = []
        for band_name in CFG.bands:
            cur_band = np.load(os.path.join(inst_path, band_name))
            bands_np.append(cur_band[:, :, CFG.chns])
            
        cur_band = np.stack(bands_np, axis=2)
        np.save(os.path.join(cur_output_path, 'band.npy'), cur_band)
        
        if folder == 'train':
            human_msks = CFG.human_msks_train
        elif folder == 'validation':
            human_msks = CFG.human_msks_valid
        
        for human_mask_name in human_msks:
            cur_mask = np.load(os.path.join(inst_path, human_mask_name))
            np.save(os.path.join(cur_output_path, human_mask_name), cur_mask.astype(np.uint8))
    
    
if __name__ == '__main__':
    preprocess_dataset('train')
    preprocess_dataset('validation')
        
        
            
        
    