import os

from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2

class CFG:
    DATA_PATH = '../data/ssd_data/vanilla_data/train'
    TH = 0.05
    size = 64
    
    PATH_TO_SAVE = f'../data/duplicates_{TH}_{size}.parquet'
    
import numpy as np
import pandas as pd
import os

def to_ashcolor(record_dir, mask=False):
    def normalize_range(data, bounds):
        """Maps data to the range [0, 1]."""
        return (data - bounds[0]) / (bounds[1] - bounds[0])
    
    def get_false_color(record_data):
        __T11_BOUNDS = (243, 303)
        _CLOUD_TOP_TDIFF_BOUNDS = (-4, 5)
        _TDIFF_BOUNDS = (-4, 2)
    
        N_TIMES_BEFORE = 4

        r = normalize_range(record_data["band_15"] - record_data["band_14"], _TDIFF_BOUNDS)
        g = normalize_range(record_data["band_14"] - record_data["band_11"], _CLOUD_TOP_TDIFF_BOUNDS)
        b = normalize_range(record_data["band_14"], __T11_BOUNDS)
        false_color = np.clip(np.stack([r, g, b], axis=2), 0, 1)
        img = false_color[..., N_TIMES_BEFORE]
        return img
    
    record_dir = os.path.join('../data/ssd_data/vanilla_data/train', record_dir)
        
    record_data = {}
    record_data['band_11'] = np.load(os.path.join(record_dir, 'band_11.npy'))
    record_data['band_14'] = np.load(os.path.join(record_dir, 'band_14.npy'))
    record_data['band_15'] = np.load(os.path.join(record_dir, 'band_15.npy'))

    false_color = get_false_color(record_data)
    
    if mask:
        mask = np.load(os.path.join(record_dir, 'human_pixel_masks.npy'))
        return false_color, mask

    else:
        return false_color  
    
    
def find_duplicates(img_names, imgs, i):
    dists = np.abs(imgs - imgs[i]).mean(axis=(1, 2, 3))
    dists_desc = []
    
    argsorted_dists = np.argsort(dists)
    
    argsorted_dists = argsorted_dists[dists[argsorted_dists] < CFG.TH]
    
    for j in argsorted_dists[1:]:
        dists_desc.append({'name_1' : img_names[i], 'name_2' : img_names[j], 'distance' : dists[j]})
    
    return dists_desc


if __name__ == '__main__':
    img_names = os.listdir(CFG.DATA_PATH)
    
    imgs = []
    for img_name in tqdm(img_names):
        cur_img = to_ashcolor(img_name)
        cur_img = cv2.resize(cur_img, (CFG.size, CFG.size))
        imgs.append(cur_img)
        
    imgs = np.stack(imgs)
    
    duplicates = []
    
    for i in tqdm(range(len(img_names))):
        cur_duplicates = find_duplicates(img_names, imgs, i)
        duplicates.extend(cur_duplicates)
        
    duplicates = pd.DataFrame(duplicates)
    
    duplicates.to_parquet(CFG.PATH_TO_SAVE)
        

    
    
    
    