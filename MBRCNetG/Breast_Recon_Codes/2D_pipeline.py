#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
2D-to-3D breast reconstruction pipeline.

Main flow:
1. Read multi-view 2D images and masks.
2. Predict a 3D voxel volume.
3. Train with projection and shape losses.
4. Validate and export results.
"""

import os  # File and folder operations.
import random  # Random choices for augmentation and sampling.
import argparse  # Command-line argument parsing.
from datetime import datetime as dt  # Timestamp strings for output folders.

import cv2  # Image loading and resize operations.
import matplotlib.pyplot as plt  # Plotting and saving debug charts.
import numpy as np  # Array math for masks and voxels.
import torch  # Core deep learning library.
import torch.backends.cudnn  # CUDA backend settings.
import torch.nn as nn  # Neural network layers.
import torch.nn.functional as F  # Functional tensor ops.
import torch.utils.data  # Dataset and DataLoader helpers.
import torchvision.models  # Pretrained image backbones.
import scipy.ndimage as ndi  # Morphology and connected components for voxel cleanup.
from skimage import measure  # Marching cubes for mesh export.
from PIL import Image  # Simple image resize helper for binary masks.
from scipy.ndimage import gaussian_filter  # Smooth voxel boundaries.
from scipy.ndimage import zoom  # Upsample volume before mesh extraction.
from torchvision import transforms  # Image to tensor and normalize transforms.

# Optional logger. Training still runs when this package is missing.
try:
    from tensorboardX import SummaryWriter
except ImportError:
    print("[WARN] tensorboardX not installed. Logging will be disabled.")
    class SummaryWriter:
        def __init__(self, *args, **kwargs): pass
        def add_scalar(self, *args, **kwargs): pass
        def close(self): pass

# ==============================================================================
# CONFIGURATION
# ==============================================================================
class Config:
    """
    Hold all default settings for training and testing.

    Input:
        None.

    Output:
        Namespaced configuration groups used by all pipeline functions.
    """
    class CONST:
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        num_workers = 4
        BATCH_SIZE = 2 
        N_VOX = 64
        IMG_H = 224
        IMG_W = 224
        CROP_IMG_H = 224
        CROP_IMG_W = 224
        WEIGHTS = None # 'path/to/checkpoint.pth'

    class DATASET:
        MEAN = [0.5, 0.5, 0.5]
        STD = [0.5, 0.5, 0.5]

    class TRAIN:
        RESUME_TRAIN = False
        NUM_EPOCHES = 35
        NUM_WORKER = 4
        SAVE_FREQ = 10
        BRIGHTNESS = 0.2
        CONTRAST = 0.2
        SATURATION = 0.2
        NOISE_STD = 0.1
        RANDOM_BG_COLOR_RANGE = [[225, 255], [225, 255], [225, 255]]
        
        ENCODER_LEARNING_RATE = 1e-4
        DECODER_LEARNING_RATE = 1e-4
        REFINER_LEARNING_RATE = 1e-4
        MERGER_LEARNING_RATE = 1e-4
        
        BETAS = (0.9, 0.999)
        ENCODER_LR_MILESTONES = [10, 20]
        DECODER_LR_MILESTONES = [10, 20]
        REFINER_LR_MILESTONES = [10, 20]
        MERGER_LR_MILESTONES = [10, 20]
        GAMMA = 0.5
        
        EPOCH_START_USE_REFINER = 0
        EPOCH_START_USE_MERGER = 0

    class TEST:
        RANDOM_BG_COLOR_RANGE = [[240, 240], [240, 240], [240, 240]]
        SAVE_MAX_N = 10

    class NETWORK:
        TCONV_USE_BIAS = False
        LEAKY_VALUE = 0.2
        USE_REFINER = True
        USE_MERGER = True 

    class DIR:
        DATA_ROOT = '/content/drive/MyDrive/NewModel/Dataset/Single Breast'
        OUT_PATH = '/content/drive/MyDrive/NewModel/2D/Pix2voxTest'

cfg = Config()

def normalize_scale_and_center(img, mask, target_max_size=160):
    """
    Crop the foreground, scale it, and paste it at image center.

    Input:
        img: Image array, shape (H, W, C).
        mask: Binary mask, shape (H, W).
        target_max_size: Largest side after scale.

    Output:
        centered_img: Image on 224x224 canvas.
        centered_mask: Mask on 224x224 canvas.
    """
    y_indices, x_indices = np.where(mask > 0)
    if len(y_indices) == 0:
        return img, mask

    # 1) Tight bounding box from foreground mask.
    min_y, max_y = np.min(y_indices), np.max(y_indices)
    min_x, max_x = np.min(x_indices), np.max(x_indices)

    # 2) Crop object region.
    cropped_mask = mask[min_y:max_y+1, min_x:max_x+1]
    cropped_img = img[min_y:max_y+1, min_x:max_x+1]

    # 3) Uniform scale so longest side matches target_max_size.
    h, w = cropped_mask.shape
    max_dim = max(h, w)
    scale = target_max_size / float(max_dim)

    new_h, new_w = int(h * scale), int(w * scale)

    # Resize cropped image and mask.
    resized_mask = cv2.resize(cropped_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    resized_img = np.zeros((new_h, new_w, img.shape[2]), dtype=img.dtype)
    for c in range(img.shape[2]):
        resized_img[:, :, c] = cv2.resize(cropped_img[:, :, c], (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 4) Create final 224x224 canvas.
    channels = img.shape[2] if img.ndim == 3 else 1
    final_img = np.zeros((224, 224, channels), dtype=img.dtype)
    final_mask = np.zeros((224, 224), dtype=mask.dtype)

    # 5) Paste at image center.
    start_y = (224 - new_h) // 2
    start_x = (224 - new_w) // 2
    final_img[start_y:start_y+new_h, start_x:start_x+new_w] = resized_img
    final_mask[start_y:start_y+new_h, start_x:start_x+new_w] = resized_mask

    return final_img, final_mask

class Voxels(object):
    """
    Simple container for voxel data and binvox metadata.

    Input:
        data: Dense or sparse voxel data.
        dims: Volume size as (D, H, W).
        translate: Binvox translation values.
        scale: Binvox scale value.
        axis_order: 'xyz' or 'xzy'.

    Output:
        Voxels object with helper methods clone() and write().
    """
    def __init__(self, data, dims, translate, scale, axis_order):
        self.data = data
        self.dims = dims
        self.translate = translate
        self.scale = scale
        assert (axis_order in ('xzy', 'xyz'))
        self.axis_order = axis_order

    def clone(self):
        data = self.data.copy()
        dims = self.dims[:]
        translate = self.translate[:]
        return Voxels(data, dims, translate, self.scale, self.axis_order)

    def write(self, fp):
        write_binvox(self, fp)

def sparse_to_dense(voxel_data, dims, dtype=bool):
    """
    Convert sparse voxel coordinates (3xN) into dense grid.

    Input:
        voxel_data: Sparse coordinates.
        dims: Target volume size.
        dtype: Output array dtype.

    Output:
        Dense voxel array.
    """
    if voxel_data.ndim != 2 or voxel_data.shape[0] != 3:
        raise ValueError('[ERROR] voxel_data is wrong shape; should be 3xN array.')
    if np.isscalar(dims):
        dims = [dims] * 3
    dims = np.atleast_2d(dims).T
    xyz = voxel_data.astype(np.int)
    valid_ix = ~np.any((xyz < 0) | (xyz >= dims), 0)
    xyz = xyz[:, valid_ix]
    out = np.zeros(dims.flatten(), dtype=dtype)
    out[tuple(xyz)] = True
    return out

def write_binvox(voxel_model, fp):
    """
    Write a voxel model to binvox format.

    Input:
        voxel_model: Voxels object.
        fp: Open binary file handle.

    Output:
        None. Writes bytes to file.
    """
    if voxel_model.data.ndim == 2:
        dense_voxel_data = sparse_to_dense(voxel_model.data, voxel_model.dims).astype(int)
    else:
        dense_voxel_data = voxel_model.data.astype(int)

    file_header = [
        '#binvox 1\n',
        'dim %s\n' % ' '.join(map(str, voxel_model.dims)),
        'translate %s\n' % ' '.join(map(str, voxel_model.translate)),
        'scale %s\n' % str(voxel_model.scale), 'data\n'
    ]
    for fh in file_header:
        fp.write(fh.encode('latin-1'))

    if voxel_model.axis_order not in ('xzy', 'xyz'):
        raise ValueError('[ERROR] Unsupported voxel model axis order')

    if voxel_model.axis_order == 'xzy':
        voxels_flat = dense_voxel_data.flatten()
    elif voxel_model.axis_order == 'xyz':
        voxels_flat = np.transpose(dense_voxel_data, (0, 2, 1)).flatten()

    state = voxels_flat[0]
    ctr = 0
    for c in voxels_flat:
        if c == state:
            ctr += 1
            if ctr == 255:
                fp.write(chr(state).encode('latin-1'))
                fp.write(chr(ctr).encode('latin-1'))
                ctr = 0
        else:
            fp.write(chr(state).encode('latin-1'))
            fp.write(chr(ctr).encode('latin-1'))
            state = c
            ctr = 1
    if ctr > 0:
        fp.write(chr(state).encode('latin-1'))
        fp.write(chr(ctr).encode('latin-1'))

# ==============================================================================
# UTILS: DATA TRANSFORMS
# ==============================================================================
class DataTransforms:
    """
    Group of image transform utilities used by dataset loader.

    Input:
        Numpy image arrays from dataset.

    Output:
        Transformed images ready for model input.
    """
    class Compose(object):
        def __init__(self, transforms):
            self.transforms = transforms
        def __call__(self, rendering_images, bounding_box=None):
            for t in self.transforms:
                if t.__class__.__name__ in ['RandomCrop', 'CenterCrop']:
                    rendering_images = t(rendering_images, bounding_box)
                else:
                    rendering_images = t(rendering_images)
            return rendering_images

    class ToTensor(object):
        def __call__(self, rendering_images):
            if isinstance(rendering_images, list):
                rendering_images = [transforms.ToTensor()(img) for img in rendering_images]
                tensor = torch.stack(rendering_images, dim=0)
            elif isinstance(rendering_images, np.ndarray):
                array = np.transpose(rendering_images, (0, 3, 1, 2))
                tensor = torch.from_numpy(array).float()
            else:
                raise TypeError(f"Unsupported type: {type(rendering_images)}")
            return tensor

    class Normalize(object):
        def __init__(self, mean, std):
            self.mean = mean
            self.std = std
        def __call__(self, rendering_images):
            rendering_images -= self.mean
            rendering_images /= self.std
            return rendering_images

    class RandomPermuteRGB(object):
        def __call__(self, rendering_images):
            random_permutation = np.random.permutation(3)
            for img_idx, img in enumerate(rendering_images):
                rendering_images[img_idx] = img[..., random_permutation]
            return rendering_images

    class CenterCrop(object):
        def __init__(self, img_size, crop_size):
            self.img_size_h, self.img_size_w = img_size
            self.crop_size_h, self.crop_size_w = crop_size
        def __call__(self, rendering_images, bounding_box=None):
            if len(rendering_images) == 0: return rendering_images
            processed_images = []
            for img in rendering_images:
                img_np = np.array(img)
                h, w = img_np.shape[:2]
                if h > self.crop_size_h and w > self.crop_size_w:
                    x_left = (w - self.crop_size_w) // 2
                    y_top = (h - self.crop_size_h) // 2
                    processed = cv2.resize(img_np[y_top:y_top+self.crop_size_h, x_left:x_left+self.crop_size_w], 
                                         (self.img_size_w, self.img_size_h))
                else:
                    processed = cv2.resize(img_np, (self.img_size_w, self.img_size_h))
                processed_images.append(processed)
            return np.array(processed_images)

    class RandomCrop(object):
        def __init__(self, img_size, crop_size):
            self.img_size_h, self.img_size_w = img_size
            self.crop_size_h, self.crop_size_w = crop_size
        def __call__(self, rendering_images, bounding_box=None):
            if len(rendering_images) == 0: return rendering_images
            processed_images = []
            for img in rendering_images:
                img_np = np.array(img)
                h, w = img_np.shape[:2]
                if h > self.crop_size_h and w > self.crop_size_w:
                    x_left = random.randint(0, w - self.crop_size_w)
                    y_top = random.randint(0, h - self.crop_size_h)
                    cropped = img_np[y_top:y_top+self.crop_size_h, x_left:x_left+self.crop_size_w]
                    processed = cv2.resize(cropped, (self.img_size_w, self.img_size_h))
                else:
                    processed = cv2.resize(img_np, (self.img_size_w, self.img_size_h))
                processed_images.append(processed)
            return np.array(processed_images)

    class RandomFlip(object):
        def __call__(self, rendering_images):
            for img_idx, img in enumerate(rendering_images):
                if random.randint(0, 1):
                    rendering_images[img_idx] = np.fliplr(img)
            return rendering_images

    class RandomBackground(object):
        def __init__(self, color_range):
            self.color_range = color_range
        def __call__(self, rendering_images):
            if len(rendering_images) == 0: return rendering_images
            h, w, c = rendering_images[0].shape
            if c != 4: return rendering_images
            processed = []
            for img in rendering_images:
                r = random.randint(self.color_range[0][0], self.color_range[0][1]) / 255.
                g = random.randint(self.color_range[1][0], self.color_range[1][1]) / 255.
                b = random.randint(self.color_range[2][0], self.color_range[2][1]) / 255.
                alpha = (img[:,:,3] == 0).astype(np.float32)[..., None]
                img_rgb = img[:,:,:3]
                bg = np.array([[[r,g,b]]])
                merged = alpha * bg + (1-alpha) * img_rgb
                processed.append(merged)
            return np.array(processed)

def post_process_voxel(volume, threshold=0.4, remove_outliers=True, fill_holes=True):
    """
    Clean predicted voxel volume with simple morphology steps.

    Input:
        volume: Float voxel grid.
        threshold: Binary threshold.
        remove_outliers: Keep only largest component when True.
        fill_holes: Fill internal holes when True.

    Output:
        Post-processed voxel grid as float32 in {0,1}.
    """
    binary = (volume > threshold).astype(np.uint8)

    # 1) Connect nearby broken regions.
    binary = ndi.binary_closing(binary, iterations=3).astype(np.uint8)

    # 2) Keep largest connected component.
    if remove_outliers:
        labels, num = ndi.label(binary)
        if num > 0:
            sizes = np.bincount(labels.ravel())
            if len(sizes) > 1:
                largest = sizes[1:].argmax() + 1
                binary = (labels == largest).astype(np.uint8)
    # 3) Remove small spikes/noise.
    binary = ndi.binary_opening(binary, iterations=2).astype(np.uint8)

    # 4) Fill holes.
    if fill_holes:
        binary = ndi.binary_fill_holes(binary).astype(np.uint8)

    # 5) Smooth and rebinarize.
    smoothed = gaussian_filter(binary.astype(float), sigma=1.5)
    final = (smoothed > 0.5).astype(np.float32)

    return final

def save_smooth_mesh(voxel_data, out_path, level=0.4):
    """
    Convert a voxel grid to a smooth OBJ mesh.

    Input:
        voxel_data: 3D voxel array.
        out_path: Output OBJ path.
        level: Marching cubes threshold.

    Output:
        None. Writes OBJ file to disk.
    """
    try:
        # Upsample before marching cubes for smoother surface.
        smooth_vol = zoom(voxel_data, 3.0, order=1) 
        verts, faces, normals, values = measure.marching_cubes(smooth_vol, level=level)

        ensure_dir(os.path.dirname(out_path))
        with open(out_path, 'w') as f:
            for v in verts:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    except RuntimeError:
        print(f"[WARN] Cannot generate mesh for empty volume: {out_path}")
    except ValueError:
        print(f"[WARN] Surface level not within volume data range.")

def ensure_dir(p: str):
    """
    Create a folder path when it does not exist.

    Input:
        p: Folder path.

    Output:
        None.
    """
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def volume_to_masks4(volume_bin: np.ndarray):
    """
    Build four binary 2D masks (F, T, SL, SR) from a 3D volume.

    Input:
        volume_bin: Binary 3D voxel array with shape (D, H, W).

    Output:
        Tuple of four binary 2D masks:
            - F: Front projection
            - T: Top projection
            - SL: Side-left projection
            - SR: Side-right projection
    """
    assert volume_bin.ndim == 3
    D, H, W = volume_bin.shape
    
    def _resize(arr, shape):
        im = Image.fromarray((arr * 255).astype(np.uint8))
        im = im.resize((shape[1], shape[0]), resample=Image.NEAREST)
        return (np.array(im) > 128).astype(np.uint8)

    F = np.max(volume_bin, axis=0).astype(np.uint8)
    T = np.max(volume_bin, axis=1).astype(np.uint8)
    if T.shape != (H, W): T = _resize(T, (H, W))
    
    SL = np.max(volume_bin, axis=2).astype(np.uint8)
    if SL.shape != (H, H): SL = _resize(SL, (H, H))
    
    flipped = np.flip(volume_bin, axis=2)
    SR = np.max(flipped, axis=2).astype(np.uint8)
    if SR.shape != (H, H): SR = _resize(SR, (H, H))
    
    return F, T, SL, SR

class NU:
    """
    Utility helper class for weight init, device move, and checkpoint save.

    Input:
        Tensors/modules/config values passed to each static method.

    Output:
        Updated tensors or saved checkpoint files.
    """
    @staticmethod
    def init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    @staticmethod
    def var_or_cuda(x):
        if torch.cuda.is_available():
            return x.cuda()
        return x

    @staticmethod
    def save_checkpoints(cfg, path, epoch, encoder, encoder_solver, decoder, decoder_solver, 
                        refiner, refiner_solver, merger, merger_solver, best_iou, best_epoch):
        state = {
            'epoch_idx': epoch,
            'best_iou': best_iou,
            'best_epoch': best_epoch,
            'encoder_state_dict': encoder.state_dict(),
            'encoder_solver_state_dict': encoder_solver.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'decoder_solver_state_dict': decoder_solver.state_dict(),
        }
        if cfg.NETWORK.USE_REFINER:
            state['refiner_state_dict'] = refiner.state_dict()
            state['refiner_solver_state_dict'] = refiner_solver.state_dict()
        if cfg.NETWORK.USE_MERGER:
            state['merger_state_dict'] = merger.state_dict()
            state['merger_solver_state_dict'] = merger_solver.state_dict()
        torch.save(state, path)

# ==============================================================================
# DATASET (Strictly 6 Images: 3L + 3R per Folder)
# ==============================================================================
class BreastDataset(torch.utils.data.Dataset):
    """
    Dataset for paired multi-view breast images and 2D masks.

    Input:
        cfg: Global config object.
        mode: Split name, usually 'train' or 'valid'.
        transform: Optional image transform pipeline.
        data_root: Dataset root path.

    Output:
        For each sample: (sample_id, images_tensor, masks_tensor).
    """
    def __init__(self, cfg, mode='train', transform=None, data_root=None):
        self.cfg = cfg
        self.transform = transform
        if data_root is None:
            data_root = getattr(cfg.DIR, "DATA_ROOT", "/content/drive/MyDrive/NewModel/Dataset/Single Breast")
        self.raw_root = os.path.join(data_root, "raw_data", mode)
        self.gt_root = os.path.join(data_root, "gt_data", "2D", mode)
        
        # Scan folder ids
        folder_ids = sorted([d for d in os.listdir(self.raw_root) 
                             if os.path.isdir(os.path.join(self.raw_root, d))])
        
        # Split each folder into two samples: Left (LS) and Right (RS)
        self.samples = []
        for fid in folder_ids:
            # Add LS sample
            self.samples.append({"id": fid, "side": "LS"})
            # Add RS sample
            self.samples.append({"id": fid, "side": "RS"})
        
        print(f"[DATA] Found {len(folder_ids)} folders. Created {len(self.samples)} samples (L+R split) in '{mode}' split.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        folder_id = item["id"]
        side = item["side"]  # "LS" or "RS"
        
        raw_dir = os.path.join(self.raw_root, folder_id)
        gt_dir = os.path.join(self.gt_root, folder_id)

        # 1) Resolve file names for this side.
        if side == "LS":
            views_map = {"F": "F_LS.png", "T": "T_LS.png", "Side": "SL_LS.png"}
        else: # RS
            views_map = {"F": "F_RS.png", "T": "T_RS.png", "Side": "SR_RS.png"}

        images = []
        masks_dict = {}

        # 2) Read image + mask, then align object center at image center.
        for v_key in ["F", "T", "Side"]:
            fname = views_map[v_key]
            
            # Read original RGB image.
            img_path = os.path.join(raw_dir, fname)
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            
            if img is None:
                img = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                # Normalize to RGB channel order.
                if img.ndim == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif img.shape[2] == 4: img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
                elif img.shape[2] == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
            # Read original GT mask.
            mask_path = os.path.join(gt_dir, fname)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if mask is None:
                # Fallback mask when file is missing.
                mask = np.zeros(img.shape[:2], dtype=np.uint8)
                
            # Key alignment step for all views.
            aligned_img, aligned_mask = normalize_scale_and_center(img, mask)
            
            # Store aligned image.
            images.append(aligned_img)
            
            # Resize aligned mask to 64x64 and normalize to [0, 1].
            mask_64 = cv2.resize(aligned_mask, (64, 64), interpolation=cv2.INTER_NEAREST) / 255.0
            masks_dict[v_key] = mask_64

        # 3) Apply transform pipeline to images.
        if self.transform:
            images = self.transform(images) 

        # 4) Build 4-view mask tensor.
        mask_F = masks_dict["F"]
        mask_T = masks_dict["T"]
        mask_Side = masks_dict["Side"]

        # Keep left/right side convention consistent.
        if side == "LS":
            mask_SL = mask_Side              
            mask_SR = np.fliplr(mask_Side)   
        else: # RS
            mask_SL = np.fliplr(mask_Side)  
            mask_SR = mask_Side             

        masks_np = np.stack([mask_F, mask_T, mask_SL, mask_SR], axis=0).astype(np.float32)
        masks_tensor = torch.from_numpy(masks_np)

        # Unique ID for tracking: e.g., "001_LS"
        sample_id_str = f"{folder_id}_{side}"
        
        return sample_id_str, images, masks_tensor
# ==============================================================================
# MODELS
# ==============================================================================
class Encoder(nn.Module):
    """
    2D feature encoder based on pretrained VGG16-BN.

    Input:
        rendering_images: Tensor [B, N, C, H, W].

    Output:
        Encoded view features [B, N, 256, 8, 8].
    """
    def __init__(self, cfg):
        super(Encoder, self).__init__()
        vgg16_bn = torchvision.models.vgg16_bn(pretrained=True)
        self.vgg = nn.Sequential(*list(vgg16_bn.features.children()))[:27]
        self.layer1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ELU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=3)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ELU()
        )
        for param in vgg16_bn.parameters():
            param.requires_grad = False

    def forward(self, rendering_images):
        B, N, C, H, W = rendering_images.size()
        rendering_images = rendering_images.view(B * N, C, H, W)
        features = self.vgg(rendering_images)
        features = self.layer1(features)
        features = self.layer2(features)
        features = self.layer3(features)
        return features.view(B, N, 256, 8, 8)

class Decoder(nn.Module):
    """
    Decode per-view 2D features into coarse 3D volumes.

    Input:
        image_features: Tensor [B, N, 256, 8, 8].

    Output:
        raw_features: Tensor with merge features per view.
        gen_volumes: Coarse volumes [B, N, D, H, W].
    """
    def __init__(self, cfg):
        super(Decoder, self).__init__()
        self.cfg = cfg
        
        # Start from [B, 2048, 2, 2, 2], then upsample step by step.
        self.layer1 = nn.Sequential(
            nn.ConvTranspose3d(2048, 512, kernel_size=4, stride=2, bias=False, padding=1),
            nn.BatchNorm3d(512), nn.ReLU()
        ) # -> 4^3
        self.layer2 = nn.Sequential(
            nn.ConvTranspose3d(512, 128, kernel_size=4, stride=2, bias=False, padding=1),
            nn.BatchNorm3d(128), nn.ReLU()
        ) # -> 8^3
        self.layer3 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, bias=False, padding=1),
            nn.BatchNorm3d(64), nn.ReLU()
        ) # -> 16^3
        self.layer4 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, bias=False, padding=1),
            nn.BatchNorm3d(32), nn.ReLU()
        ) # -> 32^3
        
        # Added layer: upsample from 32 to 64.
        self.layer5 = nn.Sequential(
            nn.ConvTranspose3d(32, 8, kernel_size=4, stride=2, bias=False, padding=1),
            nn.BatchNorm3d(8), nn.ReLU()
        ) # -> 64^3
        
        # Final probability output layer.
        self.layer6 = nn.Sequential(
            nn.ConvTranspose3d(8, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, image_features):
        image_features = image_features.permute(1, 0, 2, 3, 4).contiguous() 
        gen_volumes = []
        raw_features = []
        
        for features in image_features:
            gen_volume = features.view(-1, 2048, 2, 2, 2)
            gen_volume = self.layer1(gen_volume)
            gen_volume = self.layer2(gen_volume)
            gen_volume = self.layer3(gen_volume)
            gen_volume = self.layer4(gen_volume)
            gen_volume = self.layer5(gen_volume)
            
            raw_feature = gen_volume  # Feature map (B, 8, 64, 64, 64)
            
            gen_volume = self.layer6(gen_volume)  # Probability (B, 1, 64, 64, 64)
            
            raw_feature = torch.cat((raw_feature, gen_volume), dim=1) # (B, 9, 64, 64, 64)
            
            gen_volumes.append(torch.squeeze(gen_volume, dim=1))
            raw_features.append(raw_feature)

        gen_volumes = torch.stack(gen_volumes).permute(1, 0, 2, 3, 4).contiguous()
        raw_features = torch.stack(raw_features).permute(1, 0, 2, 3, 4, 5).contiguous()
        return raw_features, gen_volumes
    
class Merger(nn.Module):
    """
    Merge multi-view coarse volumes with learned per-voxel weights.

    Input:
        raw_features: Per-view features from decoder.
        coarse_volumes: Per-view coarse volumes.

    Output:
        One merged coarse volume per sample.
    """
    def __init__(self, cfg):
        super(Merger, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv3d(9, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16), nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.layer2 = nn.Sequential(
            nn.Conv3d(16, 8, kernel_size=3, padding=1),
            nn.BatchNorm3d(8), nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.layer3 = nn.Sequential(
            nn.Conv3d(8, 4, kernel_size=3, padding=1),
            nn.BatchNorm3d(4), nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.layer4 = nn.Sequential(
            nn.Conv3d(4, 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(2), nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.layer5 = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=3, padding=1),
            nn.BatchNorm3d(1), nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )

    def forward(self, raw_features, coarse_volumes):
        n_views = coarse_volumes.size(1)
        raw_features = torch.split(raw_features, 1, dim=1)
        volume_weights = []
        for i in range(n_views):
            raw_feature = torch.squeeze(raw_features[i], dim=1)
            w = self.layer1(raw_feature)
            w = self.layer2(w)
            w = self.layer3(w)
            w = self.layer4(w)
            w = self.layer5(w)
            w = torch.squeeze(w, dim=1)
            volume_weights.append(w)
            
        volume_weights = torch.stack(volume_weights).permute(1, 0, 2, 3, 4).contiguous()
        volume_weights = torch.softmax(volume_weights, dim=1)
        coarse_volumes = coarse_volumes * volume_weights
        coarse_volumes = torch.sum(coarse_volumes, dim=1)
        return torch.clamp(coarse_volumes, min=0, max=1)

class Refiner(nn.Module):
    """
    Refine merged coarse 3D volume to better local shape quality.

    Input:
        coarse_volumes: Coarse volume tensor.

    Output:
        Refined volume tensor with same spatial size.
    """
    def __init__(self, cfg):
        super(Refiner, self).__init__()
        self.cfg = cfg
        N_VOX = cfg.CONST.N_VOX
        
        # Calculate bottleneck size dynamically
        # 3 downsample layers (stride 2) means factor of 8
        # 32 -> 4, 64 -> 8
        self.feature_size = N_VOX // 8 
        self.flatten_dim = 128 * (self.feature_size ** 3)

        # Layer Definition
        self.layer1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=4, padding=2),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE),
            nn.MaxPool3d(kernel_size=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=4, padding=2),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE),
            nn.MaxPool3d(kernel_size=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=4, padding=2),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE),
            nn.MaxPool3d(kernel_size=2)
        )
        
        # === Dynamic Linear Layer ===
        self.layer4 = nn.Sequential(
            nn.Linear(self.flatten_dim, 2048), 
            nn.ReLU()
        )
        self.layer5 = nn.Sequential(
            nn.Linear(2048, self.flatten_dim),
            nn.ReLU()
        )
        
        self.layer6 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            nn.BatchNorm3d(64), nn.ReLU()
        )
        self.layer7 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            nn.BatchNorm3d(32), nn.ReLU()
        )
        self.layer8 = nn.Sequential(
            nn.ConvTranspose3d(32, 1, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            nn.Sigmoid()
        )

    def forward(self, coarse_volumes):
        N_VOX = self.cfg.CONST.N_VOX
        f_sz = self.feature_size
        
        volumes_32_l = coarse_volumes.view((-1, 1, N_VOX, N_VOX, N_VOX))
        
        volumes_16_l = self.layer1(volumes_32_l)
        volumes_8_l = self.layer2(volumes_16_l)
        volumes_4_l = self.layer3(volumes_8_l)
        
        # Flatten
        flatten_features = self.layer4(volumes_4_l.view(-1, self.flatten_dim))
        flatten_features = self.layer5(flatten_features)
        
        # Reshape back to feature grid (e.g., 8x8x8 for N_VOX=64)
        volumes_4_r = volumes_4_l + flatten_features.view(-1, 128, f_sz, f_sz, f_sz)
        
        volumes_8_r = volumes_8_l + self.layer6(volumes_4_r)
        volumes_16_r = volumes_16_l + self.layer7(volumes_8_r)
        volumes_32_r = (volumes_32_l + self.layer8(volumes_16_r)) * 0.5
        
        return volumes_32_r.view((-1, N_VOX, N_VOX, N_VOX))

# ==============================================================================
# TRAIN & TEST LOGIC
# ==============================================================================
def project_volume(volumes, axis):
    """
    Compute max projection along one axis.

    Input:
        volumes: 4D/5D volume tensor.
        axis: Axis index for projection.

    Output:
        Projected tensor.
    """
    return torch.max(volumes, dim=axis)[0]

class TVLoss(nn.Module):
    """
    Total variation loss for smooth 3D predictions.

    Input:
        x: Predicted volume tensor.

    Output:
        Scalar smoothness loss.
    """
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size(2)
        w_x = x.size(3)
        d_x = x.size(4)
        
        # Differences along three axes (H, W, D).
        count_h = self._tensor_size(x[:,:,1:,:,:])
        count_w = self._tensor_size(x[:,:,:,1:,:])
        count_d = self._tensor_size(x[:,:,:,:,1:])
        
        # Penalize sharp local jumps between neighbor voxels.
        h_tv = torch.pow((x[:,:,1:,:,:]-x[:,:,:h_x-1,:,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:,:]-x[:,:,:,:w_x-1,:]),2).sum()
        d_tv = torch.pow((x[:,:,:,:,1:]-x[:,:,:,:,:d_x-1]),2).sum()
        
        return self.TVLoss_weight * 2 * (h_tv/count_h + w_tv/count_w + d_tv/count_d) / batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]*t.size()[4]
def test_net(cfg, epoch_idx=-1, output_dir=None, test_data_loader=None, test_writer=None,
             encoder=None, decoder=None, refiner=None, merger=None):
    """
    Run validation/testing and return mean 2D IoU score.

    Input:
        cfg: Global config object.
        epoch_idx: Epoch index for output file naming.
        output_dir: Optional output folder.
        test_data_loader: Optional external loader.
        test_writer: Optional logger (not required).
        encoder/decoder/refiner/merger: Optional pre-built models.

    Output:
        overall_mean: Mean IoU across selected front/top/side views.
    """
    torch.backends.cudnn.benchmark = True
    # 1. DataLoader Setup
    if test_data_loader is None:
        tf = DataTransforms.Compose([
            DataTransforms.CenterCrop((cfg.CONST.IMG_H, cfg.CONST.IMG_W), (cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W)),
            DataTransforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
            DataTransforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
            DataTransforms.ToTensor(),
        ])
        dataset = BreastDataset(cfg, mode='test', transform=tf, data_root=cfg.DIR.DATA_ROOT)
        test_data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    # 2. Model Setup
    if encoder is None:
        encoder = Encoder(cfg); decoder = Decoder(cfg); refiner = Refiner(cfg); merger = Merger(cfg)
        if torch.cuda.is_available():
            encoder = nn.DataParallel(encoder).cuda()
            decoder = nn.DataParallel(decoder).cuda()
            refiner = nn.DataParallel(refiner).cuda()
            merger = nn.DataParallel(merger).cuda()
        
        if cfg.CONST.WEIGHTS:
            ckpt = torch.load(cfg.CONST.WEIGHTS)
            encoder.load_state_dict(ckpt['encoder_state_dict'])
            decoder.load_state_dict(ckpt['decoder_state_dict'])
            if cfg.NETWORK.USE_REFINER: refiner.load_state_dict(ckpt['refiner_state_dict'])
            if cfg.NETWORK.USE_MERGER: merger.load_state_dict(ckpt['merger_state_dict'])

    encoder.eval(); decoder.eval(); refiner.eval(); merger.eval()

    # 3. Output Directory
    output_dir = output_dir or os.path.join(cfg.DIR.OUT_PATH, dt.now().isoformat())
    vis_dir = os.path.join(output_dir, 'images')
    vis3d_dir = vis_dir
    ensure_dir(vis_dir)
    ensure_dir(os.path.join(vis3d_dir, '3d'))
    ensure_dir(os.path.join(vis3d_dir, 'cmp'))
    
    # 4) Visualization sample selection logic.
    target_indices = set()
    dataset = test_data_loader.dataset
    
    # Prefer one fixed case for stable visual checks.
    mandatory_id = "65"
    mandatory_side = "RS"
    mandatory_idx = -1
    
    # Locate index for the mandatory sample if present.
    if hasattr(dataset, 'samples'):
        for idx, item in enumerate(dataset.samples):
            if str(item["id"]) == mandatory_id and item["side"] == mandatory_side:
                mandatory_idx = idx
                break
    
    if mandatory_idx != -1:
        target_indices.add(mandatory_idx)
    
    # Fill up to 4 visualized samples.
    pool = list(range(len(dataset)))
    if mandatory_idx in pool: pool.remove(mandatory_idx)
    needed = 4 - len(target_indices)
    if len(pool) >= needed:
        picks = random.sample(pool, needed)
        target_indices.update(picks)
    else:
        target_indices.update(pool)
        
    print(f"[TEST] Visualizing indices: {target_indices}")

    metric_names = ["F", "T", "Side"] 
    ious = {v: [] for v in metric_names}
    
    view_names_all = ["F", "T", "SL", "SR"]  # Name order for projection dict.
    
    def iou_2d(p, g):
        inter = np.logical_and(p>0, g>0).sum()
        union = np.logical_or(p>0, g>0).sum()
        return inter / (union + 1e-8)

    def stitch_comparison(pred_dict, gt_masks, view_list, title, scale=10):
        """
        Build one visual panel (GT vs prediction), then upscale.
        """
        rows = []
        pad = 2
        
        for vn in view_list:
            idx = view_names_all.index(vn)
            gt = gt_masks[idx]
            pred = pred_dict[vn]
            
            # 1) Basic pair panel: GT | Pred.
            h, w = gt.shape
            canvas = np.zeros((h, w*2 + pad), dtype=np.uint8)
            canvas[:, :w] = (gt * 255).astype(np.uint8)
            canvas[:, w+pad:] = (pred * 255).astype(np.uint8)
            
            rows.append(canvas)
            
        # 2) Stack all rows vertically.
        full_img = np.concatenate(rows, axis=0) # Vertically
        
        # 3) Upscale for easy viewing.
        H, W = full_img.shape
        new_H, new_W = int(H * scale), int(W * scale)
        full_img_big = cv2.resize(full_img, (new_W, new_H), interpolation=cv2.INTER_NEAREST)
        
        return full_img_big

    for i, (sample_id, imgs, gt_masks) in enumerate(test_data_loader):
        sample_id = sample_id[0]
        file_prefix = f"ep{epoch_idx}_{sample_id}" if epoch_idx >= 0 else f"{sample_id}"
        imgs = NU.var_or_cuda(imgs)
        gt_masks = gt_masks.squeeze(0).numpy().astype(np.uint8)

        do_viz = (i in target_indices)

        with torch.no_grad():
            feat = encoder(imgs)
            raw, gv = decoder(feat)
            if cfg.NETWORK.USE_MERGER: gv = merger(raw, gv)
            else: gv = torch.mean(gv, dim=1)
            
            if cfg.NETWORK.USE_REFINER: gv = refiner(gv)
            
            # 1) Get predicted voxel volume.
            raw_vol_prob = gv.detach().cpu().numpy()[0] 
            if raw_vol_prob.ndim == 4: raw_vol_prob = raw_vol_prob[0]

            # 2) Post-process volume.
            clean_vol = post_process_voxel(raw_vol_prob, threshold=0.4, remove_outliers=True, fill_holes=True)

            # Optional export for selected samples only.
            if do_viz:
                print(f"[TEST] Visualizing sample {sample_id}...")
                
                # Save OBJ mesh.
                save_smooth_mesh(raw_vol_prob, os.path.join(output_dir, f'{file_prefix}_clean.obj'), level=0.4)


            # Calc 2D IoU
            preds = volume_to_masks4(clean_vol)
            pred_dict = dict(zip(view_names_all, preds))
            
            if "RS" in str(sample_id):
                # Align right-side view orientation with GT convention.
                pred_SL_flipped = np.fliplr(pred_dict["SL"])
                pred_SR_flipped = np.fliplr(pred_dict["SR"])
                pred_dict["SL"] = pred_SR_flipped
                pred_dict["SR"] = pred_SL_flipped
                
            # Detect ground-truth side from sample id.
            is_left = "LS" in str(sample_id)
            
            s_f = iou_2d(pred_dict["F"], gt_masks[0])
            ious["F"].append(s_f)
            
            s_t = iou_2d(pred_dict["T"], gt_masks[1])
            ious["T"].append(s_t)
            
            # Side view uses SL for left and SR for right.
            if is_left:
                s_side = iou_2d(pred_dict["SL"], gt_masks[2])
            else:
                s_side = iou_2d(pred_dict["SR"], gt_masks[3])
            ious["Side"].append(s_side)
            
            # Per-sample mean IoU from 3 active views.
            curr_sample_iou = (s_f + s_t + s_side) / 3.0
            
            # Save comparison image for selected sample.
            if do_viz:
                side_view = "SR" if "RS" in str(sample_id) else "SL"
                views_to_show = ["F", "T", side_view]
                combined_img = stitch_comparison(pred_dict, gt_masks, views_to_show, sample_id, scale=10)
                out_path = os.path.join(vis3d_dir, 'cmp', f'{file_prefix}_combined.png')
                cv2.imwrite(out_path, combined_img)

            if do_viz or i % 5 == 0:
                print(f"[TEST] {sample_id} IoU: {curr_sample_iou}")

    print("\n=== Test Results (3 Views Only) ===")
    mean_f = np.mean(ious["F"]) if ious["F"] else 0
    mean_t = np.mean(ious["T"]) if ious["T"] else 0
    mean_s = np.mean(ious["Side"]) if ious["Side"] else 0
    
    # Merge all valid view scores for overall mean.
    all_scores = ious["F"] + ious["T"] + ious["Side"]
    overall_mean = np.mean(all_scores) if all_scores else 0
    
    print(f"F: {mean_f:.4f}")
    print(f"T: {mean_t:.4f}")
    print(f"S: {mean_s:.4f}")
    print(f"Overall: {overall_mean:.4f}")
    
    return overall_mean

# ==============================================================================
# Losses used by training
# ==============================================================================

class OccupancyLoss(nn.Module):
    """
    Encourage enough occupied voxels in the predicted shape.

    Input:
        volume: Predicted voxel tensor.

    Output:
        Scalar density loss.
    """
    def __init__(self, target_density=0.18):
        super(OccupancyLoss, self).__init__()
        self.target_density = target_density

    def forward(self, volume):
        if volume.ndim == 5:
            volume = volume.squeeze(1)
        actual_density = torch.mean(volume)
        density_loss = F.relu(self.target_density - actual_density)
        return density_loss * 8.0


class ConnectivityLoss(nn.Module):
    """
    Encourage local voxel connectivity to avoid isolated fragments.

    Input:
        volume: Predicted voxel tensor.

    Output:
        Scalar connectivity loss.
    """
    def __init__(self):
        super(ConnectivityLoss, self).__init__()

    def forward(self, volume):
        if volume.ndim == 5:
            volume = volume.squeeze(1)
        pooled = F.avg_pool3d(volume.unsqueeze(1), kernel_size=5, stride=1, padding=2).squeeze(1)
        connectivity = torch.mean((volume - pooled) ** 2)
        return connectivity * 3.0


class Solid3DLoss(nn.Module):
    """
    Encourage solid interior regions instead of hollow structures.

    Input:
        volume: Predicted voxel tensor.

    Output:
        Scalar filling loss.
    """
    def __init__(self):
        super(Solid3DLoss, self).__init__()

    def forward(self, volume):
        if volume.ndim == 5:
            volume = volume.squeeze(1)
        local_density = F.avg_pool3d(volume.unsqueeze(1), kernel_size=7, stride=1, padding=3).squeeze(1)
        should_be_filled = (local_density > 0.3).float()
        is_filled = (volume > 0.5).float()
        filling_loss = torch.mean((should_be_filled - is_filled) ** 2)
        return filling_loss * 8.0


class DensityGradientPenalty(nn.Module):
    """
    Penalize sharp layer-like transitions in 3D density.

    Input:
        volume: Predicted voxel tensor.

    Output:
        Scalar gradient penalty loss.
    """
    def __init__(self):
        super(DensityGradientPenalty, self).__init__()

    def forward(self, volume):
        if volume.ndim == 5:
            volume = volume.squeeze(1)
        grad_x = volume[:, 1:, :, :] - volume[:, :-1, :, :]
        grad_y = volume[:, :, 1:, :] - volume[:, :, :-1, :]
        grad_z = volume[:, :, :, 1:] - volume[:, :, :, :-1]
        grad2_x = grad_x[:, 1:, :, :] - grad_x[:, :-1, :, :]
        grad2_y = grad_y[:, :, 1:, :] - grad_y[:, :, :-1, :]
        grad2_z = grad_z[:, :, :, 1:] - grad_z[:, :, :, :-1]
        penalty = torch.mean(torch.abs(grad2_x)) + torch.mean(torch.abs(grad2_y)) + torch.mean(torch.abs(grad2_z))
        return penalty * 5.0


class AntiLayeringLoss(nn.Module):
    """
    Reduce thin-layer artifacts by balancing active layers across axes.

    Input:
        volume: Predicted voxel tensor.

    Output:
        Scalar imbalance loss.
    """
    def __init__(self):
        super(AntiLayeringLoss, self).__init__()

    def forward(self, volume):
        if volume.ndim == 5:
            volume = volume.squeeze(1)
        x_layers = torch.sum(volume, dim=[2, 3])
        y_layers = torch.sum(volume, dim=[1, 3])
        z_layers = torch.sum(volume, dim=[1, 2])
        x_active = (x_layers > 0.1).float().sum(dim=1).mean()
        y_active = (y_layers > 0.1).float().sum(dim=1).mean()
        z_active = (z_layers > 0.1).float().sum(dim=1).mean()
        layer_imbalance = torch.var(torch.stack([x_active, y_active, z_active]))
        return layer_imbalance * 10.0


class AntiFragmentLoss(nn.Module):
    """
    Penalize active voxels that have very low local support.

    Input:
        volume: Predicted voxel tensor.

    Output:
        Scalar fragment penalty loss.
    """
    def __init__(self, neighbor_threshold=0.3):
        super(AntiFragmentLoss, self).__init__()
        self.neighbor_threshold = neighbor_threshold

    def forward(self, volume):
        if volume.ndim == 5:
            volume = volume.squeeze(1)
        local_density = F.avg_pool3d(volume.unsqueeze(1), kernel_size=5, stride=1, padding=2).squeeze(1)
        is_active = (volume > 0.5).float()
        is_isolated = (local_density < self.neighbor_threshold).float()
        fragment_penalty = torch.mean(is_active * is_isolated)
        return fragment_penalty * 10.0

def generate_3d_gt(masks):
    """
    Build dense 3D supervision from 2D masks.

    Input:
        masks: Tensor [B, 4, 64, 64] in order F, T, SL, SR.

    Output:
        gt_3d: Tensor [B, 64, 64, 64].
    """
    # Front mask expands along depth axis.
    m_F = masks[:, 0].unsqueeze(1).expand(-1, 64, -1, -1)
    
    # Top mask expands along height axis.
    m_T = masks[:, 1].unsqueeze(2).expand(-1, -1, 64, -1)
    
    # Side-left mask expands along width axis.
    m_SL = masks[:, 2].unsqueeze(3).expand(-1, -1, -1, 64)
    
    # Intersection keeps voxels supported by all three views.
    gt_3d = m_F * m_T * m_SL
    return gt_3d

class DiceLoss(nn.Module):
    """
    Dice overlap loss for binary masks/volumes.

    Input:
        pred: Prediction tensor.
        target: Ground-truth tensor.

    Output:
        Scalar dice loss value.
    """
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        return 1.0 - (2. * intersection + self.smooth) / (union + self.smooth)
def train_net(cfg):
    """
    Train the 2D-to-3D model and run validation each epoch.

    Input:
        cfg: Global config object.

    Output:
        None. Saves checkpoints, logs, and plots to output folder.
    """
    torch.backends.cudnn.benchmark = True
    
    # Transforms
    train_tf = DataTransforms.Compose([
        DataTransforms.RandomCrop((cfg.CONST.IMG_H, cfg.CONST.IMG_W), (cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W)),
        DataTransforms.RandomBackground(cfg.TRAIN.RANDOM_BG_COLOR_RANGE),
        DataTransforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD),
        DataTransforms.RandomFlip(),
        DataTransforms.RandomPermuteRGB(),
        DataTransforms.ToTensor(),
    ])
    val_tf = DataTransforms.Compose([
        DataTransforms.CenterCrop((cfg.CONST.IMG_H, cfg.CONST.IMG_W), (cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W)),
        DataTransforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
        DataTransforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD),
        DataTransforms.ToTensor(),
    ])

    train_ds = BreastDataset(cfg, 'train', train_tf, data_root=cfg.DIR.DATA_ROOT)
    val_ds = BreastDataset(cfg, 'valid', val_tf, data_root=cfg.DIR.DATA_ROOT)
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg.CONST.BATCH_SIZE, shuffle=True, 
                                             num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False)

    encoder = Encoder(cfg); decoder = Decoder(cfg); refiner = Refiner(cfg); merger = Merger(cfg)
    encoder.apply(NU.init_weights); decoder.apply(NU.init_weights); refiner.apply(NU.init_weights); merger.apply(NU.init_weights)
    
    if torch.cuda.is_available():
        encoder = nn.DataParallel(encoder).cuda()
        decoder = nn.DataParallel(decoder).cuda()
        refiner = nn.DataParallel(refiner).cuda()
        merger = nn.DataParallel(merger).cuda()

    enc_opt = torch.optim.Adam(encoder.parameters(), lr=cfg.TRAIN.ENCODER_LEARNING_RATE)
    dec_opt = torch.optim.Adam(decoder.parameters(), lr=cfg.TRAIN.DECODER_LEARNING_RATE)
    ref_opt = torch.optim.Adam(refiner.parameters(), lr=cfg.TRAIN.REFINER_LEARNING_RATE)
    mrg_opt = torch.optim.Adam(merger.parameters(), lr=cfg.TRAIN.MERGER_LEARNING_RATE)

    out_dir = os.path.join(cfg.DIR.OUT_PATH, dt.now().isoformat())
    ensure_dir(os.path.join(out_dir, 'checkpoints'))
    
    writer = SummaryWriter(os.path.join(out_dir, 'logs'))

    best_iou = 0

    epoch_iou_history = []
    
    # Initialize all loss terms.
    bce_loss = nn.BCELoss()
    l1_loss = nn.L1Loss()
    dice_loss = DiceLoss()
    tv_criterion = TVLoss(TVLoss_weight=0.3)
    
    # Core 3D regularization terms.
    anti_layering_criterion = AntiLayeringLoss()
    density_gradient_criterion = DensityGradientPenalty()
    solid_3d_criterion = Solid3DLoss()
    
    # Additional structure losses.
    occupancy_criterion = OccupancyLoss(target_density=0.25)
    connectivity_criterion = ConnectivityLoss()
    anti_fragment_criterion = AntiFragmentLoss(neighbor_threshold=0.3)
    
    print("\n" + "="*80)
    print("Training setup: anti-layering and solidity constraints enabled.")
    print("="*80)
    print("Main losses:")
    print("  1. AntiLayeringLoss")
    print("  2. DensityGradientPenalty")
    print("  3. Solid3DLoss")
    print("Extra losses:")
    print("  - OccupancyLoss")
    print("  - ConnectivityLoss")
    print("="*80 + "\n")

    for epoch in range(cfg.TRAIN.NUM_EPOCHES):
        encoder.train(); decoder.train(); merger.train(); refiner.train()
        
        # Three-stage schedule for stable optimization.
        if epoch < 5:
            # Stage 1 (0-4): build stable global structure.
            weight_bce = 15.0
            weight_tv = 0.05
            weight_anti_layering = 1.0
            weight_density_gradient = 0.0
            weight_solid_3d = 0.5
            weight_occupancy = 0.0
            weight_connectivity = 0.0
            weight_anti_fragment = 0.0
            stage_name = "Stage1: Basic Structure Construction"
            
        elif epoch < 10:
            # Stage 2 (5-9): increase shape smoothness.
            weight_bce = 12.0
            weight_tv = 0.05
            weight_anti_layering = 0.5
            weight_density_gradient = 0.0  
            weight_solid_3d = 0.5
            weight_occupancy = 0.0
            weight_anti_fragment = 0.0
            weight_connectivity = 0.0
            stage_name = "Stage2: Edge Carving and Smoothing"
            
        else:
            # Stage 3 (10+): final surface polish.
            weight_bce = 10.0              
            weight_tv = 0.1
            weight_anti_layering = 0.0
            weight_density_gradient = 0.0
            weight_solid_3d = 0.2
            weight_occupancy = 0.0
            weight_connectivity = 0.0
            weight_anti_fragment = 0.0
            stage_name = "Stage3: Final Surface Polishing"
        
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}: {stage_name}")
        print(f"   BCE={weight_bce} | AntiLayer={weight_anti_layering} | "
              f"DensGrad={weight_density_gradient} | Solid3D={weight_solid_3d} | "
              f"Occ={weight_occupancy} | Conn={weight_connectivity}")
        print(f"{'='*80}\n")
        
        for i, (sid, imgs, masks) in enumerate(train_loader):
            imgs = NU.var_or_cuda(imgs)
            masks = NU.var_or_cuda(masks)
            
            feats = encoder(imgs)
            raw, gv = decoder(feats)
            
            if cfg.NETWORK.USE_MERGER: 
                gv = merger(raw, gv)
            else: 
                gv = torch.mean(gv, dim=1)

            gt_3d = generate_3d_gt(masks)
            # Projection loss between predicted views and input masks.
            proj_F = project_volume(gv, 1)
            proj_T = project_volume(gv, 2)
            proj_SL = project_volume(gv, 3)
            proj_SR = torch.flip(proj_SL, dims=[-1])
            
            projs = torch.stack([proj_F, proj_T, proj_SL, proj_SR], dim=1)
            loss_proj = bce_loss(projs, masks) * 10.0 + l1_loss(projs, masks) * 10.0 + dice_loss(projs, masks) * 15.0
            # 3D volume loss against pseudo-3D supervision from masks.
            loss_3d = bce_loss(gv, gt_3d) * 1.0 + dice_loss(gv, gt_3d) * 2.0
    
            # Base loss for encoder/decoder branch.
            loss_enc = loss_proj + loss_3d
            loss = loss_enc
          
            loss_ref = torch.tensor(0.0)

            # Optional refiner branch for better shape detail.
            if cfg.NETWORK.USE_REFINER:
                gv_ref = refiner(gv)
                rp_F = rp_T = project_volume(gv_ref, 1)
                rp_T = project_volume(gv_ref, 2)
                rp_SL = project_volume(gv_ref, 3)
                rp_SR = torch.flip(rp_SL, dims=[-1])
                r_projs = torch.stack([rp_F, rp_T, rp_SL, rp_SR], dim=1)
                loss_ref_proj = bce_loss(r_projs, masks) * 10.0 + l1_loss(r_projs, masks) * 10.0 + dice_loss(r_projs, masks) * 15.0
                loss_ref_3d = bce_loss(gv_ref, gt_3d) * 1.0 + dice_loss(gv_ref, gt_3d) * 2.0
                
                loss_ref = loss_ref_proj + loss_ref_3d
                loss += loss_ref
                target_volume = gv_ref
                vis_projs = r_projs
            else:
                target_volume = gv
                vis_projs = projs

            # Total variation loss for local smoothness.
            if target_volume.ndim == 4:
                target_volume_5d = target_volume.unsqueeze(1)
            else:
                target_volume_5d = target_volume
            
            loss_tv = tv_criterion(target_volume_5d) * weight_tv
            loss += loss_tv
            
            # Core 3D constraints.
            
            # 1) Reduce layer-like artifacts.
            loss_anti_layering = anti_layering_criterion(target_volume) * weight_anti_layering
            loss += loss_anti_layering
            
            # 2) Penalize weak density transitions.
            loss_density_gradient = density_gradient_criterion(target_volume) * weight_density_gradient
            loss += loss_density_gradient
            
            # 3) Encourage solid interior voxels.
            loss_solid_3d = solid_3d_criterion(target_volume) * weight_solid_3d
            loss += loss_solid_3d
            
            # 4) Keep a minimum target occupancy.
            loss_occupancy = occupancy_criterion(target_volume) * weight_occupancy
            loss += loss_occupancy
            
            # 5) Encourage connected structures.
            loss_connectivity = connectivity_criterion(target_volume) * weight_connectivity
            loss += loss_connectivity

            loss_anti_fragment = anti_fragment_criterion(target_volume) * weight_anti_fragment
            loss += loss_anti_fragment

            # Backpropagation and optimizer step.
            enc_opt.zero_grad()
            dec_opt.zero_grad()
            mrg_opt.zero_grad()
            ref_opt.zero_grad()
            
            loss.backward()
            
            enc_opt.step()
            dec_opt.step()
            mrg_opt.step()
            ref_opt.step()

            # Batch monitoring every 5 iterations.
            if i % 5 == 0:
                with torch.no_grad():
                    # Projection IoU at threshold 0.5.
                    pred_binary = (vis_projs > 0.5).float()
                    gt_binary = (masks > 0.5).float()
                    intersection = torch.sum(pred_binary * gt_binary)
                    union = torch.sum(pred_binary + gt_binary - pred_binary * gt_binary)
                    proj_iou = (intersection / (union + 1e-6)).item()
                    
                    # Mean voxel density.
                    voxel_density = torch.mean(target_volume).item()
                    
                    # Simple count of active layers along Y axis.
                    vol_np = target_volume[0].detach().cpu().numpy()
                    y_profile = vol_np.sum(axis=(0, 2))
                    y_active_layers = (y_profile > 0.1).sum()
                
                print(f"  [Batch {i:2d}] Loss: {loss.item():.3f} | "
                      f"BCE: {loss_enc.item():.3f} | "
                      f"AntiLayer: {loss_anti_layering.item():.3f} | "
                      f"Solid3D: {loss_solid_3d.item():.3f} | "
                      f"Occ: {loss_occupancy.item():.3f} | "
                      f"ProjIoU={proj_iou:.3f} | "
                      f"VoxDens={voxel_density:.3f} | "
                      f"Y_Layers={y_active_layers}/64")
            
            # TensorBoard logs for loss tracking.
            global_step = epoch * len(train_loader) + i
            writer.add_scalar('Loss/Total', loss.item(), global_step)
            writer.add_scalar('Loss/BCE', loss_enc.item(), global_step)
            writer.add_scalar('Loss/TV', loss_tv.item(), global_step)
            writer.add_scalar('Loss/AntiLayering', loss_anti_layering.item(), global_step)
            writer.add_scalar('Loss/DensityGradient', loss_density_gradient.item(), global_step)
            writer.add_scalar('Loss/Solid3D', loss_solid_3d.item(), global_step)
            writer.add_scalar('Loss/Occupancy', loss_occupancy.item(), global_step)
            writer.add_scalar('Loss/Connectivity', loss_connectivity.item(), global_step)
            
            with torch.no_grad():
                writer.add_scalar('Metrics/VoxelDensity', voxel_density, global_step)

        # Validation at the end of each epoch.
        print(f"\n{'='*70}")
        print(f"Running validation for epoch {epoch}...")
        print(f"{'='*70}\n")
        
        iou = test_net(cfg, epoch, out_dir, val_loader, 
                      encoder=encoder, decoder=decoder, 
                      refiner=refiner, merger=merger)
        
        epoch_iou_history.append(iou)

        # Save IoU-vs-epoch curve.
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(epoch_iou_history)), epoch_iou_history, 
                marker='o', linestyle='-', color='b', linewidth=2, markersize=8)
        plt.axhline(y=max(epoch_iou_history), color='r', linestyle='--', 
                   label=f'Best: {max(epoch_iou_history):.4f}')
        plt.title(f"Overall IoU vs Epoch", fontsize=16, fontweight='bold')
        plt.xlabel("Epoch", fontsize=14)
        plt.ylabel("IoU", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "iou_curve.png"), dpi=150)
        plt.close()

        # Save best checkpoint by IoU.
        if iou > best_iou:
            best_iou = iou
            NU.save_checkpoints(cfg, os.path.join(out_dir, 'checkpoints', f'ep{epoch}_best.pth'), 
                              epoch, encoder, enc_opt, decoder, dec_opt, 
            refiner, ref_opt, merger, mrg_opt, best_iou, epoch)
            print(f"\n{'='*70}")
            print(f"New best model: IoU={best_iou:.4f} at epoch {epoch}")
            print(f"{'='*70}\n")
        else:
            print(f"\nEpoch {epoch} IoU: {iou:.4f} (Best: {best_iou:.4f})\n")
    
    print("\n" + "="*70)
    print("Training finished.")
    print(f"Best IoU: {best_iou:.4f}")
    print(f"Output directory: {out_dir}")
    print("="*70 + "\n")


# ==============================================================================
# MAIN
# ==============================================================================
def parse_cli_args():
    """
    Parse command-line options for this script.

    Input:
        None. Reads values from the command line.

    Output:
        args: Parsed arguments with fields:
            - mode: 'train' or 'test'
            - weights: Optional checkpoint path
            - data_root: Dataset root folder path
    """
    parser = argparse.ArgumentParser(
        description="2D-to-3D reconstruction pipeline (train or test)."
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='train',
        choices=['train', 'test'],
        help="Run mode: 'train' to train model, 'test' to run validation/export."
    )
    parser.add_argument(
        '--weights',
        type=str,
        default=None,
        help="Optional checkpoint path to load before running."
    )
    parser.add_argument(
        '--data_root',
        type=str,
        default='/content/drive/MyDrive/NewModel/Dataset/Single Breast',
        help="Root folder that contains train/valid sample data."
    )
    return parser.parse_args()


def main():
    """
    Entry point for command-line execution.

    Input:
        Command-line arguments from parse_cli_args().

    Output:
        Runs training or testing workflow and writes outputs to disk.
    """
    # 1) Read CLI arguments.
    args = parse_cli_args()

    # 2) Apply user-provided runtime overrides.
    cfg.DIR.DATA_ROOT = args.data_root
    if args.weights: cfg.CONST.WEIGHTS = args.weights
    
    # 3) Execute selected workflow.
    print(f"Running in {args.mode} mode...")
    if args.mode == 'train':
        train_net(cfg)
    else:
        test_net(cfg)


if __name__ == "__main__":
    main()
