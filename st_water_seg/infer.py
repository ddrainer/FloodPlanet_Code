import os
import argparse

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from einops import rearrange
from scipy.special import softmax
from torch.utils.data import DataLoader

from st_water_seg.models import build_model
from st_water_seg.tools import load_cfg_file
from st_water_seg.datasets.utils import generate_image_slice_object
from st_water_seg.utils.utils_image import ImageStitcher_v2 as ImageStitcher
from st_water_seg.datasets import build_dataset, tensors_and_lists_collate_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_path',
                        type=str,
                        help='Path to the checkpoint file')
    parser.add_argument('dataset_name', type=str, help='Name of the dataset')
    parser.add_argument('split', type=str, help='Split of the dataset')
    parser.add_argument('--n_workers',
                        type=int,
                        default=None,
                        help='Number of workers for the data loader')
    parser.add_argument('region_name',
                        type=str,
                        default='CraigMissouri',
                        help='Directory to save predictions to')
    parser.add_argument('timestamp',
                        type=str,
                        default='Pre_flood',
                        help='Directory to save predictions to')
    args = parser.parse_args()

    # Load configuration file.
    experiment_dir = '/'.join(args.checkpoint_path.split('/')[:-2])
    # cfg_path = os.path.join(experiment_dir, '.hydra', 'config.yaml')
    cfg_path = os.path.join(experiment_dir, 'config.yaml')
    cfg = load_cfg_file(cfg_path)

    ## Update config parameters.
    if args.n_workers is None:
        cfg.n_workers = cfg.n_workers
    else:
        cfg.n_workers = args.n_workers

    if hasattr(cfg, 'seed_num') is False:
        cfg.seed_num = None

    if hasattr(cfg, 'train_split_pct') is False:
        cfg.train_split_pct = 0.0

    # Create save directory.
    base_save_dir = os.path.join(
        '/media/mule/Projects/SocialPixelLab/RGV_inference/2018_extra/' +
        args.region_name, args.timestamp)

    # Load dataset.
    slice_params = generate_image_slice_object(
        cfg.crop_height, cfg.crop_width, min(cfg.crop_height, cfg.crop_width))
    eval_dataset = build_dataset(args.dataset_name,
                                 args.split,
                                 slice_params,
                                 sensor=cfg.dataset.sensor,
                                 channels=cfg.dataset.channels,
                                 norm_mode=cfg.norm_mode,
                                 eval_region=cfg.eval_region,
                                 ignore_index=cfg.ignore_index,
                                 seed_num=cfg.seed_num,
                                 train_split_pct=cfg.train_split_pct,
                                 output_metadata=True,
                                 **cfg.dataset.dataset_kwargs)

    eval_loader = DataLoader(eval_dataset,
                             batch_size=cfg.batch_size,
                             shuffle=False,
                             num_workers=cfg.n_workers,
                             collate_fn=tensors_and_lists_collate_fn)

    # Load model.
    model = build_model(cfg.model.name,
                        eval_dataset.n_channels,
                        eval_dataset.n_classes,
                        cfg.lr,
                        log_image_iter=cfg.log_image_iter,
                        to_rgb_fcn=eval_dataset.to_RGB,
                        ignore_index=eval_dataset.ignore_index,
                        **cfg.model.model_kwargs)

    # Load weights.
    model = model.load_from_checkpoint(args.checkpoint_path,
                                       in_channels=eval_dataset.n_channels,
                                       n_classes=eval_dataset.n_classes,
                                       lr=cfg.lr)
    model._set_model_to_eval()

    # Get device.
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    model = model.to(device)

    # Generate predictions on target dataset.
    # rgb_canvases, pred_canvases = {}, {}
    pred_canvases = {}
    with torch.no_grad():
        for batch in tqdm(eval_loader,
                          colour='green',
                          desc='Generating predictions'):
            # Move batch to device.
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)

            # Generate predictions.
            output = model(batch).detach().cpu().numpy()
            preds = softmax(output, axis=1)

            input_images = batch['image'].detach().cpu().numpy()

            preds = rearrange(preds, 'b c h w -> b h w c')
            input_images = rearrange(input_images, 'b c h w -> b h w c')
            batch_mean = rearrange(
                batch['mean'], 'b c 1 1 -> b 1 1 c').detach().cpu().numpy()
            batch_std = rearrange(batch['std'],
                                  'b c 1 1 -> b 1 1 c').detach().cpu().numpy()

            for b in range(output.shape[0]):
                pred = preds[b]
                metadata = batch['metadata'][b]
                input_image = input_images[b]
                region_name = metadata['region_name']

                # Check if image stitcher exists for this region.
                # if region_name not in rgb_canvases.keys():
                if region_name not in pred_canvases.keys():
                    # Get base save directories.
                    # rgb_save_dir = os.path.join(base_save_dir, region_name + '_rgb')
                    pred_save_dir = os.path.join(base_save_dir,
                                                 region_name + '_pred')

                    # Initialize image stitchers.
                    # rgb_canvases[region_name] = ImageStitcher(rgb_save_dir, save_backend='PIL', save_ext='.png')
                    pred_canvases[region_name] = ImageStitcher(
                        pred_save_dir,
                        save_backend='tifffile',
                        save_ext='.tif')

                # Add input image and prediction to stitchers.
                unnorm_img = (input_image * batch_std[b]) + batch_mean[b]
                image_name = os.path.splitext(
                    os.path.split(metadata['image_path'])[1])[0]
                # rgb_canvases[region_name].add_image(unnorm_img, image_name, metadata['crop_params'], metadata['crop_params'].og_height, metadata['crop_params'].og_width)
                pred_canvases[region_name].add_image(
                    pred, image_name, metadata['crop_params'],
                    metadata['crop_params'].og_height,
                    metadata['crop_params'].og_width)

    # Convert stitched images to proper format.
    # for region_name in rgb_canvases.keys():
    for region_name in pred_canvases.keys():
        # Combine images.
        # rgb_canvas = rgb_canvases[region_name].get_combined_images()
        pred_canvas = pred_canvases[region_name].get_combined_images()

        # for image_name, image in rgb_canvas.items():
        #     # Convert to RGB.
        #     image = rearrange(image, 'h w c -> c h w')
        #     rgb_image = eval_dataset.to_RGB(image)
        #     save_path = os.path.join(rgb_canvases[region_name].save_dir, image_name + '.png')
        #     Image.fromarray((rgb_image*255).astype('uint8')).save(save_path)

        for image_name, image in pred_canvas.items():
            # Figure out the predicted class.
            pred = np.clip(image.argmax(axis=2), 0, 1)
            save_path = os.path.join(pred_canvases[region_name].save_dir,
                                     image_name + '.tif')
            Image.fromarray((pred * 255).astype('uint8')).save(save_path)


if __name__ == '__main__':
    main()
