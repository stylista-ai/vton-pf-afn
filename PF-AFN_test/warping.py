import os
import time
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms
from models.afwm import AFWM
from models.networks import load_checkpoint
from options.test_options import TestOptions
from data.cp_dataset import CPDataset

from loguru import logger


def create_dataset(opt):
    return CPDataset(
        opt.dataroot, mode=opt.phase, image_size=opt.fineSize, unpaired=opt.unpaired
    )


def setup_data_loader(opt, dataset):
    return DataLoader(
        dataset, batch_size=opt.batchSize, shuffle=False, num_workers=1, pin_memory=True
    )


def setup_model(opt, device):
    model = AFWM(opt, 3 + opt.label_nc)
    load_checkpoint(model, opt.warp_checkpoint)
    model.eval().to(device)
    return model


def process_data(data, device):
    # Process cloth and mask (input #1)
    key = "unpaired"  # Hard-code to unpaired for inference
    cloth = data["cloth"][key].to(device)
    cloth_mask = torch.FloatTensor(
        (data["cloth_mask"][key].numpy() > 0.5).astype(float)
    ).to(device)

    # Process other necessary inputs (#2)
    parse_agnostic = data["parse_agnostic"].to(device)
    densepose = data["densepose"].to(device)
    openpose = data["pose"].cuda()

    # Downsample inputs for the model
    size_downsample = (256, 192)
    pre_cloth_mask_down = F.interpolate(
        cloth_mask, size=size_downsample, mode="nearest"
    )
    input_parse_agnostic_down = F.interpolate(
        parse_agnostic, size=size_downsample, mode="nearest"
    )
    cloth_down = F.interpolate(cloth, size=size_downsample, mode="bilinear")
    densepose_down = F.interpolate(densepose, size=size_downsample, mode="bilinear")

    conditional_input = torch.cat([input_parse_agnostic_down, densepose_down], 1)

    # UNUSED - TBD REFACTOR LATER
    # GT
    label_onehot = data["parse_onehot"].cuda()  # CE
    label = data["parse"].cuda()  # GAN loss
    parse_cloth_mask = data["pcm"].cuda()  # L1
    im_c = data["parse_cloth"].cuda()  # VGG

    # visualization
    im = data["image"]
    agnostic = data["agnostic"]
    image_name = data["image_name"]

    # Prepare inputs for the warping model
    input1 = torch.cat([cloth_down, pre_cloth_mask_down], 1)

    return {
        "conditional_input": conditional_input,
        "cloth_mask": cloth_mask,
        "cloth": cloth,
        "pre_cloth_mask_down": pre_cloth_mask_down,
        "cloth_down": cloth_down,
    }


def warp_clothes(model: AFWM, inputs: dict):
    # Apply the warping model
    # The model outputs the warped cloth image and a flow field (`last_flow`)
    warped_cloth, last_flow = model(inputs["conditional_input"], inputs["cloth_down"])

    # Apply grid sampling to create the warped mask
    # `F.grid_sample` applies a spatial transformation to `pre_cloth_mask_down` using `last_flow`
    warped_mask = F.grid_sample(
        inputs["pre_cloth_mask_down"],
        last_flow.permute(0, 2, 3, 1),  # Rearrange the dimensions of `last_flow`
        mode="bilinear",  # Bilinear interpolation
        padding_mode="zeros",  # Pads with zeros where the flow goes out of bounds
    )

    # Retrieve the original cloth image and mask
    cloth = inputs["cloth"]
    cloth_mask = inputs["cloth_mask"]

    # If the height of the original cloth image is not 256, adjust the size of the flow field and warped outputs
    N, _, iH, iW = cloth.size()
    if iH != 256:
        # Resize `last_flow` to match the dimensions of the original cloth image
        last_flow = F.interpolate(last_flow, size=(iH, iW), mode="bilinear")

        # Apply grid sampling to the original cloth image and mask
        # This warps them according to the resized flow field
        warped_cloth = F.grid_sample(
            cloth,
            last_flow.permute(0, 2, 3, 1),
            mode="bilinear",
            padding_mode="border",  # Pads with border values where the flow goes out of bounds
        )
        warped_mask = F.grid_sample(
            cloth_mask,
            last_flow.permute(0, 2, 3, 1),
            mode="bilinear",
            padding_mode="zeros",
        )

    return warped_cloth, warped_mask


def save_results(warped_cloth, warped_mask, image_names, opt):
    path = os.path.join(opt.outdir, opt.name)
    mask_path = f"{path}-mask"
    os.makedirs(path, exist_ok=True)
    os.makedirs(mask_path, exist_ok=True)

    logger.debug(f"Saving results to {path}")

    for i, (cloth, mask) in enumerate(zip(warped_cloth, warped_mask)):
        save_image(cloth, f"{path}/{image_names[i]}")
        save_mask(mask, f"{mask_path}/{image_names[i]}")


def save_image(tensor, path):
    cv_img = ((tensor.permute(1, 2, 0).detach().cpu().numpy() + 1) / 2 * 255).astype(
        np.uint8
    )
    cv2.imwrite(path, cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR))


def save_mask(tensor, path):
    mask_img = (tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
    cv2.imwrite(path, mask_img)


if __name__ == "__main__":
    opt = TestOptions().parse()
    device = torch.device(f"cuda:{opt.gpu_ids[0]}")

    start_time = time.time()
    dataset = create_dataset(opt)
    logger.debug(f"Load dataset: {time.time() - start_time:.2f}s")

    start_time = time.time()
    train_loader = setup_data_loader(opt, dataset)
    logger.debug(f"Load data loader: {time.time() - start_time:.2f}s")

    start_time = time.time()
    warp_model = setup_model(opt, device)
    logger.debug(f"Load warp model: {time.time() - start_time:.2f}s")

    with torch.no_grad():
        for epoch in range(1, 2):
            for data in train_loader:
                start_time = time.time()
                processed_data = process_data(data, device)
                warped_cloth, warped_mask = warp_clothes(warp_model, processed_data)
                logger.debug(
                    f"Processing data: {time.time() - start_time:.2f}s for {len(data['image'])} items"
                )
        save_results(warped_cloth, warped_mask, data["image_name"], opt)
