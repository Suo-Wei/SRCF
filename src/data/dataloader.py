# -*- coding: utf-8 -*-
"""
@author: zifyloo
"""

from torchvision import transforms
from PIL import Image
import torch
from data.dataset import CUHKPEDEDataset, CUHKPEDE_img_dateset, CUHKPEDE_txt_dateset


def get_dataloader(opt):
    """
    tranforms the image, downloads the image with the id by data.DataLoader
    """

    if opt.mode == 'train':
        transform_list = [
            transforms.RandomHorizontalFlip(),
            transforms.Resize((384, 128), Image.BICUBIC),   # interpolation
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))]
        tran = transforms.Compose(transform_list)

        dataset = CUHKPEDEDataset(opt,tran)

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                                 shuffle=True, drop_last=True, num_workers=12)
        print('{}-{} has {} pohtos'.format(opt.dataset, opt.mode, len(dataset)))

        return dataloader

    else:
        tran = transforms.Compose([
            transforms.Resize((384, 128), Image.BICUBIC),  # interpolation
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))]
        )

        img_dataset = CUHKPEDE_img_dateset(opt,tran)

        img_dataloader = torch.utils.data.DataLoader(img_dataset, batch_size=opt.batch_size,
                                                 shuffle=False, drop_last=False, num_workers=12)

        txt_dataset = CUHKPEDE_txt_dateset(opt)

        txt_dataloader = torch.utils.data.DataLoader(txt_dataset, batch_size=opt.batch_size,
                                                 shuffle=False, drop_last=False, num_workers=12)

        print('{}-{} has {} pohtos, {} text'.format(opt.dataset, opt.mode, len(img_dataset), len(txt_dataset)))

        return img_dataloader, txt_dataloader

def collate_fn4train(batch):
    imgs = []
    label = []
    label_swap = []
    law_swap = []
    caption = []
    caption_mask = []
    caption_cr = []
    caption_cr_mask =[]

    for sample in batch:
        imgs.append(sample[0])
        imgs.append(sample[1])
        label.append(sample[2])
        label.append(sample[2])

        label_swap.append(sample[2])
        label_swap.append(sample[3])

        law_swap.append(sample[4])
        law_swap.append(sample[5])
        # img_name.append(sample[-1])
        caption.append(sample[6])
        caption_mask.append(sample[7])
        caption_cr.append(sample[8])
        caption_cr_mask.append(sample[9])
    return torch.stack(imgs, 0), label, label_swap, law_swap, caption,caption_mask,caption_cr,caption_cr_mask