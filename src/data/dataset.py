# -*- coding: utf-8 -*-
"""
@author: zifyloo
"""

import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import os
from utils.read_write_data import read_dict
from transforms import transforms
import cv2
# import torchvision.transforms.functional as F
import random
import re
from pytorch_pretrained_bert.tokenization import BertTokenizer
from PIL import ImageStat
import copy

def fliplr(img, dim):
    """
    flip horizontal
    :param img:
    :return:
    """
    inv_idx = torch.arange(img.size(dim) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(dim, inv_idx)
    return img_flip

def read_examples(input_line, unique_id):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    # unique_id = 0
    line = input_line #reader.readline()
    # if not line:
    #     break
    line = line.strip()
    text_a = None
    text_b = None
    m = re.match(r"^(.*) \|\|\| (.*)$", line)
    if m is None:
        text_a = line
    else:
        text_a = m.group(1)
        text_b = m.group(2)
    examples.append(
        InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
    # unique_id += 1
    return examples

class InputExample(object):
    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids

def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length
        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features


def load_data_transformers(resize_reso, crop_reso, swap_num=(12, 4)):
    center_resize = 600
    Normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    data_transforms = {
       	'swap': transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            # transforms.Resize((384, 128), Image.BICUBIC),  # interpolation
            # transforms.RandomRotation(degrees=15),
            # transforms.RandomCrop((crop_reso[0], crop_reso[1])),
            transforms.Randomswap((swap_num[0], swap_num[1])),
        ]),
        'common_aug': transforms.Compose([
            # transforms.Resize((resize_reso[0], resize_reso[1]),Image.BICUBIC),
            # transforms.RandomHorizontalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((384, 128), Image.BICUBIC),  # interpolation
            transforms.RandomRotation(degrees=15),
            transforms.RandomCrop((crop_reso[0], crop_reso[1])),
            # transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        'train_totensor': transforms.Compose([
            # transforms.Resize((crop_reso[0], crop_reso[1]),Image.BICUBIC),
            # ImageNetPolicy(),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

        ]),
        'test_totensor': transforms.Compose([
            transforms.Resize((crop_reso[0], crop_reso[1]),Image.BICUBIC),
            transforms.CenterCrop((crop_reso[0], crop_reso[1])),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]),
        'None': None,
    }
    return data_transforms

class CUHKPEDEDataset(data.Dataset):
    def __init__(self, opt,tran):

        self.opt = opt
        self.flip_flag = (self.opt.mode == 'train')

        data_save = read_dict(os.path.join(opt.dataroot, 'processed_data', opt.mode + '_save.pkl'))

        self.img_path = [os.path.join(opt.dataroot, img_path) for img_path in data_save['img_path']]

        self.label = data_save['id']

        self.caption_code = data_save['lstm_caption_id']

        self.same_id_index = data_save['same_id_index']

        self.caption = data_save['captions']

        self.transform = tran

        self.num_data = len(self.img_path)

        self.tokenizer = BertTokenizer.from_pretrained('saved_models/bert-base-uncased-vocab.txt', do_lower_case=True)

        self.transformers = load_data_transformers([384,128], [384,128], [4,6])

        self.swap_size = [4,6]
    def crop_image(self, image, cropnum):
        width, high = image.size
        crop_x = [int((width / cropnum[0]) * i) for i in range(cropnum[0] + 1)]
        crop_y = [int((high / cropnum[1]) * i) for i in range(cropnum[1] + 1)]
        im_list = []
        for j in range(len(crop_y) - 1):
            for i in range(len(crop_x) - 1):
                im_list.append(image.crop((crop_x[i], crop_y[j], min(crop_x[i + 1], width), min(crop_y[j + 1], high))))
        return im_list

    def __getitem__(self, index):
        """
        :param index:
        :return: image and its label
        """
        image = Image.open(self.img_path[index])
        img_unswaps = self.transformers['common_aug'](image)
        img_unswaps = self.transformers["train_totensor"](img_unswaps)
        label = torch.from_numpy(np.array([self.label[index]],dtype='int32')).long()

        phrase = self.caption[index]
        examples = read_examples(phrase, index)
        features = convert_examples_to_features(
            examples=examples, seq_length=65, tokenizer=self.tokenizer)
        caption_code = features[0].input_ids
        caption_length = features[0].input_mask

        same_id_index = np.random.randint(len(self.same_id_index[index]))
        same_id_index = self.same_id_index[index][same_id_index]
        phrase = self.caption[same_id_index]
        examples = read_examples(phrase, index)
        features = convert_examples_to_features(
            examples=examples, seq_length=65, tokenizer=self.tokenizer)
        same_id_caption_code = features[0].input_ids
        same_id_caption_length = features[0].input_mask

        return img_unswaps, label, np.array(caption_code,dtype=int), np.array(caption_length,dtype=int), \
               np.array(same_id_caption_code,dtype=int), np.array(same_id_caption_length,dtype=int)

    def get_data(self, index, img=True):
        if img:
            image = Image.open(self.img_path[index])
            image = self.transform(image)
        else:
            image = 0

        label = torch.from_numpy(np.array([self.label[index]])).long()

        caption_code, caption_length = self.caption_mask(self.caption_code[index])

        return image, label, caption_code, caption_length

    def caption_mask(self, caption):
        caption_length = len(caption)
        caption = torch.from_numpy(np.array(caption)).view(-1).long()

        if caption_length < self.opt.caption_length_max:
            zero_padding = torch.zeros(self.opt.caption_length_max - caption_length).long()
            caption = torch.cat([caption, zero_padding], 0)
        else:
            caption = caption[:self.opt.caption_length_max]
            caption_length = self.opt.caption_length_max

        return caption, caption_length

    def __len__(self):
        return self.num_data


class CUHKPEDE_img_dateset(data.Dataset):
    def __init__(self, opt,tran):

        self.opt = opt
        if opt.mode=='train':
            path = 'dataset/CUHK-PEDES/processed_data/train_save.pkl'
        elif opt.mode=='test':
            path = 'dataset/CUHK-PEDES/processed_data/test_save.pkl'
        # data_save = read_dict(os.path.join(opt.dataroot, 'processed_data', opt.mode + '_save.pkl'))
        data_save = read_dict(path)

        self.img_path = [os.path.join(opt.dataroot, img_path) for img_path in data_save['img_path']]

        self.label = data_save['id']

        self.transform = tran

        self.num_data = len(self.img_path)

        self.transformers = load_data_transformers([384, 128], [384, 128], [12, 4])

    def __getitem__(self, index):
        """
        :param index:
        :return: image and its label
        """

        image = Image.open(self.img_path[index])
        image_path = self.img_path[index]
        raw_image = cv2.imread(image_path)
        # raw_image = cv2.resize(raw_image, (128, 384), interpolation=cv2.INTER_CUBIC)
        # # image = self.transform(image)
        image = self.transformers["test_totensor"](image)

        label = torch.from_numpy(np.array([self.label[index]])).long()

        return image, label

    def __len__(self):
        return self.num_data


class CUHKPEDE_txt_dateset(data.Dataset):
    def __init__(self, opt):

        self.opt = opt

        data_save = read_dict(os.path.join(opt.dataroot, 'processed_data', opt.mode + '_save.pkl'))

        self.label = data_save['caption_label']
        self.caption_code = data_save['lstm_caption_id']
        self.caption = data_save['captions']
        self.num_data = len(self.caption_code)
        self.tokenizer = BertTokenizer.from_pretrained('saved_models/bert-base-uncased-vocab.txt', do_lower_case=True)

    def __getitem__(self, index):
        """
        :param index:
        :return: image and its label
        """

        label = torch.from_numpy(np.array([self.label[index]])).long()

        # caption_code, caption_length = self.caption_mask(self.caption_code[index])
        phrase = self.caption[index]
        examples = read_examples(phrase, index)
        features = convert_examples_to_features(
            examples=examples, seq_length=65, tokenizer=self.tokenizer)
        caption_code = features[0].input_ids
        caption_length = features[0].input_mask
        fea_tokens = (features[0].tokens + ['0'] * (65 - len(features[0].tokens)))
        return label, np.array(caption_code,dtype=int), np.array(caption_length,dtype=int),fea_tokens

    def caption_mask(self, caption):
        caption_length = len(caption)
        caption = torch.from_numpy(np.array(caption)).view(-1).float()
        if caption_length < self.opt.caption_length_max:
            zero_padding = torch.zeros(self.opt.caption_length_max - caption_length)
            caption = torch.cat([caption, zero_padding], 0)
        else:
            caption = caption[:self.opt.caption_length_max]
            caption_length = self.opt.caption_length_max

        return caption, caption_length

    def __len__(self):
        return self.num_data





