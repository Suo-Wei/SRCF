# -*- coding: utf-8 -*-

import transformers as ppb
import math
from torch import nn
from torchvision import models
import torch
from torch.nn import init
from torch.nn import functional as F
from torch.autograd import Variable
from pytorch_pretrained_bert.modeling import BertModel
import numpy as np
# import keyboard
number = 0
import matplotlib.pyplot as plt
import seaborn as sns
def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1)
        init.constant_(m.bias.data, 0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal(m.weight.data, std=0.001)
        init.constant(m.bias.data, 0.0)

class conv(nn.Module):

    def __init__(self, input_dim, output_dim, relu=False, BN=False):
        super(conv, self).__init__()

        block = []
        block += [nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False)]

        if BN:
            block += [nn.BatchNorm2d(output_dim)]
        if relu:
            block += [nn.LeakyReLU(0.25, inplace=True)]

        self.block = nn.Sequential(*block)
        self.block.apply(weights_init_kaiming)

    def forward(self, x):
        x = self.block(x)
        x = x.squeeze(3).squeeze(2)
        return x

class NonLocalNet(nn.Module):
    def __init__(self, opt, dim_cut=8):
        super(NonLocalNet, self).__init__()
        self.opt = opt

        up_dim_conv = []
        part_sim_conv = []
        cur_sim_conv = []
        conv_local_att = []
        for i in range(opt.part):
            up_dim_conv.append(conv(opt.feature_length//dim_cut, 1024, relu=True, BN=True))
            part_sim_conv.append(conv(opt.feature_length, opt.feature_length // dim_cut, relu=True, BN=False))
            cur_sim_conv.append(conv(opt.feature_length, opt.feature_length // dim_cut, relu=True, BN=False))
            conv_local_att.append(conv(opt.feature_length, 512))

        self.up_dim_conv = nn.Sequential(*up_dim_conv)
        self.part_sim_conv = nn.Sequential(*part_sim_conv)
        self.cur_sim_conv = nn.Sequential(*cur_sim_conv)
        self.conv_local_att = nn.Sequential(*conv_local_att)

        self.zero_eye = (torch.eye(opt.part, opt.part) * -1e6).unsqueeze(0).to(opt.device)

        self.lambda_softmax = 1

    def forward(self, embedding):
        embedding = embedding.unsqueeze(3)
        embedding_part_sim = []
        embedding_cur_sim = []

        for i in range(self.opt.part):
            embedding_i = embedding[:, :, i, :].unsqueeze(2)

            embedding_part_sim_i = self.part_sim_conv[i](embedding_i).unsqueeze(2)#b,512
            embedding_part_sim.append(embedding_part_sim_i)

            embedding_cur_sim_i = self.cur_sim_conv[i](embedding_i).unsqueeze(2)#b,512
            embedding_cur_sim.append(embedding_cur_sim_i)

        embedding_part_sim = torch.cat(embedding_part_sim, dim=2)#b,512,6
        embedding_cur_sim = torch.cat(embedding_cur_sim, dim=2)#b,512,6

        embedding_part_sim_norm = l2norm(embedding_part_sim, dim=1)  # N*D*n
        embedding_cur_sim_norm = l2norm(embedding_cur_sim, dim=1)  # N*D*n
        self_att = torch.bmm(embedding_part_sim_norm.transpose(1, 2), embedding_cur_sim_norm) #b,6,6 # N*n*n
        self_att = self_att + self.zero_eye.repeat(self_att.size(0), 1, 1)
        self_att = F.softmax(self_att * self.lambda_softmax, dim=1) #64,6,6 # .transpose(1, 2).contiguous()
        embedding_att = torch.bmm(embedding_part_sim_norm, self_att).unsqueeze(3)#b,512,6

        embedding_att_up_dim = []
        for i in range(self.opt.part):
            embedding_att_up_dim_i = embedding_att[:, :, i, :].unsqueeze(2)
            embedding_att_up_dim_i = self.up_dim_conv[i](embedding_att_up_dim_i).unsqueeze(2)
            embedding_att_up_dim.append(embedding_att_up_dim_i)
        embedding_att_up_dim = torch.cat(embedding_att_up_dim, dim=2).unsqueeze(3)

        embedding_att = embedding + embedding_att_up_dim#cancha

        embedding_local_att = []
        for i in range(self.opt.part):
            embedding_att_i = embedding_att[:, :, i, :].unsqueeze(2)
            embedding_att_i = self.conv_local_att[i](embedding_att_i).unsqueeze(2)
            embedding_local_att.append(embedding_att_i)

        embedding_local_att = torch.cat(embedding_local_att, 2)

        return embedding_local_att.squeeze()




class ResNet_image_50(nn.Module):
    def __init__(self):
        super(ResNet_image_50, self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        resnet50.layer4[0].downsample[0].stride = (1, 1)
        resnet50.layer4[0].conv2.stride = (1, 1)
        self.base1 = nn.Sequential(
            resnet50.conv1,
            resnet50.bn1,
            resnet50.relu,
            resnet50.maxpool,
            resnet50.layer1,  # 256 64 32
        )
        self.base2 = nn.Sequential(
            resnet50.layer2,  # 512 32 16
        )
        self.base3 = nn.Sequential(
            resnet50.layer3,  # 1024 16 8
        )
        self.base4 = nn.Sequential(
            resnet50.layer4  # 2048 16 8
        )

    def forward(self, x):
        x1 = self.base1(x)
        x2 = self.base2(x1)
        x3 = self.base3(x2)
        x4 = self.base4(x3)
        return x1, x2, x3, x4

class TextImgPersonReidNet(nn.Module):
    def __init__(self, opt):
        super(TextImgPersonReidNet, self).__init__()
        self.opt = opt
        self.ImageExtract = ResNet_image_50()
        model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')
        self.text_embed = model_class.from_pretrained('bert_weight')
        self.text_embed.eval()
        self.BERT = True
        for p in self.text_embed.parameters():
            p.requires_grad = False
        self.global_avgpool = nn.AdaptiveMaxPool2d((1, 1))
        self.local_avgpool = nn.AdaptiveMaxPool2d((opt.part, 1))

        conv_local = []
        for i in range(opt.part):
            conv_local.append(conv(2048, opt.feature_length))
        self.conv_local = nn.Sequential(*conv_local)

        self.conv_global = conv(2048, opt.feature_length)
        self.conv_global_qiyu = conv(2048, opt.feature_length)

        self.non_local_net = NonLocalNet(opt, dim_cut=2)
        self.leaky_relu = nn.LeakyReLU(0.25, inplace=True)


        txt_change = []
        for i in range(self.opt.part):
            txt_change.append(nn.Linear(2048, 2048))
        self.txt_change = nn.Sequential(*txt_change)

        self.start_list_img = nn.Parameter(torch.randn(2048,2))
        self.start_list_txt = nn.Parameter(torch.randn(2048,2))


        self.model_txt = ResNet_text_50()
        self.number = 0
        self.Convmask = nn.Conv2d(2048, 1, 1, stride=1, padding=0, bias=True)
        self.max_pool = nn.AdaptiveMaxPool2d((1,1))

        visual_dictionary = torch.randn(2048, self.opt.part)
        self.register_buffer('end_img', visual_dictionary)
        nn.init.normal_(self.end_img)
        self.end_img.requires_grad=False
        txt_dictionary = torch.randn(2048, self.opt.part)
        self.register_buffer('end_txt', txt_dictionary)
        nn.init.normal_(self.end_txt)
        self.end_txt.requires_grad=False
        self.adapt_max_pool = nn.AdaptiveMaxPool2d((self.opt.part,1))
        self.adapt_max_pool1D = nn.AdaptiveMaxPool1d((1))
        self.temp = 1

    def forward(self, image, caption_id, text_length,epoch=None):
        if self.training:
            img_global, img_local, img_non_local,img_part_response,img_global_response = self.img_embedding(image,epoch=epoch)
            txt_global, txt_local, txt_non_local,txt_part_response,txt_global_response = self.txt_embedding(caption_id, text_length,epoch=epoch)
        else:
            img_global, img_local, img_non_local = self.img_embedding(image,epoch=epoch)
            txt_global, txt_local, txt_non_local = self.txt_embedding(caption_id, text_length,epoch=epoch)
        if self.training:
            return img_global, img_local, img_non_local, txt_global, txt_local, txt_non_local,\
                   img_part_response,txt_part_response,img_global_response,txt_global_response
        else:
            return img_global, img_local, img_non_local, txt_global, txt_local, txt_non_local

    def compute_global_local(self,img4,image=True,txt_lang=None,train=True,epoch=None):

        fine_tune_start = 25
        if image:
            feat_list = []
            img4_new = img4.permute(0, 2, 3, 1)
            img4_new = img4_new.flatten(1,2)
            foreground_background_response = (F.normalize(img4_new,dim=-1)*5) @ (F.normalize(self.start_list_img,dim=0)*5)
            foreground_background_response_soft = torch.softmax(foreground_background_response/self.temp,dim=-1)
            foreground_background_response_soft_mutual = F.normalize(foreground_background_response_soft,dim=1).permute(0,2,1) @ F.normalize(foreground_background_response_soft,dim=1)
            foreground = (foreground_background_response_soft[:,:,0].unsqueeze(-1))*img4_new
            foreground_max = self.adapt_max_pool(foreground.contiguous().view(self.opt.batch_size,24,8,-1).permute(0,3,1,2))
            if train and epoch<=fine_tune_start:
                part_axis = []
                for i in range(self.opt.part):
                    head = torch.sum(foreground_max[:,:,i,:].squeeze(),dim=0)/self.opt.batch_size
                    part_axis.append(head.unsqueeze(0))
                part_axis = torch.cat(part_axis,dim=0).t()
                embed = 0.99*self.end_img.detach() + 0.01*part_axis
                self.end_img = embed.detach()
                part_response = (F.normalize(foreground,dim=-1)*5) @ (F.normalize(embed,dim=0)*5)
            else:
                part_response = (F.normalize(foreground,dim=-1)*5) @ (F.normalize(self.end_img.detach(),dim=0)*5)
            part_response = F.softmax(part_response, dim=1)
            for i in range(self.opt.part):
                select = torch.sum(part_response[:, :, i].unsqueeze(-1) * foreground, dim=1)
                feat_list.append(select.unsqueeze(-1).unsqueeze(-1))
            return  feat_list,part_response,foreground_background_response_soft_mutual
        else:
            feat_list = []
            for i in range(self.opt.part):
               feat_list.append([])
            txt = img4.permute(1,0,2)
            part_mutual_list = []
            gloabl_mutual_list = []
            for j in range(img4.size(1)):
                foreground_background_response = (F.normalize(txt[j,1:txt_lang[j]-1,:], dim=-1)*5) @ (F.normalize(self.start_list_txt, dim=0)*5)
                foreground_background_response_soft = torch.softmax(foreground_background_response/self.temp,dim=-1)
                foreground_background_response_soft_mutual = F.normalize(foreground_background_response_soft,dim=0).t() @ F.normalize(foreground_background_response_soft,dim=0)
                gloabl_mutual_list.append(foreground_background_response_soft_mutual.unsqueeze(0))
                foreground = (foreground_background_response_soft[:,0].unsqueeze(-1)) * txt[j,1:txt_lang[j]-1,:]
                if train and epoch<=fine_tune_start:
                    part_axis = []
                    for i in range(self.opt.part):
                        head = self.adapt_max_pool1D(self.txt_change[i](foreground).unsqueeze(0).permute(0,2,1)).squeeze()
                        part_axis.append(head.unsqueeze(0))
                    part_axis = torch.cat(part_axis,dim=0).t()
                    weights = 0.99 + (((self.opt.batch_size - 1) / self.opt.batch_size) * 0.01)
                    embed = weights * self.end_txt.detach() + (1-weights) * part_axis
                    self.end_txt = embed.detach()
                    part_response = (F.normalize(foreground,dim=-1)*5) @ (F.normalize(embed,dim=0)*5)
                else:
                    part_response = (F.normalize(foreground,dim=-1)*5) @ (F.normalize(self.end_txt.detach(),dim=0)*5)
                part_response_soft = F.softmax(part_response, dim=0,)
                for i in range(self.opt.part):
                    select = torch.sum(part_response_soft[:, i].unsqueeze(-1) * foreground, dim=0)
                    feat_list[i].append(select.unsqueeze(0))
                part_response_soft_norm = F.normalize(part_response_soft,dim=0)
                part_mutual_list.append((part_response_soft_norm.t() @ part_response_soft_norm).unsqueeze(0))
            feat_list = [torch.cat(feat_list[i], dim=0).unsqueeze(-1) for i in range(self.opt.part)]

            if train:
                return feat_list,torch.cat(part_mutual_list,dim=0), torch.cat(gloabl_mutual_list,0)
            else:
                return feat_list,torch.cat(gloabl_mutual_list,0)

    def img_embedding(self, image,epoch=None):
        _,_,imgf3, image_feature = self.ImageExtract(image)#b,2048,12,4
        image_feature_global = self.global_avgpool(image_feature)  # b,2048,1
        image_global = self.conv_global(image_feature_global).unsqueeze(2)  # b,1024
        if self.training:
            image_feature_local,part_response,global_mutual = self.compute_global_local(torch.cat([image_feature],dim=0),image=True,train=self.training,epoch=epoch)
        else:
            image_feature_local,part_response,global_mutual = self.compute_global_local(image_feature,image=True,train=self.training)

        image_feature_local = torch.cat(image_feature_local,dim=2)
        image_local = []
        for i in range(self.opt.part):
            image_feature_local_i = image_feature_local[:, :, i, :]
            image_feature_local_i = image_feature_local_i.unsqueeze(2)
            image_embedding_local_i = self.conv_local[i](image_feature_local_i).unsqueeze(2)
            image_local.append(image_embedding_local_i)

        image_local = torch.cat(image_local, 2)#b,1024,6

        image_non_local = self.leaky_relu(image_local)
        image_non_local = self.non_local_net(image_non_local)

        if self.training:
            return image_global, image_local, image_non_local,part_response,global_mutual 
        else:
            return image_global, image_local, image_non_local

    def txt_embedding(self, caption_id, text_length,epoch=None):
        with torch.no_grad():
            txt = self.text_embed(caption_id, attention_mask=text_length)
            txt = txt[0]

        _, fword_1 = self.model_txt(txt)
        text_feature_l = fword_1.squeeze().unsqueeze(-1)
        text_length = torch.sum(text_length, dim=-1)
        text_global = self.global_avgpool(text_feature_l)  # 64,2048
        text_global = self.conv_global(text_global).unsqueeze(2)  # 64,1024
        if self.training:
            text_feature_local,part_response,global_mutual = self.compute_global_local(text_feature_l.squeeze().permute(2,0,1),image=False,txt_lang=text_length,train=self.training,epoch=epoch)
        else:
            text_feature_local,_ = self.compute_global_local(text_feature_l.squeeze().permute(2,0,1),image=False,txt_lang=text_length,train=self.training,epoch=epoch)

        text_feature_local = torch.cat(text_feature_local, dim=-1)#b,2048,6
        text_local = []
        for p in range(self.opt.part):
            text_feature_local_conv_p = text_feature_local[:, :, p].unsqueeze(2).unsqueeze(2)
            text_feature_local_conv_p = self.conv_local[p](text_feature_local_conv_p).unsqueeze(2)
            text_local.append(text_feature_local_conv_p)
        text_local = torch.cat(text_local, dim=2)#b,1024,6
        text_non_local = self.leaky_relu(text_local)
        text_non_local = self.non_local_net(text_non_local)#b,512,6
        if self.training:
            return text_global, text_local, text_non_local,part_response,global_mutual
        else:
            return text_global, text_local, text_non_local



class ResNet_text_50(nn.Module):

    def __init__(self, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet_text_50, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 768

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))


        self.conv1 = conv1x1(self.inplanes, 1024)
        self.bn1 = norm_layer(1024)
        self.relu = nn.ReLU(inplace=True)

        downsample = nn.Sequential(
            conv1x1(1024, 2048),
            norm_layer(2048),
        )

        # 3, 4, 6, 3

        self.branch1 = nn.Sequential(
            Bottleneck(inplanes=1024, planes=2048, width=512, downsample=downsample),
            Bottleneck(inplanes=2048, planes=2048, width=512),
            Bottleneck(inplanes=2048, planes=2048, width=512)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)


    def forward(self, x):
        x = x.permute(0,2,1).unsqueeze(2).contiguous()
        x1 = self.conv1(x)  # 1024 1 64
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x21 = self.branch1(x1)
        return x1, x21

class Bottleneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv1x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes)
        self.bn3 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

def conv1x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1,3), stride=stride,
                     padding=(0,1), groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
