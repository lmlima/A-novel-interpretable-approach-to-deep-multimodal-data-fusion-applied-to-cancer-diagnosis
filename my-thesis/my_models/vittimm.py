# -*- coding: utf-8 -*-
"""
Autor: Leandro Lima
"""

from torch import nn
from metablock import MetaBlock, MetaBlockSN
from metanet import MetaNet
from mat import MutualAttentionTransformer
from MetaCARU import MetaCARU, MetaCARUsimplified, MetaCARUsimplifiedBn, MetaCARUBn
from MutualAttention import MutualAttention, MAMCARU, MAMCARUS, MAMBlock
import torch
import warnings
import timm
from densesparse import DenseSparse


class MyViTTimm(nn.Module):

    def __init__(self, vit, num_class, neurons_reducer_block=256, freeze_conv=False, p_dropout=0.5,
                 comb_method=None, comb_config=None, n_feat_conv=1024, experimental_cfg=None):  # base = 768; huge = 1280

        super(MyViTTimm, self).__init__()

        self.features = vit
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)

        if n_feat_conv is None:
            crossvit_classes = (timm.models.crossvit.CrossViT)
            if isinstance(self.features, crossvit_classes):
                n_feat_conv = sum(self.features.embed_dim)
            else:
                n_feat_conv = self.features.num_features

        self.use_SNN = experimental_cfg["embedding"]["use_SNN"]
        self.use_DS = experimental_cfg["embedding"]["use_DS"]
        if self.use_DS:
            emb_dim = experimental_cfg["embedding"]["emb_dim"] # 25
            emb_out = experimental_cfg["embedding"]["emb_out"] # 15
            cat_max_size = experimental_cfg["embedding"]["categorical_max_size"] # 15


            col_name = experimental_cfg["embedding"]["col_sparse"]
            col_name_dense = experimental_cfg["embedding"]["col_dense"]

            self.DSTransform = DenseSparse(
                len(col_name_dense), len(col_name),
                emb_dim, emb_out, cat_max_size,
                poincare_dict=experimental_cfg["poincare"],
                use_SNN=self.use_SNN
            )
            if experimental_cfg["embedding"]["use_DDS"]:
                self.DSTransform2 = DenseSparse(
                    len(col_name_dense), len(col_name),
                    emb_dim, emb_out, cat_max_size,
                    poincare_dict=experimental_cfg["poincare"],
                    use_SNN=self.use_SNN
                )
            comb_config = self.DSTransform.output_shape

        _n_meta_data = 0
        if comb_method is not None:
            if comb_config is None:
                raise Exception("You must define the comb_config since you have comb_method not None")

            if comb_method == 'metablock':
                if not isinstance(comb_config, int):
                    raise Exception("comb_config must be int for 'metablock' method")
                # comb_div = 32
                comb_div = 1
                while n_feat_conv % (comb_div) != 0:
                    comb_div -= 1

                conv_input_dim = n_feat_conv // comb_div
                if self.use_SNN:
                    self.comb = MetaBlockSN(conv_input_dim, comb_config)  # Normally (40, x)
                else:
                    self.comb = MetaBlock(conv_input_dim, comb_config)  # Normally (40, x)

                # self.comb = MetaBlockDenseSparse(conv_input_dim, comb_config)  # Normally (40, x)

                self.comb_feat_maps = conv_input_dim
                self.comb_div = comb_div
            elif comb_method == "metacaru":
                if not isinstance(comb_config, int):
                    raise Exception("comb_config must be int for 'metacaru' method")
                self.comb = MetaCARU(n_feat_conv, comb_config)  # Normally (40, x)
                self.comb_feat_maps = n_feat_conv
            elif comb_method == "metacarubn":
                if not isinstance(comb_config, int):
                    raise Exception("comb_config must be int for 'metacarubn' method")
                self.comb = MetaCARUBn(n_feat_conv, comb_config)  # Normally (40, x)
                self.comb_feat_maps = n_feat_conv
            elif comb_method == "metacarusimplified":
                if not isinstance(comb_config, int):
                    raise Exception("comb_config must be int for 'metacarusimplified' method")
                self.comb = MetaCARUsimplified(n_feat_conv, comb_config)  # Normally (40, x)
                self.comb_feat_maps = n_feat_conv
            elif comb_method == "metacarusimplifiedbn":
                if not isinstance(comb_config, int):
                    raise Exception("comb_config must be int for 'metacarusimplifiedbn' method")
                self.comb = MetaCARUsimplifiedBn(n_feat_conv, comb_config)  # Normally (40, x)
                self.comb_feat_maps = n_feat_conv
            elif comb_method == "mamcaru":
                if not isinstance(comb_config, int):
                    raise Exception("comb_config must be int for 'mamcaru' method")
                _n_meta_data = comb_config
                self.comb = MAMCARU(n_feat_conv, comb_config)
                self.comb_feat_maps = n_feat_conv
                # n_feat_conv = n_feat_conv + _n_meta_data
                _n_meta_data = 0
            elif comb_method == "mamcarus":
                if not isinstance(comb_config, int):
                    raise Exception("comb_config must be int for 'mamcarus' method")
                _n_meta_data = comb_config
                self.comb = MAMCARUS(n_feat_conv, comb_config)
                self.comb_feat_maps = n_feat_conv
                # n_feat_conv = n_feat_conv + _n_meta_data
                _n_meta_data = 0
            elif comb_method == "mamblock":
                if not isinstance(comb_config, int):
                    raise Exception("comb_config must be int for 'mamblock' method")
                _n_meta_data = comb_config
                self.comb = MAMBlock(n_feat_conv, comb_config)
                self.comb_feat_maps = n_feat_conv
                # n_feat_conv = n_feat_conv + _n_meta_data
                _n_meta_data = 0
            elif comb_method == "mutualattention":
                if not isinstance(comb_config, int):
                    raise Exception("comb_config must be int for 'mutualattention' method")
                _n_meta_data = comb_config
                self.comb = MutualAttention(n_feat_conv, comb_config)
                self.comb_feat_maps = n_feat_conv
                n_feat_conv = n_feat_conv + _n_meta_data
                _n_meta_data = 0
            elif comb_method == 'concat':
                if not isinstance(comb_config, int):
                    raise Exception("comb_config must be int for 'concat' method")
                _n_meta_data = comb_config
                self.comb = 'concat'
            elif comb_method == 'metanet':
                if not isinstance(comb_config, int):
                    raise Exception("comb_config must be int for 'metanet' method")
                comb_div = 8
                while n_feat_conv % (comb_div * comb_div) != 0:
                    comb_div -= 1

                conv_input_dim = n_feat_conv // (comb_div * comb_div)
                middle_layer = 64
                self.comb = MetaNet(comb_config, middle_layer, conv_input_dim)  # (n_meta, middle, 20)
                self.comb_feat_maps = conv_input_dim
                self.comb_div = comb_div
            elif comb_method == 'mat':
                _n_meta_data = comb_config

                _numheads = 8  # Default number of heads in MAT multi head attention
                _d_model = (n_feat_conv // _numheads) * _numheads  # d_model must be divisible by _numheads
                self.comb = MutualAttentionTransformer(_d_model, num_heads=_numheads)  # n_meta: int
                self.comb_feat_maps = comb_config
            else:
                raise Exception("There is no comb_method called " + comb_method + ". Please, check this out.")

            # if self.comb_div is not None:
            #     warnings.warn(F"comb_div = {self.comb_div}")
            # if self.comb_feat_maps is not None:
            #     warnings.warn(F"comb_feat_maps = {self.comb_feat_maps}")
        else:
            self.comb = None

        # freezing the convolution layers
        if freeze_conv:
            for param in self.features.parameters():
                param.requires_grad = False

        # Feature reducer
        if neurons_reducer_block > 0:
            self.reducer_block = nn.Sequential(
                nn.Linear(n_feat_conv, neurons_reducer_block),
                nn.BatchNorm1d(neurons_reducer_block),
                nn.ReLU(),
                nn.Dropout(p=p_dropout)
            )
        else:
            if comb_method == 'concat':
                warnings.warn("You're using concat with neurons_reducer_block=0. Make sure you're doing it right!")
            self.reducer_block = None

        if comb_method == 'mat':
            # Projection of meta_data to image features size
            self.data_proj = nn.Linear(_n_meta_data, n_feat_conv)
            # Set _n_meta_data to 0 since MAT merge those extra information in n_feat_conv.
            _n_meta_data = 0

        # Here comes the extra information (if applicable)
        if neurons_reducer_block > 0:
            self.classifier = nn.Linear(neurons_reducer_block + _n_meta_data, num_class)
        else:
            self.classifier = nn.Linear(n_feat_conv + _n_meta_data, num_class)

        # self.use_poincare = experimental_cfg["poincare"]["use_Poincare"]
        # if self.use_poincare:
        #     c = experimental_cfg["poincare"]["c"]
        #     self.toPC = hypnn.ToPoincare(c)
        #     self.fromPC = hypnn.FromPoincare(c)
        #     self.comb = MetaBlock(2*conv_input_dim, 2*comb_config)


    def forward(self, img, meta_data=None):

        # Checking if when passing the metadata, the combination method is set
        if meta_data is not None and self.comb is None:
            raise Exception("There is no combination method defined but you passed the metadata to the model!")
        if meta_data is None and self.comb is not None:
            raise Exception("You must pass meta_data since you're using a combination method!")

        # x = self.features.forward_features(img)
        # x = x if type(x) != tuple else x[0]

        x = self.backbone_fusion(img, meta_data)

        return self.classifier(x)

    def backbone_features(self, img):
        x = self.features.forward_features(img)

        # Some models return (raw_feat, pooled_feat)
        if type(x) == tuple:
            x = x[0]

        # Effnet needs avg_pooling
        avg_classes = (
            timm.models.efficientnet.EfficientNet,
            timm.models.resnet.ResNet,
            timm.models.resnetv2.ResNetV2,
            timm.models.regnet.RegNet,
            timm.models.nfnet.NormFreeNet,
            timm.models.byobnet.ByobNet,
            timm.models.tresnet.TResNet,
            timm.models.rexnet.ReXNetV1,
            timm.models.nest.Nest,
            timm.models.densenet.DenseNet,
            timm.models.vgg.VGG
        )
        if isinstance(self.features, avg_classes):
            x = self.avg_pooling(x)

        # CrossVit ([small, big])
        crossvit_classes = (timm.models.crossvit.CrossViT)
        if isinstance(self.features, crossvit_classes):
            x = torch.cat(x, 1)

        return x

    def backbone_fusion(self, img, meta_data):
        x = self.backbone_features(img)

        # if self.use_poincare:
        #     x = self.toPC(x)

        if self.comb != None:
            if self.use_DS:
                if hasattr(self, 'DSTransform2'):
                    meta_data1 = self.DSTransform(meta_data).float()
                    meta_data2 = self.DSTransform2(meta_data).float()
                    meta_data = [meta_data1, meta_data2]
                else:
                    meta_data = self.DSTransform(meta_data).float()
            else:
                meta_data = meta_data.float()


        if self.comb == None:
            x = x.view(x.size(0), -1)  # flatting
            if self.reducer_block is not None:
                x = self.reducer_block(x)  # feat reducer block
        elif self.comb == 'concat':
            x = x.view(x.size(0), -1)  # flatting
            if self.reducer_block is not None:
                x = self.reducer_block(x)  # feat reducer block. In this case, it must be defined
            x = torch.cat([x, meta_data], dim=1)  # concatenation
        elif isinstance(self.comb, (MetaBlock, MetaBlockSN)):
            # x = x.view(x.size(0), self.comb_feat_maps, 32, -1).squeeze(-1) # getting the feature maps
            # TODO: Test for diferente archtectures
            x = x.view(x.size(0), -1, self.comb_div).squeeze(-1)
            # x = x.view(x.size(0), -1, 32).squeeze(-1) # getting the feature maps

            # Make sure there is at least 3 dimensions
            if len(x.shape) < 3:
                x = x.unsqueeze(2)
            x = self.comb(x, meta_data)  # applying MetaBlock
            x = x.view(x.size(0), -1)  # flatting
            if self.reducer_block is not None:
                x = self.reducer_block(x)  # feat reducer block
        elif isinstance(self.comb, MetaNet):
            # TODO: Test for diferente archtectures a generic shape
            x = x.view(x.size(0), self.comb_feat_maps, self.comb_div, self.comb_div).squeeze(
                -1)  # getting the feature maps
            x = self.comb(x, meta_data.float())  # applying metanet
            x = x.view(x.size(0), -1)  # flatting
            if self.reducer_block is not None:
                x = self.reducer_block(x)  # feat reducer block
        elif isinstance(self.comb, MutualAttentionTransformer):
            x = x.squeeze().unsqueeze(1)
            y = self.data_proj(meta_data.float()).unsqueeze(1)

            x = self.comb(x, y)  # applying mat
            x = x.squeeze()
            if self.reducer_block is not None:
                x = self.reducer_block(x)  # feat reducer block
        elif isinstance(self.comb, (MetaCARU, MetaCARUsimplified, MetaCARUsimplifiedBn, MetaCARUBn)):
            x = x.squeeze()
            meta = meta_data.float()

            # Fix for last batch with only 1 sample
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
            x = self.comb(x, meta)  # applying metacaru
            # x = x.squeeze()
            if self.reducer_block is not None:
                x = self.reducer_block(x)  # feat reducer block
        elif isinstance(self.comb, (MAMCARU, MAMCARUS, MAMBlock)):
            x = x.squeeze()
            meta = meta_data.float()

            # Fix for last batch with only 1 sample
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
            x = self.comb(x, meta)  # applying metacaru
            # x = x.squeeze()

            if self.reducer_block is not None:
                x = self.reducer_block(x)  # feat reducer block

        elif isinstance(self.comb, MutualAttention):
            x = x.squeeze()
            meta = meta_data.float()

            # Fix for last batch with only 1 sample
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
            x = self.comb(x, meta)  # applying metacaru
            # x = x.squeeze()
            if self.reducer_block is not None:
                x = self.reducer_block(x)  # feat reducer block

        # if self.use_poincare:
        #     x = self.fromPC(x)

        return x
