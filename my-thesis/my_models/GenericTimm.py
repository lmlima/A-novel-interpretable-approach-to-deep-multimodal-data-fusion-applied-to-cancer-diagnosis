# -*- coding: utf-8 -*-
"""
Autor: Leandro Lima
Email: leandro.m.lima@ufes.br
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
from vittimm import MyViTTimm

from ffattention import FFAttentionClassifier
import torch.nn.functional as F

from densesparse import DenseSparse
import numpy as np

class GenericTimm(MyViTTimm):

    def __init__(self, vit, num_class, classifier, neurons_reducer_block=256, freeze_conv=False, p_dropout=0.5,
                 comb_method=None, comb_config=None, n_feat_conv=1024, experimental_cfg=None):  # base = 768; huge = 1280

        super(GenericTimm, self).__init__(vit, num_class, neurons_reducer_block, freeze_conv, p_dropout,
                                          comb_method, comb_config, n_feat_conv, experimental_cfg=experimental_cfg)

        freeze = experimental_cfg["late_fusion"]["freeze_backbone"]
        use_softmax = experimental_cfg["late_fusion"]["use_softmax"]

        self.features = vit
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)

        # if classifier is None:
        #     classifier = 'mlp'
        self.classifier_name = classifier

        if n_feat_conv is None:
            crossvit_classes = (timm.models.crossvit.CrossViT)
            if isinstance(self.features, crossvit_classes):
                n_feat_conv = sum(self.features.embed_dim)
            else:
                n_feat_conv = self.features.num_features

        self.use_SNN = experimental_cfg["embedding"]["use_SNN"]
        self.use_DS = experimental_cfg["embedding"]["use_DS"]
        if self.use_DS:
            emb_dim = experimental_cfg["embedding"]["emb_dim"]  # 25
            emb_out = experimental_cfg["embedding"]["emb_out"]  # 15
            cat_max_size = experimental_cfg["embedding"]["categorical_max_size"]  # 15

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
                    self.comb = MetaBlock(conv_input_dim, comb_config)  # Normally (40, x)                self.comb_feat_maps = conv_input_dim
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
            # self.classifier = nn.Linear(neurons_reducer_block + _n_meta_data, num_class)
            self.classifier = nn.Sequential(
                    nn.Linear(neurons_reducer_block + _n_meta_data, num_class),
                    # nn.Softmax(dim=1)
                )
        else:
            # self.classifier = nn.Linear(n_feat_conv + _n_meta_data, num_class)
            self.classifier = nn.Sequential(
                nn.Linear(n_feat_conv + _n_meta_data, num_class),
                # nn.Softmax(dim=1)
            )

        # freezing the convolution layers
        # features, reducer_block, avg_pooling, classifier, comb
        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False
            if self.reducer_block is not None:
                for param in self.reducer_block.parameters():
                    param.requires_grad = False
            if self.avg_pooling is not None:
                for param in self.avg_pooling.parameters():
                    param.requires_grad = False
            if self.classifier is not None:
                for param in self.classifier.parameters():
                    param.requires_grad = False
            if self.comb is not None:
                for param in self.comb.parameters():
                    param.requires_grad = False

        if use_softmax:
            self.softmax = nn.Softmax(dim=-1)
        else:
            self.softmax = None

        # Late fusion
        self.use_outter_classifier = experimental_cfg["late_fusion"]["use_outter_classifier"]

        self.late_fusion = experimental_cfg["late_fusion"]["late_fusion"]
        self.pre_fusion = experimental_cfg["late_fusion"]["pre_fusion"]
        self.late_residual = experimental_cfg["late_fusion"]["late_residual"]

        if self.late_fusion:
            comb_div = 1
            while n_feat_conv % (comb_div) != 0:
                comb_div -= 1

            conv_input_dim = n_feat_conv // comb_div
            self.late_fusion_fn = MetaBlock(num_class, comb_config)
            if self.pre_fusion:
                self.pre_fusion_fn = MetaCARUsimplifiedBn(num_class, comb_config)
            self.comb_div = comb_div
            comb_config = 0
        else:
            self.late_fusion_fn = MetaBlock(num_class, comb_config)
            if self.pre_fusion:
                self.pre_fusion_fn = MetaCARUsimplifiedBn(num_class, comb_config)
            self.comb_div = 1


        # Here comes the extra information (if applicable)
        self.outter_classifier = self.get_classifier(self.classifier_name, comb_config, num_class)


    def get_classifier(self, classifier, size, num_class):
        classifier_size = num_class + size
        if classifier == 'linear':
            return nn.Linear(classifier_size, num_class)
        elif classifier is None:
            return None
        elif classifier == 'mlp':
            hidden_size = 128
            return nn.Sequential(
                nn.Linear(classifier_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, num_class)
            )
        elif classifier == 'mha':
            return MHAClassifier(classifier_size, num_class)
        elif classifier == 'mhamlp':
            return MHAMLPClassifier(classifier_size, num_class)
        elif classifier == 'transformer':
            return TransformerClassifier(classifier_size, num_class)
        elif classifier == 'mha2':
            return MHAClassifier2(size, num_class)
        elif classifier == 'ffattention':
            batch_size = 30
            return FFAttentionClassifier(batch_size=batch_size, D_in=classifier_size, D_out=num_class)
        elif classifier == 'ca':
            return ContextualAttention(classifier_size, num_class)
        else:
            raise Exception("There is no classifier called " + classifier + ". Please, check this out.")

    # def forward(self, img, meta_data=None):
    #
    #     # Checking if when passing the metadata, the combination method is set
    #     # if meta_data is not None and self.comb is None:
    #     #     raise Exception("There is no combination method defined but you passed the metadata to the model!")
    #     if meta_data is None and self.comb is not None:
    #         raise Exception("You must pass meta_data since you're using a combination method!")
    #
    #     # x = self.features.forward_features(img)
    #     # x = x if type(x) != tuple else x[0]
    #
    #     if self.outter_classifier is None:
    #         img_class_feat = self.backbone_fusion(img, meta_data=meta_data)
    #         out = self.classifier(img_class_feat)
    #     else:
    #         img_class_feat = self.backbone_fusion(img, meta_data=None)
    #         x = self.classifier(img_class_feat)
    #
    #         if self.softmax:
    #             x = self.softmax(x)
    #
    #         # Concatenate backbone_fusion features with meta_data features
    #         if self.classifier_name == 'mha2':
    #             # x = F.softmax(x, dim=1)
    #             out = self.outter_classifier(x, meta_data)
    #             # out = self.outter_classifier(meta_data, x)
    #
    #         else:
    #             x = torch.cat((x, meta_data), dim=1)
    #
    #             out = self.outter_classifier(x)
    #
    #
    #     # out = self.outter_classifier(x)
    #     # return out if len(out) == 1 else out[0]
    #     return out

    #Forward com late late fusion - experimental
    def forward(self, img, meta_data=None):

        # Checking if when passing the metadata, the combination method is set
        # if meta_data is not None and self.comb is None:
        #     raise Exception("There is no combination method defined but you passed the metadata to the model!")
        if meta_data is None and self.comb is not None:
            raise Exception("You must pass meta_data since you're using a combination method!")

        # x = self.features.forward_features(img)
        # x = x if type(x) != tuple else x[0]

        if (self.outter_classifier is None) and (self.use_outter_classifier == True):
            img_class_feat = self.backbone_fusion(img, meta_data=meta_data)
            out = self.classifier(img_class_feat)
        else:
            img_class_feat = self.backbone_fusion(img, meta_data=None)
            x = self.classifier(img_class_feat)

            if self.softmax:
                x = self.softmax(x)

            if self.use_DS:
                meta_data = self.DSTransform(meta_data)

            # Concatenate backbone_fusion features with meta_data features
            if self.classifier_name == 'mha2':
                # x = F.softmax(x, dim=1)
                out = self.outter_classifier(x, meta_data)
                # out = self.outter_classifier(meta_data, x)

            else:
                if not self.pre_fusion:
                    x_orig = x
                    x = torch.cat((x, meta_data), dim=1)
                else:
                    x_orig = x

                    x = x.view(x.size(0), -1, self.comb_div).squeeze(-1)
                    # x = x.view(x.size(0), -1, 32).squeeze(-1) # getting the feature maps

                    # Make sure there is at least 3 dimensions, only MetaBlock
                    # if len(x.shape) < 3:
                    #     x = x.unsqueeze(2)
                    x = self.pre_fusion_fn(x, meta_data).squeeze(-1)
                    x = torch.cat((x, meta_data), dim=1)

                if self.use_outter_classifier:
                    out = self.outter_classifier(x)
                else:
                    out = x

                if self.late_residual:
                    x_lf = x_orig.view(x_orig.size(0), -1, self.comb_div).squeeze(-1)
                    # x = x.view(x.size(0), -1, 32).squeeze(-1) # getting the feature maps

                    # Make sure there is at least 3 dimensions
                    if len(x_lf.shape) < 3:
                        x_lf = x_lf.unsqueeze(2)
                    x_lf = self.late_fusion_fn(x_lf, meta_data).squeeze(-1)

                    out = out + x_lf

        # out = self.outter_classifier(x)
        # return out if len(out) == 1 else out[0]
        return out
class MHAClassifier(nn.Module):
    def __init__(self, classifier_size, num_class, hidden_size=128):
        super(MHAClassifier, self).__init__()

        self.linear1 = nn.Linear(classifier_size, hidden_size)
        self.mha = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8)
        self.linear2 = nn.Linear(hidden_size, num_class)
        self.act = nn.ReLU()

    def forward(self, x):
        # x = x.unsqueeze(0)
        # attn_output, attn_output_weights = self.mha(x, y, y) #key = value
        x = self.linear1(x)
        x = self.act(x)
        x = x.unsqueeze(0)

        attn_output, _ = self.mha(x, x, x, need_weights=False)  # key = value = metadata
        x = self.act(attn_output.squeeze(0))

        return self.linear2(x.squeeze(0))


class MHAMLPClassifier(nn.Module):
    def __init__(self, classifier_size, num_class, hidden_size=128):
        super(MHAMLPClassifier, self).__init__()

        self.linear1 = nn.Linear(classifier_size, hidden_size)
        self.mha = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8)
        self.linear2 = nn.Linear(hidden_size, num_class)
        self.act = nn.ReLU()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_class)
        )

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = x.unsqueeze(0)

        attn_output, _ = self.mha(x, x, x, need_weights=False)  # key = value = metadata
        x = self.mlp(attn_output.squeeze(0))

        return x

class TransformerClassifier(nn.Module):
    def __init__(self, classifier_size, num_class, hidden_size=128, nheads=1):
        super(TransformerClassifier, self).__init__()

        self.linear1 = nn.Linear(classifier_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_class)
        self.act = nn.ReLU()

        self.emb = nn.Linear(classifier_size, hidden_size)
        self.mha = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=nheads)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.class_output = nn.Sequential(
            nn.Linear(hidden_size, num_class),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.emb(x)
        # x = self.act(x)
        x = x.unsqueeze(0)

        z = self.ln1(x)
        attn_output, _ = self.mha(z, z, z, need_weights=False)  # key = value = metadata
        y = attn_output + x

        y = y.squeeze(0)
        mlp_output = self.mlp(self.ln2(y))
        out = mlp_output + y

        out = self.class_output(out)

        return out
    def predict_proba(self, x):
        # Ensure x is a Tensor
        if isinstance(x, np.ndarray):
            device = next(self.parameters()).device
            x = torch.from_numpy(x).float().to(device) # Convert it to a float Tensor and move to model's device
        return self.forward(x).cpu().detach().numpy()

class MHAClassifier2(nn.Module):
    def __init__(self, classifier_size, num_class, hidden_size=128):

        super(MHAClassifier2, self).__init__()

        self.linear1y = nn.Linear(classifier_size, hidden_size)
        self.linear1x = nn.Linear(num_class, hidden_size)

        self.bn = nn.BatchNorm1d(classifier_size)

        # self.mha = nn.MultiheadAttention(embed_dim=classifier_size, num_heads=1, kdim=num_class, vdim=num_class)
        # self.mha = nn.MultiheadAttention(embed_dim=num_class, num_heads=1, kdim=classifier_size, vdim=classifier_size)
        self.mha = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=1, kdim=hidden_size, vdim=hidden_size)

        self.linear2a = nn.Linear(hidden_size, hidden_size)
        # self.linear2a = nn.Linear(classifier_size, hidden_size)
        # self.linear2a = nn.Linear(num_class, hidden_size)

        self.linear2 = nn.Linear(hidden_size, num_class)
        self.act = nn.ReLU()

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_class)
        )

    def forward(self, x, y=None):
        x = x.unsqueeze(0)
        x = self.linear1x(x)
        # x = self.act(x)

        if y is None:
            y = x
        else:
            y = self.bn(y)
            y = y.unsqueeze(0)
            y = self.linear1y(y)
            # y = self.act(y)

        # x = x.unsqueeze(0)
        # attn_output, attn_output_weights = self.mha(x, y, y) #key = value
        # x = self.linear1(x)

        attn_output, _ = self.mha(x, y, y, need_weights=False)  # key = value = metadata
        out = self.mlp(attn_output.squeeze(0))
        #
        # out = self.linear2a(out)
        # out = self.act(out)
        # out = self.linear2(out)
        # # out = self.act(out)
        # # out = F.softmax(out, dim=1)
        return out


class ContextualAttention(nn.Module):
    # Contextual Attention Network: Transformer Meets U-Net - https://arxiv.org/abs/2203.01932
    def __init__(self, emb_size, num_class, hidden_size=128):
        super(ContextualAttention, self).__init__()

        self.max_len = 1
        self.emb_size = emb_size

        self.weight = nn.Parameter(torch.rand(self.emb_size, 1))
        self.bias = nn.Parameter(torch.Tensor(self.max_len, 1))

        self.mlp = nn.Sequential(
            nn.Linear(emb_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_class)
        )
    # def reset_parameters(self):
    #     nn.init.kaiming_uniform_(self.weight, a=torch.sqrt(torch.tensor([5])))
    def forward(self, x, mask=None):
        # Here    x should be [batch_size, time_step, emb_size]
        #      mask should be [batch_size, time_step, 1]
        x = x.unsqueeze(1)

        W_bs = self.weight.unsqueeze(0).repeat(x.size()[0], 1, 1)  # Copy the Attention Matrix for batch_size times
        scores = torch.bmm(x, W_bs)  # Dot product between input and attention matrix
        scores = torch.tanh(scores)

        # scores = Cal_Attention()(x, self.weight, self.bias)

        if mask is not None:
            mask = mask.long()
            scores = scores.masked_fill(mask == 0, -1e9)

        a_ = F.softmax(scores.squeeze(-1), dim=-1)
        a = a_.unsqueeze(-1).repeat(1, 1, x.size()[2])

        weighted_input = x * a

        attn_output = torch.sum(weighted_input, dim=1)

        out = self.mlp(attn_output.squeeze(0))


        return out
