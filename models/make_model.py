import torch
import torch.nn as nn
from .modeling_resnet import ResNetV2
from .modeling import VisionTransformer
from .grouping import GroupingUnit

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)



class build_transformer(nn.Module):
    def __init__(self, config, args, zero_head=True, num_classes=1000, vis=True):
        super(build_transformer, self).__init__()
            

        self.inter_arch = ResNetV2(block_units=config.resnet.num_layers,
                                         width_factor=config.resnet.width_factor)
        # perform weakly supervised semantic segmentation
        self.grouping = GroupingUnit(config.resnet.num_features, config.num_parts)
        self.grouping.reset_parameters(init_weight=None, init_smooth_factor=None)
        
        # vit
        self.vit = VisionTransformer(config, 448, zero_head=True, num_classes=num_classes, vis=True, smoothing_value=0.0)
        

    def forward(self, x, label=None):
        x = self.inter_arch(x)  # [64, 768]
        
        global_feat, assign = self.grouping(x)
        
        if label is not None:
            ce_loss, mask_ce_loss, bce_loss, logits = self.vit(global_feat, label)
            return ce_loss, mask_ce_loss, bce_loss, logits, assign
        
        else:
            logits, attn_weights = self.vit(global_feat)
            return logits, assign, attn_weights
            #return logits, assign
        # feat = self.bottleneck(global_feat)  # [64, 768]
        # cls_score = self.classifier(feat)


    def load_from(self, weights):
        self.vit.load_from(weights)
        with torch.no_grad():
            if self.vit.transformer.embeddings.hybrid:
                self.inter_arch.root.conv.weight.copy_(np2th(weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.inter_arch.root.gn.weight.copy_(gn_weight)
                self.inter_arch.root.gn.bias.copy_(gn_bias)
                for bname, block in self.inter_arch.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)
        
        

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


def make_model(config, args, zero_head=True, num_classes=1000, vis=True, model_name='transformer'):
    if model_name == 'transformer':
        model = build_transformer(config, args, zero_head=True, num_classes=num_classes, vis=True)
        print('===========building transformer===========')
    else:
        pass
        #model = Backbone(num_class, cfg)
        #print('===========building ResNet===========')
    return model


