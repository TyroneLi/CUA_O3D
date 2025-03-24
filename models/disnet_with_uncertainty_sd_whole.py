'''3D model for distillation.'''
import torch
from collections import OrderedDict
import math
from models.mink_unet_with_uncertainty_sd import mink_unet as model3D
from torch import nn


def state_dict_remove_moudle(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v
    return new_state_dict


def constructor3d(**kwargs):
    model = model3D(**kwargs)
    return model


def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)

def custom_deepcopy(model):
    model_copy = type(model)()  # Create a new instance of the model's class
    model_copy.load_state_dict(model.state_dict())  # Copy parameters and buffers
    return model_copy

class ModelEMA(nn.Module):
    """ Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """
 
    def __init__(self, model, cfg, last_dim, decay=0.9999, updates=0):
        super(ModelEMA, self).__init__()
        # Create EMA
        # self.ema = copy.deepcopy(model).eval()
        # self.ema = copy.deepcopy(model)
        # self.ema = get_model(args)
        self.ema = constructor3d(
            in_channels=3, 
            out_channels=last_dim, 
            out_channels_sd=3200, 
            D=3, 
            arch=cfg.arch_3d
        )
        self.ema.load_state_dict(model.state_dict(), strict=True)
        self.ema = self.ema.cuda()
        # self.ema = self.ema.to(torch.device("cuda"))
        self.ema.eval()
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))
        for p in self.ema.parameters():
            p.requires_grad_(False)
 
    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)
            msd = model.module.state_dict() 
            # print(msd.keys())
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd["teacher_net3d.ema." + k].detach()
 
    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)


class DisNet(nn.Module):
    '''3D Sparse UNet for Distillation.'''
    def __init__(self, cfg=None):
        super(DisNet, self).__init__()
        if not hasattr(cfg, 'feature_2d_extractor'):
            cfg.feature_2d_extractor = 'openseg'
        if 'lseg' in cfg.feature_2d_extractor:
            last_dim = 512
        elif 'openseg' in cfg.feature_2d_extractor:
            last_dim = 768
        else:
            raise NotImplementedError

        # MinkowskiNet for 3D point clouds
        net3d = constructor3d(
            in_channels=3, 
            out_channels=last_dim, 
            out_channels_sd=3200, 
            D=3, 
            arch=cfg.arch_3d)
        self.net3d = net3d
        
        teacher = ModelEMA(
            net3d,
            cfg=cfg,
            last_dim=last_dim,
            decay=0.99, 
            updates=0
        )
        
        self.teacher_net3d = teacher

    def forward(self, sparse_3d):
        '''Forward method.'''
        return self.net3d(sparse_3d)
