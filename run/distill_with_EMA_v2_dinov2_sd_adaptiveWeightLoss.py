import os
import time
import random
import numpy as np
import logging
import argparse
import math
from copy import copy, deepcopy
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from tensorboardX import SummaryWriter

from MinkowskiEngine import SparseTensor
from util import config
from util.util import AverageMeter, intersectionAndUnionGPU, \
    poly_learning_rate, save_checkpoint, save_ema_checkpoint, \
    export_pointcloud, get_palette, convert_labels_with_palette, extract_clip_feature
from dataset.label_constants import *
from dataset.feature_loader_with_dinov2_sd import FusedFeatureLoader, collation_fn
from dataset.point_loader_with_dinov2_sd import Point3DLoader, collation_fn_eval_all
from models.disnet_with_uncertainty_dinov2_sd import DisNet as Model
from tqdm import tqdm


best_iou = 0.0
best_ema_iou = 0.0

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


def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def de_parallel(model):
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model


class ModelEMA:
    """ Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """
 
    def __init__(self, model, args, decay=0.9999, updates=0):
        # Create EMA
        # self.ema = copy.deepcopy(model).eval()
        # self.ema = copy.deepcopy(model)
        self.ema = get_model(args)
        # self.ema = deepcopy(de_parallel(model))
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
            # msd = model.state_dict()  
            msd = de_parallel(model).state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()
 
    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)
        

def worker_init_fn(worker_id):
    '''Worker initialization.'''
    random.seed(time.time() + worker_id)


def get_parser():
    '''Parse the config file.'''

    parser = argparse.ArgumentParser(description='OpenScene 3D distillation.')
    parser.add_argument('--config', type=str,
                        default='config/scannet/distill_openseg.yaml',
                        help='config file')
    parser.add_argument('opts',
                        default=None,
                        help='see config/scannet/distill_openseg.yaml for all options',
                        nargs=argparse.REMAINDER)
    args_in = parser.parse_args()
    assert args_in.config is not None
    cfg = config.load_cfg_from_cfg_file(args_in.config)
    if args_in.opts:
        cfg = config.merge_cfg_from_list(cfg, args_in.opts)
    os.makedirs(cfg.save_path, exist_ok=True)
    model_dir = os.path.join(cfg.save_path, 'model')
    result_dir = os.path.join(cfg.save_path, 'result')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(result_dir + '/last', exist_ok=True)
    os.makedirs(result_dir + '/best', exist_ok=True)
    return cfg


def get_logger():
    '''Define logger.'''

    logger_name = "main-logger"
    logger_in = logging.getLogger(logger_name)
    logger_in.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(filename)s line %(lineno)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger_in.addHandler(handler)
    return logger_in


def main_process():
    return not args.multiprocessing_distributed or (
        args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)
    # return (dist.get_rank() == 0)


def main():
    '''Main function.'''

    args = get_parser()

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(
        str(x) for x in args.train_gpu)
    cudnn.benchmark = True
    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
    
    # By default we use shared memory for training
    if not hasattr(args, 'use_shm'):
        args.use_shm = True

    print(
        'torch.__version__:%s\ntorch.version.cuda:%s\ntorch.backends.cudnn.version:%s\ntorch.backends.cudnn.enabled:%s' % (
            torch.__version__, torch.version.cuda, torch.backends.cudnn.version(), torch.backends.cudnn.enabled))

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False
        args.use_apex = False

    if args.multiprocessing_distributed:
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node,
                 args=(args.ngpus_per_node, args))
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
    global args
    global best_iou
    global best_ema_iou
    args = argss

    if args.distributed:
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, 
                                world_size=args.world_size, rank=args.rank)

    model = get_model(args)

    if main_process():
        global logger, writer
        logger = get_logger()
        writer = SummaryWriter(args.save_path)
        logger.info(args)
        logger.info("=> creating model ...")
        logger.info(model)

    # ####################### Optimizer ####################### #
    optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr)
    args.index_split = 0

    if args.distributed:
        torch.cuda.set_device(gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.batch_size_val = int(args.batch_size_val / ngpus_per_node)
        args.workers = int(args.workers / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(
            model.cuda(), device_ids=[gpu])
        
    else:
        model = model.cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(
                args.resume, map_location=lambda storage, loc: storage.cuda())
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_iou = checkpoint['best_iou']
            if main_process():
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint['epoch']))
        else:
            if main_process():
                logger.info(
                    "=> no checkpoint found at '{}'".format(args.resume))
    
    # TODO: add EMA Model
    EMA_model = ModelEMA(model.module, args, decay=0.99, updates=0)
    
    if args.resume:
        if os.path.isfile(args.resume):
            if main_process():
                logger.info("=> loading checkpoint for ema model '{}'".format(args.resume))
            checkpoint = torch.load(
                args.resume, map_location=lambda storage, loc: storage.cuda())
            EMA_model.ema.load_state_dict(checkpoint['ema_state_dict'], strict=True)
            
            EMA_model.updates = checkpoint['updates']
            
            best_ema_iou = checkpoint['best_ema_iou']
            
            if main_process():
                logger.info("=> loaded checkpoint for ema model '{}' (epoch {})".format(
                    args.resume, checkpoint['epoch']))
        else:
            if main_process():
                logger.info(
                    "=> no checkpoint found at '{}'".format(args.resume))
    ######################
    
    
    # ####################### Data Loader ####################### #
    if not hasattr(args, 'input_color'):
        # by default we do not use the point color as input
        args.input_color = False
    train_data = FusedFeatureLoader(datapath_prefix=args.data_root,
                                    datapath_prefix_feat=args.data_root_2d_fused_feature,
                                    datapath_prefix_feat_dinov2=args.data_root_2d_fused_feature_dinov2,
                                    datapath_prefix_feat_sd=args.data_root_2d_fused_feature_sd,
                                    voxel_size=args.voxel_size,
                                    split='train', aug=args.aug,
                                    memcache_init=args.use_shm, loop=args.loop,
                                    input_color=args.input_color
                                    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_data) if args.distributed else None
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                            shuffle=(train_sampler is None),
                                            num_workers=args.workers, pin_memory=True,
                                            sampler=train_sampler,
                                            drop_last=True, collate_fn=collation_fn,
                                            worker_init_fn=worker_init_fn)
    if args.evaluate:
        val_data = Point3DLoader(datapath_prefix=args.data_root,
                                 voxel_size=args.voxel_size,
                                 split='val', aug=False,
                                 memcache_init=args.use_shm,
                                 eval_all=True,
                                 input_color=args.input_color)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_data) if args.distributed else None
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val,
                                                shuffle=False,
                                                num_workers=args.workers, pin_memory=True,
                                                drop_last=False, collate_fn=collation_fn_eval_all,
                                                sampler=val_sampler)

        criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda(gpu) # for evaluation

    # ####################### Distill ####################### #
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
            if args.evaluate:
                val_sampler.set_epoch(epoch)
            
        # loss_val, mIoU_val, mAcc_val, allAcc_val = validate(
        #         val_loader, model, criterion)
        # ema_loss_val, ema_mIoU_val, ema_mAcc_val, ema_allAcc_val = validate(
        #     val_loader, EMA_model.ema, criterion)
        
        loss_train = distill(train_loader, model, optimizer, epoch, EMA_model)
        epoch_log = epoch + 1
        if main_process():
            writer.add_scalar('loss_train', loss_train, epoch_log)

        is_best = False
        if args.evaluate and (epoch_log % args.eval_freq == 0):
            loss_val, mIoU_val, mAcc_val, allAcc_val = validate(
                val_loader, model, criterion)
            ema_loss_val, ema_mIoU_val, ema_mAcc_val, ema_allAcc_val = validate(
                val_loader, EMA_model.ema, criterion)
            # raise NotImplementedError

            if main_process():
                writer.add_scalar('loss_val', loss_val, epoch_log)
                writer.add_scalar('mIoU_val', mIoU_val, epoch_log)
                writer.add_scalar('mAcc_val', mAcc_val, epoch_log)
                writer.add_scalar('allAcc_val', allAcc_val, epoch_log)
                
                writer.add_scalar('ema_loss_val', ema_loss_val, epoch_log)
                writer.add_scalar('ema_mIoU_val', ema_mIoU_val, epoch_log)
                writer.add_scalar('ema_mAcc_val', ema_mAcc_val, epoch_log)
                writer.add_scalar('ema_allAcc_val', ema_allAcc_val, epoch_log)
                
                # remember best iou and save checkpoint
                is_best = mIoU_val > best_iou
                best_iou = max(best_iou, mIoU_val)
                
                is_ema_best = ema_mIoU_val > best_ema_iou
                best_ema_iou = max(best_ema_iou, ema_mIoU_val)

        if (epoch_log % args.save_freq == 0) and main_process():
            
            save_checkpoint(
                {
                    'epoch': epoch_log,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_iou': best_iou,
                    'best_ema_iou': best_ema_iou,
                    'ema_state_dict': EMA_model.ema.state_dict(),
                    'updates': EMA_model.updates,
                }, is_best, os.path.join(args.save_path, 'model')
            )
            
            save_ema_checkpoint(
                {
                    'epoch': epoch_log,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_iou': best_iou,
                    'best_ema_iou': best_ema_iou,
                    'ema_state_dict': EMA_model.ema.state_dict(),
                    'updates': EMA_model.updates,
                }, is_ema_best, os.path.join(args.save_path, 'model')
            )
            
    if main_process():
        writer.close()
        logger.info('==>Training done!\nBest Iou: %.3f' % (best_iou))


def get_model(cfg):
    '''Get the 3D model.'''

    model = Model(cfg=cfg)
    return model

def obtain_text_features_and_palette():
    '''obtain the CLIP text feature and palette.'''

    if 'scannet' in args.data_root:
        labelset = list(SCANNET_LABELS_20)
        labelset[-1] = 'other'
        palette = get_palette()
        dataset_name = 'scannet'
    elif 'matterport' in args.data_root:
        labelset = list(MATTERPORT_LABELS_21)
        palette = get_palette(colormap='matterport')
        dataset_name = 'matterport'
    elif 'nuscenes' in args.data_root:
        labelset = list(NUSCENES_LABELS_16)
        palette = get_palette(colormap='nuscenes16')
        dataset_name = 'nuscenes'

    if not os.path.exists('saved_text_embeddings'):
        os.makedirs('saved_text_embeddings')

    if 'openseg' in args.feature_2d_extractor:
        model_name="ViT-L/14@336px"
        postfix = '_768' # the dimension of CLIP features is 768
    elif 'lseg' in args.feature_2d_extractor:
        model_name="ViT-B/32"
        postfix = '_512' # the dimension of CLIP features is 512
    else:
        raise NotImplementedError

    clip_file_name = 'saved_text_embeddings/clip_{}_labels{}.pt'.format(dataset_name, postfix)

    try: # try to load the pre-saved embedding first
        logger.info('Load pre-computed embeddings from {}'.format(clip_file_name))
        text_features = torch.load(clip_file_name).cuda()
    except: # extract CLIP text features and save them
        text_features = extract_clip_feature(labelset, model_name=model_name)
        torch.save(text_features, clip_file_name)

    return text_features, palette


def distill(train_loader, model, optimizer, epoch, EMA_model):
    '''Distillation pipeline.'''

    torch.backends.cudnn.enabled = True
    batch_time = AverageMeter()
    data_time = AverageMeter()

    loss_meter = AverageMeter()
    uncertainty_lseg_loss_meter = AverageMeter()
    uncertainty_dinov2_loss_meter = AverageMeter()
    uncertainty_sd_loss_meter = AverageMeter()
    distill_lseg_loss_meter = AverageMeter()
    distill_dinov2_loss_meter = AverageMeter()
    distill_sd_loss_meter = AverageMeter()

    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)

    text_features, palette = obtain_text_features_and_palette()

    # start the distillation process
    for i, batch_data in enumerate(train_loader):
        data_time.update(time.time() - end)

        (coords, feat, label_3d, feat_3d, sd_feat_dinov2, sd_feat_3d, mask) = batch_data
        
        ori_coords = coords.clone()
        ori_sinput = SparseTensor(
            feat.cuda(non_blocking=True), ori_coords.cuda(non_blocking=True))
        
        coords[:, 1:4] += (torch.rand(3) * 100).type_as(coords)
        sinput = SparseTensor(
            feat.cuda(non_blocking=True), coords.cuda(non_blocking=True))
        feat_3d, mask = feat_3d.cuda(
            non_blocking=True), mask.cuda(non_blocking=True)

        sd_feat_dinov2 = sd_feat_dinov2.cuda(non_blocking=True)
        
        sd_feat_3d = sd_feat_3d.cuda(non_blocking=True)

        output_3d, uncertainty_pred_lseg, output_3d_dinov2, uncertainty_pred_dinov2, output_3d_sd, uncertainty_pred_sd = model(sinput)
        
        output_3d = output_3d[mask]
        output_3d_dinov2 = output_3d_dinov2[mask]
        output_3d_sd = output_3d_sd[mask]
        # uncertainty_pred = uncertainty_pred[mask]
        
        with torch.no_grad():
            ema_output_3d, ema_uncertainty_pred_lseg, ema_output_3d_dinov2, ema_uncertainty_pred_dinov2, ema_output_3d_sd, ema_uncertainty_pred_sd = EMA_model.ema(ori_sinput)

        if hasattr(args, 'loss_type') and args.loss_type == 'cosine':
            # loss = (1 - torch.nn.CosineSimilarity()
            #         (output_3d, feat_3d)).mean()
            ##################################################
            lseg_loss = (1 - torch.nn.CosineSimilarity()
                    (output_3d, feat_3d))
            ema_cosine_loss_lseg = (1.0 - torch.nn.CosineSimilarity()(
                ema_output_3d[mask].detach(),
                feat_3d
            ))
            uncertainty_loss_lseg = torch.nn.L1Loss()(
                uncertainty_pred_lseg[mask], 
                ema_cosine_loss_lseg[..., None].detach(),
            )
            ##################################################
            dinov2_loss = (1 - torch.nn.CosineSimilarity()
                    (F.normalize(output_3d_dinov2, 2, dim=1), F.normalize(sd_feat_dinov2, 2, dim=1)))
            ema_cosine_loss_dinov2 = (1.0 - torch.nn.CosineSimilarity()(
                F.normalize(ema_output_3d_dinov2[mask].detach(), 2, dim=1),
                F.normalize(sd_feat_dinov2, 2, dim=1)
            ))
            uncertainty_loss_dinov2 = torch.nn.L1Loss()(
                uncertainty_pred_dinov2[mask], 
                ema_cosine_loss_dinov2[..., None].detach(),
            )
            ##################################################
            sd_loss = (1 - torch.nn.CosineSimilarity()
                    (F.normalize(output_3d_sd, 2, dim=1), F.normalize(sd_feat_3d, 2, dim=1)))
            ema_cosine_loss_sd = (1.0 - torch.nn.CosineSimilarity()(
                F.normalize(ema_output_3d_sd[mask].detach(), 2, dim=1),
                F.normalize(sd_feat_3d, 2, dim=1)
            ))
            uncertainty_loss_sd = torch.nn.L1Loss()(
                uncertainty_pred_sd[mask], 
                ema_cosine_loss_sd[..., None].detach(),
            )
            ###################################################
            
            cosine_loss = lseg_loss.mean() + dinov2_loss.mean() + sd_loss.mean()
            uncertainty_loss = uncertainty_loss_lseg + uncertainty_loss_dinov2 + uncertainty_loss_sd
            loss = cosine_loss + uncertainty_loss
            
        elif hasattr(args, 'loss_type') and args.loss_type == 'l1':
            # loss = torch.nn.L1Loss()(output_3d, feat_3d)
            cosine_loss = torch.nn.L1Loss(reduction='none')(output_3d, feat_3d)
            # import pdb;pdb.set_trace()
            cosine_loss = cosine_loss.mean(dim=-1)
            
            uncertainty_loss = torch.nn.L1Loss()(
                uncertainty_pred[mask], 
                cosine_loss[..., None].detach(),
            )
            
            cosine_loss = cosine_loss.mean()
            loss = cosine_loss + uncertainty_loss
        elif hasattr(args, 'loss_type') and args.loss_type == 'l2':
            # loss = torch.nn.MSELoss()(output_3d, feat_3d)
            cosine_loss = torch.nn.MSELoss(reduction='none')(output_3d.float(), feat_3d.float())
            
            cosine_loss = cosine_loss.mean(dim=-1)
            
            uncertainty_loss = torch.nn.L1Loss()(
                uncertainty_pred[mask], 
                cosine_loss[..., None].detach(),
            )
            
            cosine_loss = cosine_loss.mean()
            loss = cosine_loss + uncertainty_loss
        else:
            raise NotImplementedError
        
        # if not args.distributed:
            
        #     loss.backward()
            
        #     if (i % 2) == 0:
        #         optimizer.zero_grad()
        #         optimizer.step()
        # else:
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        EMA_model.update(model)

        loss_meter.update(loss.item(), args.batch_size)
        distill_lseg_loss_meter.update(lseg_loss.mean().item(), args.batch_size)
        distill_dinov2_loss_meter.update(dinov2_loss.mean().item(), args.batch_size)
        distill_sd_loss_meter.update(sd_loss.mean().item(), args.batch_size)
        uncertainty_lseg_loss_meter.update(uncertainty_loss_lseg.mean().item(), args.batch_size)
        uncertainty_dinov2_loss_meter.update(uncertainty_loss_dinov2.mean().item(), args.batch_size)
        uncertainty_sd_loss_meter.update(uncertainty_loss_sd.mean().item(), args.batch_size)
        batch_time.update(time.time() - end)

        # adjust learning rate
        current_iter = epoch * len(train_loader) + i + 1
        current_lr = poly_learning_rate(
            args.base_lr, current_iter, max_iter, power=args.power)

        for index in range(0, args.index_split):
            optimizer.param_groups[index]['lr'] = current_lr
        for index in range(args.index_split, len(optimizer.param_groups)):
            optimizer.param_groups[index]['lr'] = current_lr * 10

        # calculate remain time
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(
            int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'CurrentIter {current_iter:.4f} '
                        "CurrentLR {current_lr:.4f} "
                        "DDP {ddp_training} "
                        'Loss {loss_meter.val:.4f} '
                        'LsegDistillLoss {distill_lseg_loss_meter.val:.4f} '
                        'LsegUncertaintyLoss {uncertainty_lseg_loss_meter.val:.4f} '
                        'Dinov2DistillLoss {distill_dinov2_loss_meter.val:.4f} '
                        'Dinov2UncertaintyLoss {uncertainty_dinov2_loss_meter.val:.4f} '
                        'SDDistillLoss {distill_sd_loss_meter.val:.4f} '
                        'SDUncertaintyLoss {uncertainty_sd_loss_meter.val:.4f} '.format(
                            epoch + 1, 
                            args.epochs, 
                            i + 1, 
                            len(train_loader),
                            batch_time=batch_time, data_time=data_time,
                            remain_time=remain_time,
                            current_iter=current_iter,
                            current_lr=current_lr,
                            ddp_training=args.distributed,
                            loss_meter=loss_meter,
                            distill_lseg_loss_meter=distill_lseg_loss_meter,
                            uncertainty_lseg_loss_meter=uncertainty_lseg_loss_meter,
                            distill_dinov2_loss_meter=distill_dinov2_loss_meter,
                            uncertainty_dinov2_loss_meter=uncertainty_dinov2_loss_meter,
                            distill_sd_loss_meter=distill_sd_loss_meter,
                            uncertainty_sd_loss_meter=uncertainty_sd_loss_meter,
                            ))
        if main_process():
            writer.add_scalar('loss_train_batch', loss_meter.val, current_iter)
            writer.add_scalar('ditill_lseg_loss_train_batch', distill_lseg_loss_meter.val, current_iter)
            writer.add_scalar('uncertainty_lseg_loss_train_batch', uncertainty_lseg_loss_meter.val, current_iter)
            writer.add_scalar('ditill_dinov2_loss_train_batch', distill_dinov2_loss_meter.val, current_iter)
            writer.add_scalar('uncertainty_dinov2_loss_train_batch', uncertainty_dinov2_loss_meter.val, current_iter)
            writer.add_scalar('ditill_sd_loss_train_batch', distill_sd_loss_meter.val, current_iter)
            writer.add_scalar('uncertainty_sd_loss_train_batch', uncertainty_sd_loss_meter.val, current_iter)
            writer.add_scalar('learning_rate', current_lr, current_iter)

        end = time.time()

    # mask_first = (coords[mask][:, 0] == 0)
    # output_3d = output_3d[mask_first]
    # feat_3d = feat_3d[mask_first]
    # logits_pred = output_3d.half() @ text_features.t()
    # logits_img = feat_3d.half() @ text_features.t()
    # logits_pred = torch.max(logits_pred, 1)[1].cpu().numpy()
    # logits_img = torch.max(logits_img, 1)[1].cpu().numpy()
    # mask = mask.cpu().numpy()
    # logits_gt = label_3d.numpy()[mask][mask_first.cpu().numpy()]
    # logits_gt[logits_gt == 255] = args.classes

    # pcl = coords[:, 1:].cpu().numpy()

    # seg_label_color = convert_labels_with_palette(
    #     logits_img, palette)
    # pred_label_color = convert_labels_with_palette(
    #     logits_pred, palette)
    # gt_label_color = convert_labels_with_palette(
    #     logits_gt, palette)
    # pcl_part = pcl[mask][mask_first.cpu().numpy()]

    # export_pointcloud(os.path.join(args.save_path, 'result', 'last', '{}_{}.ply'.format(
    #     args.feature_2d_extractor, epoch)), pcl_part, colors=seg_label_color)
    # export_pointcloud(os.path.join(args.save_path, 'result', 'last',
    #                     'pred_{}.ply'.format(epoch)), pcl_part, colors=pred_label_color)
    # export_pointcloud(os.path.join(args.save_path, 'result', 'last',
    #                     'gt_{}.ply'.format(epoch)), pcl_part, colors=gt_label_color)
    
    return loss_meter.avg


def validate(val_loader, model, criterion):
    '''Validation.'''

    torch.backends.cudnn.enabled = False
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    # obtain the CLIP feature
    text_features, _ = obtain_text_features_and_palette()

    with torch.no_grad():
        for batch_data in tqdm(val_loader):
            (coords, feat, label, inds_reverse) = batch_data
            sinput = SparseTensor(
                feat.cuda(non_blocking=True), coords.cuda(non_blocking=True))
            label = label.cuda(non_blocking=True)
            output, _, _, _, _, _ = model(sinput)
            output = output[inds_reverse, :]
            output = output.half() @ text_features.t()
            loss = criterion(output, label)
            output = torch.max(output, 1)[1]

            intersection, union, target = intersectionAndUnionGPU(output, label.detach(),
                                                                  args.classes, args.ignore_label)
            if args.multiprocessing_distributed:
                dist.all_reduce(intersection), dist.all_reduce(
                    union), dist.all_reduce(target)
            intersection, union, target = intersection.cpu(
            ).numpy(), union.cpu().numpy(), target.cpu().numpy()
            intersection_meter.update(intersection), union_meter.update(
                union), target_meter.update(target)

            loss_meter.update(loss.item(), args.batch_size)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if main_process():
        logger.info(
            'Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    return loss_meter.avg, mIoU, mAcc, allAcc


if __name__ == '__main__':
    main()



# import os
# import time
# import random
# import numpy as np
# import logging
# import argparse
# import math
# from copy import copy
# import torch
# import torch.backends.cudnn as cudnn
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.nn.parallel
# import torch.optim
# import torch.utils.data
# import torch.multiprocessing as mp
# import torch.distributed as dist
# from tensorboardX import SummaryWriter

# from MinkowskiEngine import SparseTensor
# from util import config
# from util.util import AverageMeter, intersectionAndUnionGPU, \
#     poly_learning_rate, save_checkpoint, save_ema_checkpoint, \
#     export_pointcloud, get_palette, convert_labels_with_palette, extract_clip_feature
# from dataset.label_constants import *
# from dataset.feature_loader_with_dinov2_sd import FusedFeatureLoader, collation_fn
# from dataset.point_loader_with_dinov2_sd import Point3DLoader, collation_fn_eval_all
# from models.disnet_with_uncertainty_dinov2_sd import DisNet as Model
# from tqdm import tqdm


# best_iou = 0.0
# best_ema_iou = 0.0

# def copy_attr(a, b, include=(), exclude=()):
#     # Copy attributes from b to a, options to only include [...] and to exclude [...]
#     for k, v in b.__dict__.items():
#         if (len(include) and k not in include) or k.startswith('_') or k in exclude:
#             continue
#         else:
#             setattr(a, k, v)

# def custom_deepcopy(model):
#     model_copy = type(model)()  # Create a new instance of the model's class
#     model_copy.load_state_dict(model.state_dict())  # Copy parameters and buffers
#     return model_copy

# class ModelEMA:
#     """ Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
#     Keep a moving average of everything in the model state_dict (parameters and buffers).
#     This is intended to allow functionality like
#     https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
#     A smoothed version of the weights is necessary for some training schemes to perform well.
#     This class is sensitive where it is initialized in the sequence of model init,
#     GPU assignment and distributed training wrappers.
#     """
 
#     def __init__(self, model, args, decay=0.9999, updates=0):
#         # Create EMA
#         # self.ema = copy.deepcopy(model).eval()
#         # self.ema = copy.deepcopy(model)
#         self.ema = get_model(args)
#         self.ema.load_state_dict(model.state_dict(), strict=True)
#         self.ema = self.ema.cuda()
#         # self.ema = self.ema.to(torch.device("cuda"))
#         self.ema.eval()
#         self.updates = updates  # number of EMA updates
#         self.decay = lambda x: decay * (1 - math.exp(-x / 2000))
#         for p in self.ema.parameters():
#             p.requires_grad_(False)
 
#     def update(self, model):
#         # Update EMA parameters
#         with torch.no_grad():
#             self.updates += 1
#             d = self.decay(self.updates)
#             msd = model.state_dict()  
#             for k, v in self.ema.state_dict().items():
#                 if v.dtype.is_floating_point:
#                     v *= d
#                     v += (1. - d) * msd[k].detach()
 
#     def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
#         # Update EMA attributes
#         copy_attr(self.ema, model, include, exclude)
        

# def worker_init_fn(worker_id):
#     '''Worker initialization.'''
#     random.seed(time.time() + worker_id)


# def get_parser():
#     '''Parse the config file.'''

#     parser = argparse.ArgumentParser(description='OpenScene 3D distillation.')
#     parser.add_argument('--config', type=str,
#                         default='config/scannet/distill_openseg.yaml',
#                         help='config file')
#     parser.add_argument('opts',
#                         default=None,
#                         help='see config/scannet/distill_openseg.yaml for all options',
#                         nargs=argparse.REMAINDER)
#     args_in = parser.parse_args()
#     assert args_in.config is not None
#     cfg = config.load_cfg_from_cfg_file(args_in.config)
#     if args_in.opts:
#         cfg = config.merge_cfg_from_list(cfg, args_in.opts)
#     os.makedirs(cfg.save_path, exist_ok=True)
#     model_dir = os.path.join(cfg.save_path, 'model')
#     result_dir = os.path.join(cfg.save_path, 'result')
#     os.makedirs(model_dir, exist_ok=True)
#     os.makedirs(result_dir, exist_ok=True)
#     os.makedirs(result_dir + '/last', exist_ok=True)
#     os.makedirs(result_dir + '/best', exist_ok=True)
#     return cfg


# def get_logger():
#     '''Define logger.'''

#     logger_name = "main-logger"
#     logger_in = logging.getLogger(logger_name)
#     logger_in.setLevel(logging.DEBUG)
#     handler = logging.StreamHandler()
#     fmt = "[%(asctime)s %(filename)s line %(lineno)d] %(message)s"
#     handler.setFormatter(logging.Formatter(fmt))
#     logger_in.addHandler(handler)
#     return logger_in


# def main_process():
#     return not args.multiprocessing_distributed or (
#         args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


# def main():
#     '''Main function.'''

#     args = get_parser()

#     os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(
#         str(x) for x in args.train_gpu)
#     cudnn.benchmark = True
#     if args.manual_seed is not None:
#         random.seed(args.manual_seed)
#         np.random.seed(args.manual_seed)
#         torch.manual_seed(args.manual_seed)
#         torch.cuda.manual_seed(args.manual_seed)
#         torch.cuda.manual_seed_all(args.manual_seed)
    
#     # By default we use shared memory for training
#     if not hasattr(args, 'use_shm'):
#         args.use_shm = True

#     print(
#         'torch.__version__:%s\ntorch.version.cuda:%s\ntorch.backends.cudnn.version:%s\ntorch.backends.cudnn.enabled:%s' % (
#             torch.__version__, torch.version.cuda, torch.backends.cudnn.version(), torch.backends.cudnn.enabled))

#     args.distributed = args.world_size > 1 or args.multiprocessing_distributed
#     args.ngpus_per_node = len(args.train_gpu)
#     if len(args.train_gpu) == 1:
#         args.sync_bn = False
#         args.distributed = False
#         args.multiprocessing_distributed = False
#         args.use_apex = False

#     if args.multiprocessing_distributed:
#         args.world_size = args.ngpus_per_node * args.world_size
#         mp.spawn(main_worker, nprocs=args.ngpus_per_node,
#                  args=(args.ngpus_per_node, args))
#     else:
#         main_worker(args.train_gpu, args.ngpus_per_node, args)


# def main_worker(gpu, ngpus_per_node, argss):
#     global args
#     global best_iou
#     global best_ema_iou
#     args = argss

#     if args.distributed:
#         if args.multiprocessing_distributed:
#             args.rank = args.rank * ngpus_per_node + gpu
#         dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, 
#                                 world_size=args.world_size, rank=args.rank)

#     model = get_model(args)
#     # TODO: add EMA Model
#     EMA_model = ModelEMA(model, args, decay=0.99, updates=0)

#     # if main_process():
#     #     global logger, writer
#     #     logger = get_logger()
#     #     writer = SummaryWriter(args.save_path)
#     #     logger.info(args)
#     #     logger.info("=> creating model ...")
#     #     logger.info(model)
    
#     global logger, writer
#     logger = get_logger()
#     writer = SummaryWriter(args.save_path)
#     logger.info(args)
#     logger.info("=> creating model ...")
#     logger.info(model)
    
    
    
#     if args.resume:
#         if os.path.isfile(args.resume):
#             # if main_process():
#             #     logger.info("=> loading checkpoint for ema model '{}'".format(args.resume))
#             logger.info("=> loading checkpoint for ema model '{}'".format(args.resume))
#             checkpoint = torch.load(
#                 args.resume, map_location=lambda storage, loc: storage.cuda())
#             EMA_model.ema.load_state_dict(checkpoint['ema_state_dict'], strict=True)
            
#             EMA_model.updates = checkpoint['updates']
            
#             best_ema_iou = checkpoint['best_ema_iou']
            
#             # if main_process():
#             #     logger.info("=> loaded checkpoint for ema model '{}' (epoch {})".format(
#             #         args.resume, checkpoint['epoch']))
#             logger.info("=> loaded checkpoint for ema model '{}' (epoch {})".format(
#                     args.resume, checkpoint['epoch']))
#         else:
#             # if main_process():
#             #     logger.info(
#             #         "=> no checkpoint found at '{}'".format(args.resume))
#             logger.info(
#                     "=> no checkpoint found at '{}'".format(args.resume))
#     ######################

#     # ####################### Optimizer ####################### #
#     optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr)
#     args.index_split = 0

#     if args.distributed:
#         torch.cuda.set_device(gpu)
#         args.batch_size = int(args.batch_size / ngpus_per_node)
#         args.batch_size_val = int(args.batch_size_val / ngpus_per_node)
#         args.workers = int(args.workers / ngpus_per_node)
#         # model = torch.nn.parallel.DistributedDataParallel(
#         #     model.cuda(), device_ids=[gpu])
#         # EMA_model.ema = torch.nn.parallel.DistributedDataParallel(
#         #     EMA_model.ema, device_ids=[gpu])
#         model = torch.nn.DataParallel(model, device_ids=[gpu])
#         EMA_model.ema = torch.nn.DataParallel(EMA_model.ema, device_ids=[gpu])
#     else:
#         model = model.cuda()

#     if args.resume:
#         if os.path.isfile(args.resume):
#             # if main_process():
#             #     logger.info("=> loading checkpoint '{}'".format(args.resume))
#             logger.info("=> loading checkpoint '{}'".format(args.resume))
#             checkpoint = torch.load(
#                 args.resume, map_location=lambda storage, loc: storage.cuda())
#             args.start_epoch = checkpoint['epoch']
#             model.load_state_dict(checkpoint['state_dict'], strict=True)
#             optimizer.load_state_dict(checkpoint['optimizer'])
#             best_iou = checkpoint['best_iou']
#             # if main_process():
#             #     logger.info("=> loaded checkpoint '{}' (epoch {})".format(
#             #         args.resume, checkpoint['epoch']))
#             logger.info("=> loaded checkpoint '{}' (epoch {})".format(
#                     args.resume, checkpoint['epoch']))
#         else:
#             # if main_process():
#             #     logger.info(
#             #         "=> no checkpoint found at '{}'".format(args.resume))
#             logger.info(
#                     "=> no checkpoint found at '{}'".format(args.resume))
    
    
#     # ####################### Data Loader ####################### #
#     if not hasattr(args, 'input_color'):
#         # by default we do not use the point color as input
#         args.input_color = False
#     train_data = FusedFeatureLoader(datapath_prefix=args.data_root,
#                                     datapath_prefix_feat=args.data_root_2d_fused_feature,
#                                     datapath_prefix_feat_dinov2=args.data_root_2d_fused_feature_dinov2,
#                                     datapath_prefix_feat_sd=args.data_root_2d_fused_feature_sd,
#                                     voxel_size=args.voxel_size,
#                                     split='train', aug=args.aug,
#                                     memcache_init=args.use_shm, loop=args.loop,
#                                     input_color=args.input_color
#                                     )
#     # train_sampler = torch.utils.data.distributed.DistributedSampler(
#     #     train_data) if args.distributed else None
#     train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
#                                             # shuffle=(train_sampler is None),
#                                             shuffle=None,
#                                             num_workers=args.workers, pin_memory=True,
#                                             # sampler=train_sampler,
#                                             drop_last=True, collate_fn=collation_fn,
#                                             worker_init_fn=worker_init_fn)
#     if args.evaluate:
#         val_data = Point3DLoader(datapath_prefix=args.data_root,
#                                  voxel_size=args.voxel_size,
#                                  split='val', aug=False,
#                                  memcache_init=args.use_shm,
#                                  eval_all=True,
#                                  input_color=args.input_color)
#         # val_sampler = torch.utils.data.distributed.DistributedSampler(
#         #     val_data) if args.distributed else None
#         val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val,
#                                                 shuffle=False,
#                                                 num_workers=args.workers, pin_memory=True,
#                                                 drop_last=False, collate_fn=collation_fn_eval_all,
#                                                 # sampler=val_sampler,
#                                                 )

#         criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda(gpu) # for evaluation

#     # ####################### Distill ####################### #
#     for epoch in range(args.start_epoch, args.epochs):
#         # if args.distributed:
#         #     train_sampler.set_epoch(epoch)
#         #     if args.evaluate:
#         #         val_sampler.set_epoch(epoch)
            
#         # loss_val, mIoU_val, mAcc_val, allAcc_val = validate(
#         #         val_loader, model, criterion)
        
#         loss_train = distill(train_loader, model, optimizer, epoch, EMA_model)
#         epoch_log = epoch + 1
#         # if main_process():
#         #     writer.add_scalar('loss_train', loss_train, epoch_log)
#         writer.add_scalar('loss_train', loss_train, epoch_log)

#         is_best = False
#         if args.evaluate and (epoch_log % args.eval_freq == 0):
#             loss_val, mIoU_val, mAcc_val, allAcc_val = validate(
#                 val_loader, model, criterion)
#             ema_loss_val, ema_mIoU_val, ema_mAcc_val, ema_allAcc_val = validate(
#                 val_loader, EMA_model.ema, criterion)
#             # raise NotImplementedError

#             # if main_process():
#             #     writer.add_scalar('loss_val', loss_val, epoch_log)
#             #     writer.add_scalar('mIoU_val', mIoU_val, epoch_log)
#             #     writer.add_scalar('mAcc_val', mAcc_val, epoch_log)
#             #     writer.add_scalar('allAcc_val', allAcc_val, epoch_log)
                
#             #     writer.add_scalar('ema_loss_val', ema_loss_val, epoch_log)
#             #     writer.add_scalar('ema_mIoU_val', ema_mIoU_val, epoch_log)
#             #     writer.add_scalar('ema_mAcc_val', ema_mAcc_val, epoch_log)
#             #     writer.add_scalar('ema_allAcc_val', ema_allAcc_val, epoch_log)
                
#             #     # remember best iou and save checkpoint
#             #     is_best = mIoU_val > best_iou
#             #     best_iou = max(best_iou, mIoU_val)
                
#             #     is_ema_best = ema_mIoU_val > best_ema_iou
#             #     best_ema_iou = max(best_ema_iou, ema_mIoU_val)
#             writer.add_scalar('loss_val', loss_val, epoch_log)
#             writer.add_scalar('mIoU_val', mIoU_val, epoch_log)
#             writer.add_scalar('mAcc_val', mAcc_val, epoch_log)
#             writer.add_scalar('allAcc_val', allAcc_val, epoch_log)
            
#             writer.add_scalar('ema_loss_val', ema_loss_val, epoch_log)
#             writer.add_scalar('ema_mIoU_val', ema_mIoU_val, epoch_log)
#             writer.add_scalar('ema_mAcc_val', ema_mAcc_val, epoch_log)
#             writer.add_scalar('ema_allAcc_val', ema_allAcc_val, epoch_log)
            
#             # remember best iou and save checkpoint
#             is_best = mIoU_val > best_iou
#             best_iou = max(best_iou, mIoU_val)
            
#             is_ema_best = ema_mIoU_val > best_ema_iou
#             best_ema_iou = max(best_ema_iou, ema_mIoU_val)

#         # if (epoch_log % args.save_freq == 0) and main_process():
#         if (epoch_log % args.save_freq == 0):
#             save_checkpoint(
#                 {
#                     'epoch': epoch_log,
#                     'state_dict': model.module.state_dict(),
#                     'optimizer': optimizer.module.state_dict(),
#                     'best_iou': best_iou,
#                     'best_ema_iou': best_ema_iou,
#                     'ema_state_dict': EMA_model.ema.module.state_dict(),
#                     'updates': EMA_model.updates,
#                 }, is_best, os.path.join(args.save_path, 'model')
#             )
            
#             save_ema_checkpoint(
#                 {
#                     'epoch': epoch_log,
#                     'state_dict': model.module.state_dict(),
#                     'optimizer': optimizer.module.state_dict(),
#                     'best_iou': best_iou,
#                     'best_ema_iou': best_ema_iou,
#                     'ema_state_dict': EMA_model.ema.module.state_dict(),
#                     'updates': EMA_model.updates,
#                 }, is_ema_best, os.path.join(args.save_path, 'model')
#             )
            
#     # if main_process():
#     #     writer.close()
#     #     logger.info('==>Training done!\nBest Iou: %.3f' % (best_iou))
#     writer.close()
#     logger.info('==>Training done!\nBest Iou: %.3f' % (best_iou))


# def get_model(cfg):
#     '''Get the 3D model.'''

#     model = Model(cfg=cfg)
#     return model

# def obtain_text_features_and_palette():
#     '''obtain the CLIP text feature and palette.'''

#     if 'scannet' in args.data_root:
#         labelset = list(SCANNET_LABELS_20)
#         labelset[-1] = 'other'
#         palette = get_palette()
#         dataset_name = 'scannet'
#     elif 'matterport' in args.data_root:
#         labelset = list(MATTERPORT_LABELS_21)
#         palette = get_palette(colormap='matterport')
#         dataset_name = 'matterport'
#     elif 'nuscenes' in args.data_root:
#         labelset = list(NUSCENES_LABELS_16)
#         palette = get_palette(colormap='nuscenes16')
#         dataset_name = 'nuscenes'

#     if not os.path.exists('saved_text_embeddings'):
#         os.makedirs('saved_text_embeddings')

#     if 'openseg' in args.feature_2d_extractor:
#         model_name="ViT-L/14@336px"
#         postfix = '_768' # the dimension of CLIP features is 768
#     elif 'lseg' in args.feature_2d_extractor:
#         model_name="ViT-B/32"
#         postfix = '_512' # the dimension of CLIP features is 512
#     else:
#         raise NotImplementedError

#     clip_file_name = 'saved_text_embeddings/clip_{}_labels{}.pt'.format(dataset_name, postfix)

#     try: # try to load the pre-saved embedding first
#         logger.info('Load pre-computed embeddings from {}'.format(clip_file_name))
#         text_features = torch.load(clip_file_name).cuda()
#     except: # extract CLIP text features and save them
#         text_features = extract_clip_feature(labelset, model_name=model_name)
#         torch.save(text_features, clip_file_name)

#     return text_features, palette


# def distill(train_loader, model, optimizer, epoch, EMA_model):
#     '''Distillation pipeline.'''

#     torch.backends.cudnn.enabled = True
#     batch_time = AverageMeter()
#     data_time = AverageMeter()

#     loss_meter = AverageMeter()
#     uncertainty_lseg_loss_meter = AverageMeter()
#     uncertainty_dinov2_loss_meter = AverageMeter()
#     uncertainty_sd_loss_meter = AverageMeter()
#     distill_lseg_loss_meter = AverageMeter()
#     distill_dinov2_loss_meter = AverageMeter()
#     distill_sd_loss_meter = AverageMeter()

#     model.train()
#     end = time.time()
#     max_iter = args.epochs * len(train_loader)

#     text_features, palette = obtain_text_features_and_palette()

#     # start the distillation process
#     for i, batch_data in enumerate(train_loader):
#         data_time.update(time.time() - end)

#         (coords, feat, label_3d, feat_3d, sd_feat_dinov2, sd_feat_3d, mask) = batch_data
        
#         ori_coords = coords.clone()
#         ori_sinput = SparseTensor(
#             feat.cuda(non_blocking=True), ori_coords.cuda(non_blocking=True))
        
#         coords[:, 1:4] += (torch.rand(3) * 100).type_as(coords)
#         sinput = SparseTensor(
#             feat.cuda(non_blocking=True), coords.cuda(non_blocking=True))
#         feat_3d, mask = feat_3d.cuda(
#             non_blocking=True), mask.cuda(non_blocking=True)

#         sd_feat_dinov2 = sd_feat_dinov2.cuda(non_blocking=True)
        
#         sd_feat_3d = sd_feat_3d.cuda(non_blocking=True)

#         output_3d, uncertainty_pred_lseg, output_3d_dinov2, uncertainty_pred_dinov2, output_3d_sd, uncertainty_pred_sd = model(sinput)
        
#         output_3d = output_3d[mask]
#         output_3d_dinov2 = output_3d_dinov2[mask]
#         output_3d_sd = output_3d_sd[mask]
#         # uncertainty_pred = uncertainty_pred[mask]
        
#         with torch.no_grad():
#             ema_output_3d, ema_uncertainty_pred_lseg, ema_output_3d_dinov2, ema_uncertainty_pred_dinov2, ema_output_3d_sd, ema_uncertainty_pred_sd = EMA_model.ema(ori_sinput)

#         if hasattr(args, 'loss_type') and args.loss_type == 'cosine':
#             # loss = (1 - torch.nn.CosineSimilarity()
#             #         (output_3d, feat_3d)).mean()
#             ##################################################
#             lseg_loss = (1 - torch.nn.CosineSimilarity()
#                     (output_3d, feat_3d))
#             ema_cosine_loss_lseg = (1.0 - torch.nn.CosineSimilarity()(
#                 ema_output_3d[mask].detach(),
#                 feat_3d
#             ))
#             uncertainty_loss_lseg = torch.nn.L1Loss()(
#                 uncertainty_pred_lseg[mask], 
#                 ema_cosine_loss_lseg[..., None].detach(),
#             )
#             ##################################################
#             dinov2_loss = (1 - torch.nn.CosineSimilarity()
#                     (F.normalize(output_3d_dinov2, 2, dim=1), F.normalize(sd_feat_dinov2, 2, dim=1)))
#             ema_cosine_loss_dinov2 = (1.0 - torch.nn.CosineSimilarity()(
#                 F.normalize(ema_output_3d_dinov2[mask].detach(), 2, dim=1),
#                 F.normalize(sd_feat_dinov2, 2, dim=1)
#             ))
#             uncertainty_loss_dinov2 = torch.nn.L1Loss()(
#                 uncertainty_pred_dinov2[mask], 
#                 ema_cosine_loss_dinov2[..., None].detach(),
#             )
#             ##################################################
#             sd_loss = (1 - torch.nn.CosineSimilarity()
#                     (F.normalize(output_3d_sd, 2, dim=1), F.normalize(sd_feat_3d, 2, dim=1)))
#             ema_cosine_loss_sd = (1.0 - torch.nn.CosineSimilarity()(
#                 F.normalize(ema_output_3d_sd[mask].detach(), 2, dim=1),
#                 F.normalize(sd_feat_3d, 2, dim=1)
#             ))
#             uncertainty_loss_sd = torch.nn.L1Loss()(
#                 uncertainty_pred_sd[mask], 
#                 ema_cosine_loss_sd[..., None].detach(),
#             )
#             ###################################################
            
#             cosine_loss = lseg_loss.mean() + dinov2_loss.mean() + sd_loss.mean()
#             uncertainty_loss = uncertainty_loss_lseg + uncertainty_loss_dinov2 + uncertainty_loss_sd
#             loss = cosine_loss + uncertainty_loss
            
#         elif hasattr(args, 'loss_type') and args.loss_type == 'l1':
#             # loss = torch.nn.L1Loss()(output_3d, feat_3d)
#             cosine_loss = torch.nn.L1Loss(reduction='none')(output_3d, feat_3d)
#             # import pdb;pdb.set_trace()
#             cosine_loss = cosine_loss.mean(dim=-1)
            
#             uncertainty_loss = torch.nn.L1Loss()(
#                 uncertainty_pred[mask], 
#                 cosine_loss[..., None].detach(),
#             )
            
#             cosine_loss = cosine_loss.mean()
#             loss = cosine_loss + uncertainty_loss
#         elif hasattr(args, 'loss_type') and args.loss_type == 'l2':
#             # loss = torch.nn.MSELoss()(output_3d, feat_3d)
#             cosine_loss = torch.nn.MSELoss(reduction='none')(output_3d.float(), feat_3d.float())
            
#             cosine_loss = cosine_loss.mean(dim=-1)
            
#             uncertainty_loss = torch.nn.L1Loss()(
#                 uncertainty_pred[mask], 
#                 cosine_loss[..., None].detach(),
#             )
            
#             cosine_loss = cosine_loss.mean()
#             loss = cosine_loss + uncertainty_loss
#         else:
#             raise NotImplementedError
        
#         # if not args.distributed:
            
#         #     loss.backward()
            
#         #     if (i % 2) == 0:
#         #         optimizer.zero_grad()
#         #         optimizer.step()
#         # else:
#         #     optimizer.zero_grad()
#         #     loss.backward()
#         #     optimizer.step()
        
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         EMA_model.update(model)

#         loss_meter.update(loss.item(), args.batch_size)
#         distill_lseg_loss_meter.update(lseg_loss.mean().item(), args.batch_size)
#         distill_dinov2_loss_meter.update(dinov2_loss.mean().item(), args.batch_size)
#         distill_sd_loss_meter.update(sd_loss.mean().item(), args.batch_size)
#         uncertainty_lseg_loss_meter.update(uncertainty_loss_lseg.mean().item(), args.batch_size)
#         uncertainty_dinov2_loss_meter.update(uncertainty_loss_dinov2.mean().item(), args.batch_size)
#         uncertainty_sd_loss_meter.update(uncertainty_loss_sd.mean().item(), args.batch_size)
#         batch_time.update(time.time() - end)

#         # adjust learning rate
#         current_iter = epoch * len(train_loader) + i + 1
#         current_lr = poly_learning_rate(
#             args.base_lr, current_iter, max_iter, power=args.power)

#         for index in range(0, args.index_split):
#             optimizer.param_groups[index]['lr'] = current_lr
#         for index in range(args.index_split, len(optimizer.param_groups)):
#             optimizer.param_groups[index]['lr'] = current_lr * 10

#         # calculate remain time
#         remain_iter = max_iter - current_iter
#         remain_time = remain_iter * batch_time.avg
#         t_m, t_s = divmod(remain_time, 60)
#         t_h, t_m = divmod(t_m, 60)
#         remain_time = '{:02d}:{:02d}:{:02d}'.format(
#             int(t_h), int(t_m), int(t_s))

#         # if (i + 1) % args.print_freq == 0 and main_process():
#         if (i + 1) % args.print_freq == 0:
#             logger.info('Epoch: [{}/{}][{}/{}] '
#                         'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
#                         'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
#                         'Remain {remain_time} '
#                         'CurrentIter {current_iter:.4f} '
#                         "CurrentLR {current_lr:.4f} "
#                         "DDP {ddp_training} "
#                         'Loss {loss_meter.val:.4f} '
#                         'LsegDistillLoss {distill_lseg_loss_meter.val:.4f} '
#                         'LsegUncertaintyLoss {uncertainty_lseg_loss_meter.val:.4f} '
#                         'Dinov2DistillLoss {distill_dinov2_loss_meter.val:.4f} '
#                         'Dinov2UncertaintyLoss {uncertainty_dinov2_loss_meter.val:.4f} '
#                         'SDDistillLoss {distill_sd_loss_meter.val:.4f} '
#                         'SDUncertaintyLoss {uncertainty_sd_loss_meter.val:.4f} '.format(
#                             epoch + 1, 
#                             args.epochs, 
#                             i + 1, 
#                             len(train_loader),
#                             batch_time=batch_time, data_time=data_time,
#                             remain_time=remain_time,
#                             current_iter=current_iter,
#                             current_lr=current_lr,
#                             ddp_training=args.distributed,
#                             loss_meter=loss_meter,
#                             distill_lseg_loss_meter=distill_lseg_loss_meter,
#                             uncertainty_lseg_loss_meter=uncertainty_lseg_loss_meter,
#                             distill_dinov2_loss_meter=distill_dinov2_loss_meter,
#                             uncertainty_dinov2_loss_meter=uncertainty_dinov2_loss_meter,
#                             distill_sd_loss_meter=distill_sd_loss_meter,
#                             uncertainty_sd_loss_meter=uncertainty_sd_loss_meter,
#                             ))
#         # if main_process():
#         #     writer.add_scalar('loss_train_batch', loss_meter.val, current_iter)
#         #     writer.add_scalar('ditill_lseg_loss_train_batch', distill_lseg_loss_meter.val, current_iter)
#         #     writer.add_scalar('uncertainty_lseg_loss_train_batch', uncertainty_lseg_loss_meter.val, current_iter)
#         #     writer.add_scalar('ditill_dinov2_loss_train_batch', distill_dinov2_loss_meter.val, current_iter)
#         #     writer.add_scalar('uncertainty_dinov2_loss_train_batch', uncertainty_dinov2_loss_meter.val, current_iter)
#         #     writer.add_scalar('ditill_sd_loss_train_batch', distill_sd_loss_meter.val, current_iter)
#         #     writer.add_scalar('uncertainty_sd_loss_train_batch', uncertainty_sd_loss_meter.val, current_iter)
#         #     writer.add_scalar('learning_rate', current_lr, current_iter)
#         writer.add_scalar('loss_train_batch', loss_meter.val, current_iter)
#         writer.add_scalar('ditill_lseg_loss_train_batch', distill_lseg_loss_meter.val, current_iter)
#         writer.add_scalar('uncertainty_lseg_loss_train_batch', uncertainty_lseg_loss_meter.val, current_iter)
#         writer.add_scalar('ditill_dinov2_loss_train_batch', distill_dinov2_loss_meter.val, current_iter)
#         writer.add_scalar('uncertainty_dinov2_loss_train_batch', uncertainty_dinov2_loss_meter.val, current_iter)
#         writer.add_scalar('ditill_sd_loss_train_batch', distill_sd_loss_meter.val, current_iter)
#         writer.add_scalar('uncertainty_sd_loss_train_batch', uncertainty_sd_loss_meter.val, current_iter)
#         writer.add_scalar('learning_rate', current_lr, current_iter)

#         end = time.time()

#     # mask_first = (coords[mask][:, 0] == 0)
#     # output_3d = output_3d[mask_first]
#     # feat_3d = feat_3d[mask_first]
#     # logits_pred = output_3d.half() @ text_features.t()
#     # logits_img = feat_3d.half() @ text_features.t()
#     # logits_pred = torch.max(logits_pred, 1)[1].cpu().numpy()
#     # logits_img = torch.max(logits_img, 1)[1].cpu().numpy()
#     # mask = mask.cpu().numpy()
#     # logits_gt = label_3d.numpy()[mask][mask_first.cpu().numpy()]
#     # logits_gt[logits_gt == 255] = args.classes

#     # pcl = coords[:, 1:].cpu().numpy()

#     # seg_label_color = convert_labels_with_palette(
#     #     logits_img, palette)
#     # pred_label_color = convert_labels_with_palette(
#     #     logits_pred, palette)
#     # gt_label_color = convert_labels_with_palette(
#     #     logits_gt, palette)
#     # pcl_part = pcl[mask][mask_first.cpu().numpy()]

#     # export_pointcloud(os.path.join(args.save_path, 'result', 'last', '{}_{}.ply'.format(
#     #     args.feature_2d_extractor, epoch)), pcl_part, colors=seg_label_color)
#     # export_pointcloud(os.path.join(args.save_path, 'result', 'last',
#     #                     'pred_{}.ply'.format(epoch)), pcl_part, colors=pred_label_color)
#     # export_pointcloud(os.path.join(args.save_path, 'result', 'last',
#     #                     'gt_{}.ply'.format(epoch)), pcl_part, colors=gt_label_color)
    
#     return loss_meter.avg


# def validate(val_loader, model, criterion):
#     '''Validation.'''

#     torch.backends.cudnn.enabled = False
#     loss_meter = AverageMeter()
#     intersection_meter = AverageMeter()
#     union_meter = AverageMeter()
#     target_meter = AverageMeter()

#     # obtain the CLIP feature
#     text_features, _ = obtain_text_features_and_palette()

#     with torch.no_grad():
#         for batch_data in tqdm(val_loader):
#             (coords, feat, label, inds_reverse) = batch_data
#             sinput = SparseTensor(
#                 feat.cuda(non_blocking=True), coords.cuda(non_blocking=True))
#             label = label.cuda(non_blocking=True)
#             output, _, _, _, _, _ = model(sinput)
#             output = output[inds_reverse, :]
#             output = output.half() @ text_features.t()
#             loss = criterion(output, label)
#             output = torch.max(output, 1)[1]

#             intersection, union, target = intersectionAndUnionGPU(output, label.detach(),
#                                                                   args.classes, args.ignore_label)
#             if args.multiprocessing_distributed:
#                 dist.all_reduce(intersection), dist.all_reduce(
#                     union), dist.all_reduce(target)
#             intersection, union, target = intersection.cpu(
#             ).numpy(), union.cpu().numpy(), target.cpu().numpy()
#             intersection_meter.update(intersection), union_meter.update(
#                 union), target_meter.update(target)

#             loss_meter.update(loss.item(), args.batch_size)

#     iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
#     accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
#     mIoU = np.mean(iou_class)
#     mAcc = np.mean(accuracy_class)
#     allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
#     # if main_process():
#     #     logger.info(
#     #         'Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
#     logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
#     return loss_meter.avg, mIoU, mAcc, allAcc


# if __name__ == '__main__':
#     main()
