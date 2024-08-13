'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2024-08-12 10:13:29
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2024-08-13 09:34:02
FilePath: \PointNeXt\testcsv.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import argparse
import pandas as pd
from datetime import datetime
from utils.config import *
from utils import dist_utils
from utils import logger
from parser import *
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter

from utils.random import set_random_seed

def main(gpu,cfg):
    # Initialize the detectron2 logger and set its verbosity level to "INFO".
    """
    Args:
        output (str): a file name or a directory to save log. If None, will not save log file.
            If ends with ".txt" or ".log", assumed to be a file name.
            Otherwise, logs will be saved to `output/log.txt`.
        name (str): the root module name of this logger
    Returns:
        logging.Logger: a logger
    """
    logger.setup_logger_dist(cfg.log_path, cfg.rank, name=cfg.dataset.common.NAME)
    
    writer = SummaryWriter(log_dir=cfg.run_dir) if cfg.is_training else None
    # 设置cuda随机数种子
    set_random_seed(cfg.seed + cfg.rank, deterministic=cfg.deterministic)
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Scene segmentation training/testing')
    parser.add_argument('--cfg', type=str, help='config file',default="./cfg/s3dis/pointnet++.yaml")
    parser.add_argument('--profile', action='store_true', default=False, help='set to True to profile speed')
    args, opts = parser.parse_known_args()
    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    logging.info("opts is: {}\n"%opts)
    logging.info("{}\n"%cfg.get("model"))
    
    if cfg.seed is None:
        cfg.seed = np.random.randint(1, 10000)
    
# init distributed env first, since logger depends on the dist info.
    cfg.rank, cfg.world_size, cfg.distributed, cfg.mp = dist_utils.get_dist_info(cfg)
    cfg.sync_bn = cfg.world_size > 1
    
    # init log dir
    cfg.task_name = args.cfg.split('.')[-2].split('/')[-2]  # task/dataset name, \eg s3dis, modelnet40_cls
    cfg.cfg_basename = args.cfg.split('.')[-2].split('/')[-1]  # cfg_basename, \eg pointnext-xl
    tags = [
        cfg.task_name,  # task name (the folder of name under ./cfgs
        cfg.mode,
        cfg.cfg_basename,  # cfg file name
        f'ngpus{cfg.world_size}',
    ]
    opt_list = [] # for checking experiment configs from logging file
    for i, opt in enumerate(opts):
        if 'rank' not in opt and 'dir' not in opt and 'root' not in opt and 'pretrain' not in opt and 'path' not in opt and 'wandb' not in opt and '/' not in opt:
            opt_list.append(opt)
    cfg.root_dir = os.path.join(cfg.root_dir, cfg.task_name)
    cfg.opts = '-'.join(opt_list)

    cfg.is_training = cfg.mode not in ['test', 'testing', 'val', 'eval', 'evaluation']
    if cfg.mode in ['resume', 'val', 'test']:
        logger.resume_exp_directory(cfg, pretrained_path=cfg.pretrained_path)
    
    else:
        logger.generate_exp_directory(cfg, tags, additional_id=os.environ.get('MASTER_PORT', None))
    
    os.environ["JOB_LOG_DIR"] = cfg.log_dir
    cfg_path = os.path.join(cfg.run_dir, "cfg.yaml")
    with open(cfg_path, 'w') as f:
        yaml.dump(cfg, f, indent=2)#将cfg 转储为 f
        os.system('cp %s %s' % (args.cfg, cfg.run_dir))#复制文件
    cfg.cfg_path = cfg_path
    main(0,cfg)
