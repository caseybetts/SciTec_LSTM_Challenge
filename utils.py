# utils.py
import config
import random
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)

def set_seed(seed):
    logger.info(f"Setting random seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(config.CONFIG["random_seed"])