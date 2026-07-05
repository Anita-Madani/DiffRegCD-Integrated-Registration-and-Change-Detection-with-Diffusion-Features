import logging
logger = logging.getLogger('base')


def create_model(opt):
    """Create the (frozen) denoising-diffusion feature extractor."""
    from .model import DDPM as M
    m = M(opt)
    logger.info('Model [{}] is created.'.format(m.__class__.__name__))
    return m


def create_CD_model_256(opt):
    """Create the joint registration + change-detection head (DiffRegCD)."""
    from .cd_model_256 import CD as M
    m = M(opt)
    logger.info('CD Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
