import mmcv
import torch
import torch.nn.parallel._functions as torch_parallel_functions
from mmcv.utils.parrots_wrapper import SyncBatchNorm
import mmcv.parallel._functions as mmcv_parallel_functions

from .version import __version__, version_info

MMCV_MIN = '1.1.4'
MMCV_MAX = '1.3.0'


def digit_version(version_str):
    digit_version = []
    for x in version_str.split('.'):
        if x.isdigit():
            digit_version.append(int(x))
        elif x.find('rc') != -1:
            patch_version = x.split('rc')
            digit_version.append(int(patch_version[0]) - 1)
            digit_version.append(int(patch_version[1]))
    return digit_version


mmcv_min_version = digit_version(MMCV_MIN)
mmcv_max_version = digit_version(MMCV_MAX)
mmcv_version = digit_version(mmcv.__version__)


assert (mmcv_min_version <= mmcv_version <= mmcv_max_version), \
    f'MMCV=={mmcv.__version__} is used but incompatible. ' \
    f'Please install mmcv>={mmcv_min_version}, <={mmcv_max_version}.'


def _specify_ddp_gpu_num(self, gpu_size):
    return None


SyncBatchNorm._specify_ddp_gpu_num = _specify_ddp_gpu_num


def _get_stream(device):
    if isinstance(device, int):
        device = torch.device('cuda', device)
    if device.type == 'cpu' or not torch.cuda.is_available():
        return None
    return torch.cuda.Stream(device=device.index)


torch_parallel_functions._get_stream = _get_stream
mmcv_parallel_functions._get_stream = _get_stream

__all__ = ['__version__', 'version_info']
