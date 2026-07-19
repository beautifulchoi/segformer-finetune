from __future__ import annotations

import os.path as osp

import mmcv
import numpy as np

from ..builder import PIPELINES


@PIPELINES.register_module()
class LoadBinaryAnnotations:
    def __init__(self,
                 reduce_zero_label=False,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        if reduce_zero_label:
            raise ValueError('binary masks require reduce_zero_label=False')
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        if results.get('seg_prefix') is not None:
            filename = osp.join(results['seg_prefix'], results['ann_info']['seg_map'])
        else:
            filename = results['ann_info']['seg_map']
        mask = mmcv.imfrombytes(
            self.file_client.get(filename),
            flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)
        values = set(np.unique(mask).tolist())
        if values.issubset({0, 255}):
            mask = (mask == 255).astype(np.uint8)
        elif not values.issubset({0, 1}):
            raise ValueError(f'binary mask contains invalid values: {sorted(values)}')
        results['gt_semantic_seg'] = mask
        results['seg_fields'].append('gt_semantic_seg')
        return results
