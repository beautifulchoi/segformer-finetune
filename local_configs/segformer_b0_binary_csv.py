dataset_type = 'CsvBinaryDataset'
csv_path = 'data/coco_binary/manifest.csv'
data_root = None
image_col = 'image'
mask_col = 'mask'
split_col = 'split'
work_dir = 'work_dirs/segformer_b0_binary_smoke'
checkpoint_path = 'pretrained/segformer_b0_ade512_mmseg.pth'

norm_cfg = dict(type='SyncBN', requires_grad=True)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)
crop_size = (512, 512)

model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(type='mit_b0', style='pytorch'),
    decode_head=dict(
        type='SegFormerHead',
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(embed_dim=256),
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadBinaryAnnotations'),
    dict(type='Resize', img_scale=crop_size, keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=0),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=crop_size,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]),
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        csv_path=csv_path,
        data_root=data_root,
        image_col=image_col,
        mask_col=mask_col,
        split_col=split_col,
        split='train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        csv_path=csv_path,
        data_root=data_root,
        image_col=image_col,
        mask_col=mask_col,
        split_col=split_col,
        split='val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        csv_path=csv_path,
        data_root=data_root,
        image_col=image_col,
        mask_col=mask_col,
        split_col=split_col,
        split='test',
        pipeline=test_pipeline))

optimizer = dict(
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01)
optimizer_config = dict()
lr_config = dict(policy='poly', power=1.0, min_lr=0.0, by_epoch=False)
runner = dict(type='EpochBasedRunner', max_epochs=1)
workflow = [('train', 1)]
evaluation = dict(interval=1, metric=['mIoU', 'mDice'])
checkpoint_config = dict(interval=1)
log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = checkpoint_path
resume_from = None
cudnn_benchmark = False
seed = 20260719
gpu_ids = range(1)
