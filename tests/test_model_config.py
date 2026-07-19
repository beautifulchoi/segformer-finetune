import mmcv

from mmseg.models import build_segmentor


def test_binary_config_builds_two_class_decode_head():
    config = mmcv.Config.fromfile('local_configs/segformer_b0_binary_csv.py')
    model = build_segmentor(
        config.model,
        train_cfg=config.get('train_cfg'),
        test_cfg=config.get('test_cfg'))

    assert model.decode_head.num_classes == 2
    assert model.decode_head.linear_pred.out_channels == 2
