from __future__ import annotations

import argparse
import json
from pathlib import Path

import mmcv
import torch
from transformers import SegformerForSemanticSegmentation

from mmseg.models import build_segmentor


def local_key_for_hf(source_key):
    key = source_key.removeprefix('segformer.')
    if key.startswith('encoder.patch_embeddings.'):
        stage, suffix = key.split('.', 3)[2:]
        prefix = f'backbone.patch_embed{int(stage) + 1}'
        return f'{prefix}.{suffix.replace("layer_norm", "norm")}'
    if key.startswith('encoder.block.'):
        _, _, stage, block, suffix = key.split('.', 4)
        prefix = f'backbone.block{int(stage) + 1}.{block}'
        replacements = {
            'layer_norm_1': 'norm1',
            'layer_norm_2': 'norm2',
            'attention.self.query': 'attn.q',
            'attention.self.key': 'attn.kv',
            'attention.self.value': 'attn.kv',
            'attention.self.sr': 'attn.sr',
            'attention.self.layer_norm': 'attn.norm',
            'attention.output.dense': 'attn.proj',
            'mlp.dense1': 'mlp.fc1',
            'mlp.dwconv.dwconv': 'mlp.dwconv.dwconv',
            'mlp.dense2': 'mlp.fc2',
        }
        for source_prefix, local_prefix in replacements.items():
            if suffix.startswith(source_prefix):
                return f'{prefix}.{local_prefix}{suffix[len(source_prefix):]}'
    if key.startswith('encoder.layer_norm.'):
        stage, suffix = key.split('.', 3)[2:]
        return f'backbone.norm{int(stage) + 1}.{suffix}'
    if key.startswith('decode_head.linear_c.'):
        index, suffix = key.split('.', 3)[2:]
        local_name = {'0': 'linear_c1', '1': 'linear_c2', '2': 'linear_c3', '3': 'linear_c4'}[index]
        return f'decode_head.{local_name}.{suffix}'
    if key.startswith('decode_head.linear_fuse.'):
        return f'decode_head.linear_fuse.conv.{key.split(".", 2)[2]}'
    if key.startswith('decode_head.batch_norm.'):
        return f'decode_head.linear_fuse.bn.{key.split(".", 2)[2]}'
    return None


def convert_state_dict(hf_state, model_state):
    mapped = {}
    used = set()
    skipped = []
    unexpected = []
    shape_mismatched = []
    qkv_groups = {}
    for source_key, tensor in hf_state.items():
        if source_key.startswith('decode_head.classifier.'):
            skipped.append(source_key)
            continue
        if '.attention.self.key.' in source_key or '.attention.self.value.' in source_key:
            target = local_key_for_hf(source_key)
            role = 'key' if '.attention.self.key.' in source_key else 'value'
            group = qkv_groups.setdefault(target, {})
            group[role] = tensor
            group[f'{role}_source'] = source_key
            continue
        target = local_key_for_hf(source_key)
        if target is None:
            unexpected.append(source_key)
            continue
        if target not in model_state:
            unexpected.append(source_key)
            continue
        if tuple(model_state[target].shape) != tuple(tensor.shape):
            shape_mismatched.append({'source': source_key, 'target': target})
            continue
        mapped[target] = tensor.detach().cpu().clone()
        used.add(source_key)
    for target, group in qkv_groups.items():
        if 'key' not in group or 'value' not in group:
            unexpected.extend(group.get(f'{role}_source') for role in ('key', 'value') if f'{role}_source' in group)
            continue
        combined = torch.cat([group['key'], group['value']], dim=0)
        if target not in model_state or tuple(model_state[target].shape) != tuple(combined.shape):
            shape_mismatched.append({'source': [group['key_source'], group['value_source']], 'target': target})
            continue
        mapped[target] = combined.detach().cpu().clone()
        used.update((group['key_source'], group['value_source']))
    missing = [key for key in model_state if key not in mapped]
    reinitialized = [
        key for key in missing
        if key.startswith('decode_head.linear_pred.') or key.startswith('decode_head.conv_seg.')
    ]
    report = {
        'backbone_coverage': 0.0,
        'loaded': sorted(mapped),
        'skipped': sorted(skipped),
        'missing': sorted(missing),
        'reinitialized': sorted(reinitialized),
        'unexpected': sorted(unexpected),
        'shape_mismatched': shape_mismatched,
        'source_keys_used': len(used),
    }
    return mapped, report


def convert_checkpoint(config_path, output_path, report_path, model_id):
    config = mmcv.Config.fromfile(config_path)
    model = build_segmentor(config.model, train_cfg=config.get('train_cfg'), test_cfg=config.get('test_cfg'))
    hf_model = SegformerForSemanticSegmentation.from_pretrained(model_id)
    model_state = model.state_dict()
    mapped, report = convert_state_dict(hf_model.state_dict(), model_state)
    backbone_keys = [key for key in model_state if key.startswith('backbone.')]
    loaded_backbone = [key for key in mapped if key.startswith('backbone.')]
    report['backbone_coverage'] = len(loaded_backbone) / len(backbone_keys)
    if report['backbone_coverage'] < 1.0 or report['shape_mismatched']:
        raise RuntimeError(json.dumps(report, indent=2))
    state_dict = {key: value.detach().cpu().clone() for key, value in model_state.items()}
    state_dict.update(mapped)
    checkpoint = {
        'meta': {
            'CLASSES': ['background', 'foreground'],
            'PALETTE': [[0, 0, 0], [255, 0, 0]],
            'conversion': report,
        },
        'state_dict': state_dict,
    }
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    saved = torch.load(output, map_location='cpu')
    if any(not torch.equal(saved['state_dict'][key], value) for key, value in mapped.items()):
        raise RuntimeError('serialized checkpoint tensor equality verification failed')
    Path(report_path).resolve().write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('backbone_coverage', 'source_keys_used', 'skipped', 'missing', 'unexpected', 'shape_mismatched')}, indent=2))
    return report


def _parse_args():
    parser = argparse.ArgumentParser(description='Convert the HF SegFormer-B0 checkpoint to local MMSeg format.')
    parser.add_argument('--config', default='local_configs/segformer_b0_binary_csv.py')
    parser.add_argument('--output', default='pretrained/segformer_b0_ade512_mmseg.pth')
    parser.add_argument('--report', default='pretrained/segformer_b0_ade512_mmseg.json')
    parser.add_argument('--model-id', default='nvidia/segformer-b0-finetuned-ade-512-512')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    convert_checkpoint(args.config, args.output, args.report, args.model_id)
