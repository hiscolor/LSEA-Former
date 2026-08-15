import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.environ.get('FCAD_CODE_ROOT', '.'))

from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import fix_random_seed
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, balanced_accuracy_score


def get_data_parallel_device_ids(device):
    """Return the single CUDA index requested by the caller."""
    if not str(device).startswith('cuda'):
        return None
    _, _, index = str(device).partition(':')
    return [int(index)] if index else [0]


def get_video_probability(result, score_key=''):
    """Read video probability from adapted heads or official UMMAFormer output."""
    if score_key:
        if score_key not in result:
            raise KeyError(f"Requested score key is unavailable: {score_key}")
        value = result[score_key]
    else:
        value = result.get('vid_prob', result.get('presence_score', 0.5))
    if isinstance(value, torch.Tensor):
        value = value.item()
    return float(value)


def select_checkpoint_state(checkpoint, weights='raw'):
    if not isinstance(checkpoint, dict):
        if weights != 'raw':
            raise KeyError("state_dict_ema is unavailable in a raw state dict")
        return checkpoint
    if weights == 'ema':
        if 'state_dict_ema' not in checkpoint:
            raise KeyError("state_dict_ema is unavailable in this checkpoint")
        return checkpoint['state_dict_ema']
    return checkpoint.get('state_dict', checkpoint)


def build_prediction_records(video_ids, labels, probabilities):
    if not (len(video_ids) == len(labels) == len(probabilities)):
        raise ValueError("video_ids, labels, and probabilities must have equal length")
    return [
        {
            'video_id': str(video_id),
            'label': int(label),
            'probability': float(probability),
        }
        for video_id, label, probability in zip(video_ids, labels, probabilities)
    ]


def load_model(config_path, checkpoint_path, device='cuda:0', weights='raw'):
    cfg = load_config(config_path)
    fix_random_seed(cfg.get('init_rand_seed', 1234567891), include_cuda=True)
    model = make_meta_arch(cfg['model_name'], **cfg['model'])

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = select_checkpoint_state(checkpoint, weights)

    new_state_dict = {}
    for k, v in state_dict.items():
        name = k

        if name.startswith('module.'):
            name = name[len('module.'):]

        if name.startswith('student.'):
            name = name[len('student.'):]
        new_state_dict[name] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        print(f"  Warning: {len(missing)} missing keys")
    if unexpected:
        print(f"  Warning: {len(unexpected)} unexpected keys (teacher weights from wrapper, safe to ignore)")

    model = model.to(device)
    device_ids = get_data_parallel_device_ids(device)
    if device_ids is not None:
        model = nn.DataParallel(model, device_ids=device_ids)
    model.eval()
    return model, cfg


def load_source_info():
    source_path = os.path.join(
        os.environ.get('FCAD_DATA_ROOT', '/home/lby/featurize/data/TAD_TVIL'),
        'dataset_source_info.json',
    )
    if not os.path.exists(source_path):
        return None
    with open(source_path) as f:
        data = json.load(f)
    vid_to_source = {}
    for source, splits in data.items():
        for split, vids in splits.items():
            for vid in vids:
                vid_to_source[vid] = source
    return vid_to_source


def compute_metrics(probs, labels, report_threshold=None):
    if len(probs) == 0:
        return None

    auc = roc_auc_score(labels, probs) * 100
    ap = average_precision_score(labels, probs) * 100

    best_f1 = 0
    best_f1_thresh = 0.5
    best_acc = 0
    best_acc_thresh = 0.5
    best_bal = 0
    best_bal_thresh = 0.5
    for t in np.arange(0.05, 0.95, 0.01):
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        acc = accuracy_score(labels, preds)
        bal = balanced_accuracy_score(labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_f1_thresh = t
        if acc > best_acc:
            best_acc = acc
            best_acc_thresh = t
        if bal > best_bal:
            best_bal = bal
            best_bal_thresh = t


    preds_adaptive = (probs >= best_acc_thresh).astype(int)
    preds_fixed = (probs >= 0.5).astype(int)

    acc_adaptive = accuracy_score(labels, preds_adaptive) * 100
    acc_fixed = accuracy_score(labels, preds_fixed) * 100
    bal_acc = balanced_accuracy_score(labels, preds_fixed) * 100
    bal_acc_adaptive = best_bal * 100
    f1_adaptive = f1_score(labels, preds_adaptive, zero_division=0) * 100

    tp = int(((preds_adaptive == 1) & (labels == 1)).sum())
    fp = int(((preds_adaptive == 1) & (labels == 0)).sum())
    tn = int(((preds_adaptive == 0) & (labels == 0)).sum())
    fn = int(((preds_adaptive == 0) & (labels == 1)).sum())

    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0

    metrics = {
        'total_videos': len(probs),
        'auc': round(auc, 2),
        'ap': round(ap, 2),
        'acc_fixed': round(acc_fixed, 2),
        'balanced_acc': round(bal_acc, 2),
        'balanced_acc_adaptive': round(bal_acc_adaptive, 2),
        'acc_adaptive': round(acc_adaptive, 2),
        'best_threshold': round(best_acc_thresh, 2),
        'best_f1_threshold': round(best_f1_thresh, 2),
        'best_bal_threshold': round(best_bal_thresh, 2),
        'f1': round(f1_adaptive, 2),
        'precision': round(precision, 2),
        'recall': round(recall, 2),
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
    }
    if report_threshold is not None:
        report_preds = (probs >= report_threshold).astype(int)
        metrics.update({
            'report_threshold': round(float(report_threshold), 4),
            'acc_at_report_threshold': round(
                accuracy_score(labels, report_preds) * 100, 2
            ),
            'balanced_acc_at_report_threshold': round(
                balanced_accuracy_score(labels, report_preds) * 100, 2
            ),
            'f1_at_report_threshold': round(
                f1_score(labels, report_preds, zero_division=0) * 100, 2
            ),
        })
    return metrics


def evaluate_split(
    model, cfg, split_name, device='cuda:0', source_filter=None, score_key='',
    report_threshold=None,
):
    if split_name == 'validation':
        split_list = cfg.get('val_split', ['validation'])
    elif split_name == 'test':
        split_list = cfg.get('test_split', ['test'])
    else:
        split_list = [split_name]

    dataset = make_dataset(cfg['dataset_name'], False, split_list, **cfg['dataset'])
    loader = make_data_loader(dataset, False, None, 1, 4)

    vid_to_source = load_source_info() if source_filter else None

    all_probs = []
    all_labels = []
    all_vid_ids = []

    model.eval()
    with torch.no_grad():
        for idx, video_list in enumerate(loader):
            output = model(video_list)

            if isinstance(output, dict):
                continue

            for vid_idx, result in enumerate(output):
                vid_prob = get_video_probability(result, score_key)

                if 'video_label' in video_list[vid_idx]:
                    gt_label = video_list[vid_idx]['video_label']
                    if isinstance(gt_label, torch.Tensor):
                        gt_label = gt_label.item()
                else:
                    has_segments = video_list[vid_idx].get('segments') is not None
                    if has_segments and len(video_list[vid_idx]['segments']) > 0:
                        gt_label = 1.0
                    else:
                        gt_label = 0.0

                vid_id = video_list[vid_idx].get('video_id', f'video_{idx}_{vid_idx}')

                all_probs.append(vid_prob)
                all_labels.append(int(gt_label))
                all_vid_ids.append(vid_id)

    probs = np.array(all_probs)
    labels = np.array(all_labels)

    if source_filter and vid_to_source:
        mask = np.array([vid_to_source.get(vid, 'dota') == source_filter for vid in all_vid_ids])
        probs = probs[mask]
        labels = labels[mask]
        all_vid_ids = [vid for vid, keep in zip(all_vid_ids, mask) if keep]
        print(f"  Source filter '{source_filter}': {mask.sum()}/{len(mask)} videos selected")

    if len(probs) == 0:
        return None

    results = compute_metrics(probs, labels, report_threshold=report_threshold)
    results['split'] = split_name
    results['predictions'] = build_prediction_records(all_vid_ids, labels, probs)
    if source_filter:
        results['source_filter'] = source_filter
    return results


def print_results(results, label=''):
    if not results:
        print(f"  No results for {label}")
        return
    tag = f" [{label}]" if label else ""
    print(f"\n{'='*50}")
    print(f"Results{tag} ({results['total_videos']} videos):")
    print(f"  AUC:       {results['auc']:.2f}%")
    print(f"  AP:        {results['ap']:.2f}%")
    print(f"  Acc (0.5): {results['acc_fixed']:.2f}%")
    print(f"  Acc (best):{results['acc_adaptive']:.2f}% (threshold={results['best_threshold']})")
    if 'acc_at_report_threshold' in results:
        print(
            f"  Acc (validation threshold): {results['acc_at_report_threshold']:.2f}% "
            f"(threshold={results['report_threshold']})"
        )
    print(f"  F1:        {results['f1']:.2f}%")
    print(f"  Precision: {results['precision']:.2f}%")
    print(f"  Recall:    {results['recall']:.2f}%")
    print(f"  TP={results['tp']} FP={results['fp']} TN={results['tn']} FN={results['fn']}")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--split', default='validation', choices=['validation', 'test'])
    parser.add_argument('--output', default='')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--weights', choices=['raw', 'ema'], default='raw',
                        help='Checkpoint weights to evaluate (default: raw)')
    parser.add_argument('--score-key', default='',
                        help='Explicit model output key used as video score')
    parser.add_argument('--report-threshold', type=float, default=None,
                        help='Also report metrics at a preselected validation threshold')
    parser.add_argument('--source-filter', default='', choices=['', 'dota', 'uitdrone'],
                        help='Filter test videos by source dataset')
    parser.add_argument('--per-domain', action='store_true',
                        help='Evaluate separately on DoTA, UIT-ADrone, and All')
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint}")
    model, cfg = load_model(
        args.config, args.checkpoint, device=args.device, weights=args.weights
    )

    if args.per_domain:
        all_results = {}
        for source in ['dota', 'uitdrone', None]:
            label = source if source else 'all'
            print(f"\nEvaluating on {args.split} set [{label}]...")
            results = evaluate_split(model, cfg, args.split,
                                     device=args.device, source_filter=source,
                                     score_key=args.score_key,
                                     report_threshold=args.report_threshold)
            all_results[label] = results
            print_results(results, label)

        if args.output:
            output_dir = os.path.dirname(args.output)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            with open(args.output, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"\nSaved to {args.output}")
    else:
        src = args.source_filter if args.source_filter else None
        print(f"Evaluating on {args.split} set...")
        results = evaluate_split(
            model, cfg, args.split, device=args.device, source_filter=src,
            score_key=args.score_key, report_threshold=args.report_threshold
        )
        if results:
            results['evaluation'] = {
                'config': os.path.realpath(args.config),
                'checkpoint': os.path.realpath(args.checkpoint),
                'weights': args.weights,
                'device': args.device,
            }
        print_results(results, args.split)

        if results and args.output:
            output_dir = os.path.dirname(args.output)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()
