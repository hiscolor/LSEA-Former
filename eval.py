import argparse
import os
import glob
import time
import json
from pprint import pprint


import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data


from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import valid_one_epoch, ANETdetection, fix_random_seed


def oracle_evaluation(val_dataset, output_file, gt_file, subset, tiou_thresholds, dataset_name):
    """
    Oracle 评测: 用 GT 自身作为预测结果 (score=1)，验证评测流程是否正确。
    如果评测流程正确，mAP@0.5 应该接近 100%。
    """
    from libs.utils.Evaluation import run_evaluation
    import numpy as np

    print("\n" + "="*60)
    print("[Oracle Evaluation] Using GT as predictions (score=1)")
    print("="*60)


    with open(gt_file, 'r') as f:
        gt_data = json.load(f)

    if 'database' in gt_data:
        gt_db = gt_data['database']
    else:
        gt_db = gt_data


    results = {
        'video-id': [],
        't-start': [],
        't-end': [],
        'label': [],
        'score': []
    }

    num_gt_segments = 0
    num_videos_with_gt = 0

    for vid_key, vid_info in gt_db.items():
        vid_split = vid_info.get('split', vid_info.get('subset', '')).lower()
        if vid_split != subset:
            continue

        annotations = vid_info.get('annotations', vid_info.get('fake_periods', []))
        if not annotations:
            continue

        num_videos_with_gt += 1

        for ann in annotations:
            if isinstance(ann, dict):
                t_start = float(ann['segment'][0])
                t_end = float(ann['segment'][1])
            else:
                t_start = float(ann[0])
                t_end = float(ann[1])

            results['video-id'].append(vid_key)
            results['t-start'].append(t_start)
            results['t-end'].append(t_end)
            results['label'].append(0)
            results['score'].append(1.0)
            num_gt_segments += 1

    print(f"[Oracle] Found {num_gt_segments} GT segments in {num_videos_with_gt} videos")


    results['t-start'] = np.array(results['t-start'])
    results['t-end'] = np.array(results['t-end'])
    results['label'] = np.array(results['label'])
    results['score'] = np.array(results['score'])


    oracle_output = output_file.replace('.json', '_oracle.json')
    mAP, mAR = run_evaluation(
        results, gt_file, oracle_output,
        max_avg_nr_proposal=100,
        tiou_thre=tiou_thresholds,
        subset=subset,
        cls_score_file=None
    )

    print("\n" + "="*60)
    print("[Oracle Result] If mAP@0.5 is NOT close to 100%, there's a bug in evaluation!")
    print("="*60 + "\n")

    return mAP



def main(args):
    """0. load config"""

    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")
    assert len(cfg['test_split']) > 0, "Test set must be specified!"
    if ".pth.tar" in args.ckpt:
        assert os.path.isfile(args.ckpt), "CKPT file does not exist!"
        ckpt_file = args.ckpt
    else:
        assert os.path.isdir(args.ckpt), "CKPT file folder does not exist!"
        if args.epoch > 0:
            ckpt_file = os.path.join(
                args.ckpt, 'epoch_{:03d}.pth.tar'.format(args.epoch)
            )
        else:
            ckpt_file_list = sorted(glob.glob(os.path.join(args.ckpt, '*.pth.tar')))
            ckpt_file = ckpt_file_list[-1]
        assert os.path.exists(ckpt_file)

    if args.topk > 0:
        cfg['model']['test_cfg']['max_seg_num'] = args.topk
    pprint(cfg)

    """1. fix all randomness"""

    _ = fix_random_seed(0, include_cuda=True)

    """2. create dataset / dataloader"""
    val_dataset = make_dataset(
        cfg['dataset_name'], False, cfg['test_split'], **cfg['dataset']
    )

    val_loader = make_data_loader(
        val_dataset, False, None, 1, cfg['loader']['num_workers']
    )

    """3. create model and evaluator"""

    model = make_meta_arch(cfg['model_name'], **cfg['model'])

    model = nn.DataParallel(model, device_ids=cfg['devices'])

    """4. load ckpt"""
    print("=> loading checkpoint '{}'".format(ckpt_file))

    checkpoint = torch.load(
        ckpt_file,
        map_location = lambda storage, loc: storage.cuda(cfg['devices'][0])
    )

    print("Loading from EMA model ...")

    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['state_dict_ema'], strict=False)
    if missing_keys:
        print(f"Missing keys (新模块参数，将随机初始化): {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys (旧 checkpoint 中多余的参数): {unexpected_keys}")
    del checkpoint


    det_eval, output_file = None, None
    val_db_vars = val_dataset.get_attributes()
    if cfg['dataset_name'].lower() in ['lavdf','lavdfv2','tvil','psynd','tvilnotnone','psyndnotnone','tad_tvil']:
        output_file = os.path.join(os.path.split(ckpt_file)[0], 'test_results.json')
    elif not args.saveonly:

        det_eval = ANETdetection(
            val_dataset.json_file,
            val_dataset.split[0],
            tiou_thresholds = val_db_vars['tiou_thresholds']
        )
    else:
        output_file = os.path.join(os.path.split(ckpt_file)[0], 'test_results.pkl')

    """5. Test the model"""

    if args.oracle_eval:
        oracle_mAP = oracle_evaluation(
            val_dataset,
            output_file,
            val_dataset.json_file,
            val_dataset.split[0],
            val_db_vars['tiou_thresholds'],
            cfg['dataset_name']
        )
        print("Oracle evaluation done!")
        return

    print("\nStart testing model {:s} ...".format(cfg['model_name']))
    start = time.time()
    mAP = valid_one_epoch(
        val_loader,
        model,
        -1,
        evaluator=det_eval,
        output_file=output_file,
        ext_score_file=cfg['test_cfg']['ext_score_file'],
        tb_writer=None,
        print_freq=args.print_freq,
        gt_file=val_dataset.json_file,
        subset=val_dataset.split[0],
        tiou_thre=val_db_vars['tiou_thresholds'],
        max_avg_nr_proposal=cfg['model']['test_cfg']['max_seg_num'],
        dataset_name=cfg['dataset_name']
    )
    end = time.time()
    print("All done! Total time: {:0.2f} sec".format(end - start))
    return


if __name__ == '__main__':
    """Entry Point"""

    parser = argparse.ArgumentParser(
      description='Train a point-based transformer for action localization')
    parser.add_argument('config', type=str, metavar='DIR',
                        help='path to a config file')
    parser.add_argument('ckpt', type=str, metavar='DIR',
                        help='path to a checkpoint')
    parser.add_argument('-epoch', type=int, default=-1,
                        help='checkpoint epoch')
    parser.add_argument('-t', '--topk', default=-1, type=int,
                        help='max number of output actions (default: -1)')
    parser.add_argument('--saveonly', action='store_true',
                        help='Only save the ouputs without evaluation (e.g., for test set)')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        help='print frequency (default: 10 iterations)')
    parser.add_argument('--oracle_eval', action='store_true',
                        help='Oracle evaluation: use GT as predictions to verify evaluation pipeline')
    args = parser.parse_args()
    main(args)
