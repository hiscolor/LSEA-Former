import argparse
import os
import time
import datetime
from pprint import pprint


import torch
import torch.nn as nn
import torch.utils.data

from torch.utils.tensorboard import SummaryWriter


from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import (train_one_epoch, valid_one_epoch, ANETdetection,
                        save_checkpoint, make_optimizer, make_scheduler,
                        fix_random_seed, ModelEma, Logger,
                        DistillationWrapper, load_teacher_model)
from libs.modeling import SKDDistillationWrapper, SKDDistillationWrapperV2, load_fcad_teacher

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"



def load_finetune_state_dict(model, state_dict):
    """Load a checkpoint into a raw module or a DataParallel wrapper."""
    stripped = {}
    for key, value in state_dict.items():
        name = key[7:] if key.startswith('module.') else key
        if name.startswith('student.'):
            name = name[len('student.'):]
        stripped[name] = value
    target = model.module if isinstance(model, nn.DataParallel) else model
    return target.load_state_dict(stripped, strict=False)


def main(args):
    """main function that handles training / inference"""

    """1. setup parameters / folders"""

    args.start_epoch = 0
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")
    pprint(cfg)


    if not os.path.exists(cfg['output_folder']):
        os.makedirs(cfg['output_folder'], exist_ok=True)
    cfg_filename = os.path.basename(args.config).replace('.yaml', '')
    if len(args.output) == 0:

        ts = time.strftime("%Y_%m_%d_%H_%M_%S")
        ckpt_folder = os.path.join(
            cfg['output_folder'], cfg_filename + '_' + str(ts))
    else:
        ckpt_folder = os.path.join(
            cfg['output_folder'], cfg_filename + '_' + str(args.output))
    if not os.path.exists(ckpt_folder):
        os.mkdir(ckpt_folder)

    tb_writer = SummaryWriter(os.path.join(ckpt_folder, 'logs'))


    logger = Logger(os.path.join(ckpt_folder, 'log.json'))
    logger.start()


    print("Config:")
    pprint(cfg)


    rng_generator = fix_random_seed(cfg['init_rand_seed'], include_cuda=True)


    cfg['opt']["learning_rate"] *= len(cfg['devices'])
    cfg['loader']['num_workers'] *= len(cfg['devices'])

    """2. create dataset / dataloader"""
    train_dataset = make_dataset(
        cfg['dataset_name'], True, cfg['train_split'], **cfg['dataset']
    )


    train_db_vars = train_dataset.get_attributes()
    cfg['model']['train_cfg']['head_empty_cls'] = train_db_vars['empty_label_ids']


    train_loader = make_data_loader(
        train_dataset, True, rng_generator, **cfg['loader'])



    det_eval, output_file = None, None
    bestmAP = 0
    if args.eval:
        val_dataset = make_dataset(
            cfg['dataset_name'], False, cfg['val_split'], **cfg['dataset']
        )
        val_loader = make_data_loader(
            val_dataset, False, None,1, cfg['loader']['num_workers'])
        val_db_vars = val_dataset.get_attributes()
        if cfg['dataset_name'].lower() in ['lavdf','lavdfv2','vil','psynd','vilnotnone','lavdfvm','psyndnotnone','tad_tvil','tadiff_tvil','tvil_videocls']:
            output_file = os.path.join(ckpt_folder, 'val_results.json')
        else:

            det_eval = ANETdetection(
                val_dataset.json_file,
                val_dataset.split[0],
                tiou_thresholds = val_db_vars['tiou_thresholds']
            )
    """3. create model, optimizer, and scheduler"""

    model = make_meta_arch(cfg['model_name'], **cfg['model'])


    distill_cfg = cfg['train_cfg'].get('distillation', {})
    use_distillation = distill_cfg.get('enabled', False)

    if use_distillation:
        teacher_cfg_path = distill_cfg.get('teacher_config', '')
        teacher_ckpt = distill_cfg.get('teacher_ckpt', '')
        if not teacher_cfg_path or not os.path.isfile(teacher_cfg_path):
            raise ValueError(f"Distillation enabled but teacher_config not found: {teacher_cfg_path}")
        if not teacher_ckpt or not os.path.isfile(teacher_ckpt):
            raise ValueError(f"Distillation enabled but teacher_ckpt not found: {teacher_ckpt}")

        print(f"Loading teacher model for distillation from: {teacher_ckpt}")


        use_skd = distill_cfg.get('use_skd', False) or cfg['model_name'] == 'FCADFormer'

        if use_skd:
            teacher = load_fcad_teacher(teacher_cfg_path, teacher_ckpt, cfg['devices'][0])
            skd_config = {
                'enabled': True,
                'temperature': distill_cfg.get('temperature', 4.0),
                'alpha': distill_cfg.get('alpha', 0.5),
                'beta': distill_cfg.get('beta', 1.0),
                'theta_conf': distill_cfg.get('theta_conf', 0.3),
                'kd_clip_weight': distill_cfg.get('kd_clip_weight', 1.0),
                'kd_vid_weight': distill_cfg.get('kd_vid_weight', 0.5),
                'use_teacher_calibrated_logits': distill_cfg.get('use_teacher_calibrated_logits', True),
                'use_motion_clip_weight': distill_cfg.get('use_motion_clip_weight', True),
            }
            model = SKDDistillationWrapperV2(
                student=model,
                teacher=teacher,
                skd_config=skd_config,
            )
            print(f"SKD V2 Distillation: temperature={skd_config['temperature']}, "
                  f"theta_conf={skd_config['theta_conf']}, gate={'ON' if skd_config['theta_conf'] > 0 else 'OFF'}")
        else:

            teacher = load_teacher_model(teacher_cfg_path, teacher_ckpt, cfg['devices'][0])
            model = DistillationWrapper(
                student=model,
                teacher=teacher,
                kd_video_weight=distill_cfg.get('kd_video_weight', 0.5),
                kd_temp_weight=distill_cfg.get('kd_temp_weight', 1.0),
                temperature=distill_cfg.get('temperature', 4.0),
            )
            print(f"Distillation enabled: kd_video_weight={distill_cfg.get('kd_video_weight', 0.5)}, "
                  f"kd_temp_weight={distill_cfg.get('kd_temp_weight', 1.0)}, "
                  f"temperature={distill_cfg.get('temperature', 4.0)}")


    model = nn.DataParallel(model, device_ids=cfg['devices'])

    optimizer = make_optimizer(model, cfg['opt'])

    num_iters_per_epoch = len(train_loader)
    scheduler = make_scheduler(optimizer, cfg['opt'], num_iters_per_epoch)


    print("Using model EMA ...")
    model_ema = ModelEma(model)

    """4. Resume from model / Misc"""

    if getattr(args, 'init_from', ''):
        if os.path.isfile(args.init_from):
            checkpoint = torch.load(args.init_from, map_location='cpu')
            state = checkpoint.get('state_dict', checkpoint)
            state_ema = checkpoint.get('state_dict_ema', state)
            missing_keys, unexpected_keys = load_finetune_state_dict(model, state)
            if missing_keys:
                print(f"Missing keys (init-from): {len(missing_keys)}")
            if unexpected_keys:
                print(f"Unexpected keys (init-from): {len(unexpected_keys)}")
            ema_missing, ema_unexpected = load_finetune_state_dict(
                model_ema.module, state_ema
            )
            if ema_missing:
                print(f"EMA missing keys (init-from): {len(ema_missing)}")
            if ema_unexpected:
                print(f"EMA unexpected keys (init-from): {len(ema_unexpected)}")
            args.start_epoch = 0
            print("=> init weights from '{:s}' (epoch reset to 0, optimizer fresh)".format(
                args.init_from
            ))
            del checkpoint
        else:
            print("=> no init checkpoint found at '{}'".format(args.init_from))
            return


    if args.resume:
        if os.path.isfile(args.resume):

            checkpoint = torch.load(args.resume,
                map_location = lambda storage, loc: storage.cuda(
                    cfg['devices'][0]))
            args.start_epoch = checkpoint['epoch']

            missing_keys, unexpected_keys = model.load_state_dict(checkpoint['state_dict'], strict=False)
            if missing_keys:
                print(f"Missing keys (新模块参数，将随机初始化): {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys (旧 checkpoint 中多余的参数): {unexpected_keys}")

            missing_keys_ema, unexpected_keys_ema = model_ema.module.load_state_dict(checkpoint['state_dict_ema'], strict=False)
            if missing_keys_ema:
                print(f"EMA Missing keys: {missing_keys_ema}")


            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{:s}' (epoch {:d}".format(
                args.resume, checkpoint['epoch']
            ))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return


    with open(os.path.join(ckpt_folder, 'config.txt'), 'w') as fid:
        pprint(cfg, stream=fid)
        fid.flush()

    """4. training / validation loop"""
    print("\nStart training model {:s} ...".format(cfg['model_name']))


    max_epochs = cfg['opt'].get(
        'early_stop_epochs',
        cfg['opt']['epochs'] + cfg['opt']['warmup_epochs']
    )
    for epoch in range(args.start_epoch, max_epochs):

        train_one_epoch(
            train_loader,
            model,
            optimizer,
            scheduler,
            epoch,
            model_ema = model_ema,
            clip_grad_l2norm = cfg['train_cfg']['clip_grad_l2norm'],
            tb_writer=tb_writer,
            print_freq=args.print_freq
        )


        should_save = (
            ((epoch + 1) == max_epochs) or
            ((args.ckpt_freq > 0) and ((epoch + 1) % args.ckpt_freq == 0))
        )

        should_eval = should_save and (epoch + 1) >= args.eval_start_epoch

        if should_save:
            mAP=0.0
            if should_eval and ((output_file is not None) or (det_eval is not None)):
                print(f"\n[Eval]: Running evaluation at epoch {epoch + 1}...")
                mAP = valid_one_epoch(
                    val_loader,
                    model,
                    epoch,
                    evaluator=det_eval,
                    output_file=output_file,
                    ext_score_file=None,
                    tb_writer=tb_writer,
                    print_freq=args.print_freq,
                    gt_file=val_dataset.json_file,
                    subset=val_dataset.split[0],
                    tiou_thre=val_db_vars['tiou_thresholds'],
                    max_avg_nr_proposal=cfg['model']['test_cfg']['max_seg_num'],
                    dataset_name=cfg['dataset_name']
                )
            elif not should_eval:
                print(f"\n[Eval]: Skipping evaluation at epoch {epoch + 1} (starts at epoch {args.eval_start_epoch})")

            save_states = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'scheduler': scheduler.state_dict(),
                'optimizer': optimizer.state_dict(),
            }

            save_states['state_dict_ema'] = model_ema.module.state_dict()


            is_best = mAP > bestmAP
            is_last = (epoch + 1) == max_epochs

            if is_best:
                save_checkpoint(
                    save_states,
                    True,
                    file_folder=ckpt_folder,
                    file_name='best_model.pth.tar'
                )
                print(f"[Save] Best model saved at epoch {epoch + 1} with mAP={mAP:.4f}")
                bestmAP = mAP

            if is_last:
                save_checkpoint(
                    save_states,
                    False,
                    file_folder=ckpt_folder,
                    file_name='last_model.pth.tar'
                )
                print(f"[Save] Last model saved at epoch {epoch + 1}")





    tb_writer.close()
    print("All done!")
    logger.close()
    return


if __name__ == '__main__':
    """Entry Point"""

    parser = argparse.ArgumentParser(
      description='Train a point-based transformer for action localization')
    parser.add_argument('config', metavar='DIR',
                        help='path to a config file')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        help='print frequency (default: 10 iterations)')
    parser.add_argument('-c', '--ckpt-freq', default=2, type=int,
                        help='checkpoint frequency (default: every 2 epochs)')
    parser.add_argument('--eval-start-epoch', default=1, type=int,
                        help='start evaluation from this epoch (default: 1)')
    parser.add_argument('--output', default='', type=str,
                        help='name of exp folder (default: none)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to a checkpoint (default: none)')
    parser.add_argument('--init-from', default='', type=str, metavar='PATH',
                        help='load model weights only (fresh optimizer/epoch), for fine-tuning')
    parser.add_argument('--eval',action='store_true',
                        help='evaluation')
    args = parser.parse_args()
    main(args)
