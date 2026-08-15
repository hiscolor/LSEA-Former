import os
import sys
import shutil
import time
import pickle
import datetime

import numpy as np
import random
from copy import deepcopy

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

from .lr_schedulers import LinearWarmupMultiStepLR, LinearWarmupCosineAnnealingLR
from .postprocessing import postprocess_results
from ..modeling import MaskedConv1D, Scale, AffineDropPath, LayerNorm
from .Evaluation import run_evaluation


class Logger:
    """
    Logger class that redirects stdout to both terminal and log file.
    Similar to OpenTAD's logging format.
    """
    def __init__(self, log_file, mode='w'):
        self.terminal = sys.stdout
        self.log_file = log_file
        self.mode = mode
        self.file = None

    def start(self):
        """Start logging to file"""
        log_dir = os.path.dirname(self.log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        self.file = open(self.log_file, self.mode, encoding='utf-8')
        sys.stdout = self
        return self

    def write(self, message):
        """Write message to both terminal and log file"""
        self.terminal.write(message)
        if self.file is not None:

            if message.strip():
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                self.file.write(f'{timestamp} Train INFO: {message}')
                if not message.endswith('\n'):
                    self.file.write('\n')
            self.file.flush()

    def flush(self):
        """Flush both outputs"""
        self.terminal.flush()
        if self.file is not None:
            self.file.flush()

    def close(self):
        """Close the log file and restore stdout"""
        if self.file is not None:
            self.file.close()
            self.file = None
        sys.stdout = self.terminal


def fix_random_seed(seed, include_cuda=True):
    rng_generator = torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if include_cuda:

        cudnn.enabled = True
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        cudnn.enabled = True
        cudnn.benchmark = True
    return rng_generator


def save_checkpoint(state, is_best, file_folder,
                    file_name='checkpoint.pth.tar'):
    """save checkpoint to file"""
    if not os.path.exists(file_folder):
        os.mkdir(file_folder)
    torch.save(state, os.path.join(file_folder, file_name))
    if is_best:



        torch.save(state, os.path.join(file_folder, 'model_best.pth.tar'))





def print_model_params(model):
    for name, param in model.named_parameters():
        print(name, param.min().item(), param.max().item(), param.mean().item())
    return


def make_optimizer(model, optimizer_config):
    """create optimizer
    return a supported optimizer
    """


    decay = set()
    no_decay = set()
    whitelist_weight_modules = (
        torch.nn.Linear,
        torch.nn.Conv1d,
        torch.nn.Conv2d,
        MaskedConv1D,
        torch.nn.ConvTranspose1d,
    )
    blacklist_weight_modules = (LayerNorm, torch.nn.GroupNorm, torch.nn.Embedding, torch.nn.LayerNorm,torch.nn.Parameter)


    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn
            if not p.requires_grad:
                no_decay.add(fpn)
            elif pn.endswith('bias'):

                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):

                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):

                no_decay.add(fpn)
            elif pn.endswith('scale') and isinstance(m, (Scale, AffineDropPath)):

                no_decay.add(fpn)
            elif pn.endswith('rel_pe'):

                no_decay.add(fpn)
            elif pn.endswith('cls_token'):

                no_decay.add(fpn)
            elif ('gru' in pn) and ('bias' in pn):

                no_decay.add(fpn)
            elif  ('gru' in pn) and ('weight' in pn):

                decay.add(fpn)
            elif pn.endswith('time_weighting'):
                no_decay.add(fpn)
            elif pn.endswith('in_proj_weight'):
                decay.add(fpn)
            elif pn.endswith('temperature') or pn.endswith('attn1') or pn.endswith('attn2') or pn.endswith('attn3') or pn.endswith('attn3') or pn.endswith('attn4'):
                no_decay.add(fpn)

            elif pn.endswith('lambda_param') or pn.endswith('bias_param') or pn.endswith('gamma') or pn.endswith('_delta'):
                no_decay.add(fpn)




    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
    assert len(param_dict.keys() - union_params) == 0,        "parameters %s were not separated into either decay/no_decay set!"        % (str(param_dict.keys() - union_params), )


    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": optimizer_config["weight_decay"]},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]

    if optimizer_config["type"] == "SGD":
        optimizer = optim.SGD(
            optim_groups,
            lr=optimizer_config["learning_rate"],
            momentum=optimizer_config["momentum"]
        )
    elif optimizer_config["type"] == "AdamW":
        optimizer = optim.AdamW(
            optim_groups,
            lr=optimizer_config["learning_rate"]
        )
    else:
        raise TypeError("Unsupported optimizer!")

    return optimizer


def make_scheduler(
    optimizer,
    optimizer_config,
    num_iters_per_epoch,
    last_epoch=-1
):
    """create scheduler
    return a supported scheduler
    All scheduler returned by this function should step every iteration
    """
    if optimizer_config["warmup"]:
        max_epochs = optimizer_config["epochs"] + optimizer_config["warmup_epochs"]
        max_steps = max_epochs * num_iters_per_epoch


        warmup_epochs = optimizer_config["warmup_epochs"]
        warmup_steps = warmup_epochs * num_iters_per_epoch


        if optimizer_config["schedule_type"] == "cosine":

            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_steps,
                max_steps,
                last_epoch=last_epoch
            )

        elif optimizer_config["schedule_type"] == "multistep":

            steps = [num_iters_per_epoch * step for step in optimizer_config["schedule_steps"]]
            scheduler = LinearWarmupMultiStepLR(
                optimizer,
                warmup_steps,
                steps,
                gamma=optimizer_config["schedule_gamma"],
                last_epoch=last_epoch
            )
        else:
            raise TypeError("Unsupported scheduler!")

    else:
        max_epochs = optimizer_config["epochs"]
        max_steps = max_epochs * num_iters_per_epoch


        if optimizer_config["schedule_type"] == "cosine":

            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                max_steps,
                last_epoch=last_epoch
            )

        elif optimizer_config["schedule_type"] == "multistep":

            steps = [num_iters_per_epoch * step for step in optimizer_config["schedule_steps"]]
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                steps,
                gamma=schedule_config["gamma"],
                last_epoch=last_epoch
            )
        else:
            raise TypeError("Unsupported scheduler!")

    return scheduler


class AverageMeter(object):
    """Computes and stores the average and current value.
    Used to compute dataset stats from mini-batches
    """
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = 0.0

    def initialize(self, val, n):
        self.val = val
        self.avg = val
        self.sum = val * n
        self.count = n
        self.initialized = True

    def update(self, val, n=1):
        if not self.initialized:
            self.initialize(val, n)
        else:
            self.add(val, n)

    def add(self, val, n):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.999, device=None):
        super().__init__()

        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)



def train_one_epoch(
    train_loader,
    model,
    optimizer,
    scheduler,
    curr_epoch,
    model_ema = None,
    clip_grad_l2norm = -1,
    tb_writer = None,
    print_freq = 20
):
    """Training the model for one epoch"""

    batch_time = AverageMeter()
    losses_tracker = {}
    grad_tracker = {'grad_norm': AverageMeter(), 'grad_max': AverageMeter()}

    num_iters = len(train_loader)

    model.train()


    print("\n[Train]: Epoch {:d} started".format(curr_epoch))
    start = time.time()
    for iter_idx, video_list in enumerate(train_loader, 0):

        optimizer.zero_grad(set_to_none=True)

        losses = model(video_list)
        losses['final_loss'].backward()


        total_grad_norm = 0.0
        max_grad = 0.0
        num_params_with_grad = 0
        for p in model.parameters():
            if p.grad is not None:
                param_grad_norm = p.grad.data.norm(2).item()
                total_grad_norm += param_grad_norm ** 2
                max_grad = max(max_grad, p.grad.data.abs().max().item())
                num_params_with_grad += 1
        total_grad_norm = total_grad_norm ** 0.5
        grad_tracker['grad_norm'].update(total_grad_norm)
        grad_tracker['grad_max'].update(max_grad)


        if clip_grad_l2norm > 0.0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), clip_grad_l2norm
            )

        optimizer.step()
        scheduler.step()

        if model_ema is not None:
            model_ema.update(model)



        if (iter_idx % print_freq) == 0:

            torch.cuda.synchronize()
            batch_time.update((time.time() - start) / print_freq)
            start = time.time()


            for key, value in losses.items():

                if key not in losses_tracker:
                    losses_tracker[key] = AverageMeter()

                if isinstance(value, torch.Tensor):
                    losses_tracker[key].update(value.item())
                else:
                    losses_tracker[key].update(value)


            lr = scheduler.get_last_lr()[0]
            global_step = curr_epoch * num_iters + iter_idx
            if tb_writer is not None:

                tb_writer.add_scalar(
                    'train/learning_rate',
                    lr,
                    global_step
                )

                tag_dict = {}
                for key, value in losses_tracker.items():
                    if key not in ["final_loss", "num_pos", "num_neg"]:
                        tag_dict[key] = value.val
                tb_writer.add_scalars(
                    'train/all_losses',
                    tag_dict,
                    global_step
                )

                tb_writer.add_scalar(
                    'train/final_loss',
                    losses_tracker['final_loss'].val,
                    global_step
                )

                if 'num_pos' in losses_tracker:
                    tb_writer.add_scalar('train/num_pos', losses_tracker['num_pos'].val, global_step)
                if 'num_neg' in losses_tracker:
                    tb_writer.add_scalar('train/num_neg', losses_tracker['num_neg'].val, global_step)


            block1 = 'Epoch: [{:03d}][{:05d}/{:05d}]'.format(
                curr_epoch, iter_idx, num_iters
            )
            block2 = 'Time {:.2f} ({:.2f})'.format(
                batch_time.val, batch_time.avg
            )
            block3 = 'Loss {:.2f} ({:.2f})\n'.format(
                losses_tracker['final_loss'].val,
                losses_tracker['final_loss'].avg
            )
            block4 = ''
            for key, value in losses_tracker.items():
                if key not in ["final_loss", "num_pos", "num_neg"]:
                    block4  += '\t{:s} {:.2f} ({:.2f})'.format(
                        key, value.val, value.avg
                    )


            block5 = ''
            if 'num_pos' in losses_tracker:
                num_pos = losses_tracker['num_pos'].val
                if 'num_neg' in losses_tracker:
                    num_neg = losses_tracker['num_neg'].val
                    ratio = num_neg / max(num_pos, 1)
                    block5 = '\n\t[Sample Stats] pos={:.0f} neg={:.0f} neg/pos_ratio={:.1f}'.format(
                        num_pos, num_neg, ratio
                    )
                else:
                    block5 = '\n\t[Sample Stats] pos={:.0f}'.format(num_pos)


            block6 = '\n\t[Grad Stats] norm={:.4f} max={:.6f}'.format(
                grad_tracker['grad_norm'].avg, grad_tracker['grad_max'].avg
            )

            if grad_tracker['grad_norm'].val < 1e-7:
                block6 += ' ⚠️ WARN: grad_norm too small!'
            elif grad_tracker['grad_norm'].val > 100:
                block6 += ' ⚠️ WARN: grad_norm too large!'

            print('\t'.join([block1, block2, block3, block4]) + block5 + block6)


    lr = scheduler.get_last_lr()[0]
    print("[Train]: Epoch {:d} finished with lr={:.8f}\n".format(curr_epoch, lr))
    return


def valid_one_epoch(
    val_loader,
    model,
    curr_epoch,
    ext_score_file = None,
    evaluator = None,
    output_file = None,
    tb_writer = None,
    print_freq = 20,
    gt_file = None,
    subset= 'test',
    tiou_thre=np.linspace(0.5, 1.0, 11),
    max_avg_nr_proposal=100,
    dataset_name=''
):
    """Test the model on the validation set"""

    assert (evaluator is not None) or (output_file is not None)


    batch_time = AverageMeter()

    model.eval()

    results = {
        'video-id': [],
        't-start' : [],
        't-end': [],
        'label': [],
        'score': []
    }


    video_cls_preds = []
    video_cls_labels = []


    start = time.time()
    for iter_idx, video_list in enumerate(val_loader, 0):

        with torch.no_grad():
            output = model(video_list)


            num_vids = len(output)
            for vid_idx in range(num_vids):
                if output[vid_idx]['segments'].shape[0] > 0:
                    results['video-id'].extend(
                        [output[vid_idx]['video_id']] *
                        output[vid_idx]['segments'].shape[0]
                    )
                    results['t-start'].append(output[vid_idx]['segments'][:, 0])
                    results['t-end'].append(output[vid_idx]['segments'][:, 1])
                    results['label'].append(output[vid_idx]['labels'])
                    results['score'].append(output[vid_idx]['scores'])


                if 'vid_prob' in output[vid_idx]:
                    video_cls_preds.append(output[vid_idx]['vid_prob'])

                    if 'video_label' in video_list[vid_idx]:
                        gt_label = video_list[vid_idx]['video_label']
                        if isinstance(gt_label, torch.Tensor):
                            gt_label = gt_label.item()
                        video_cls_labels.append(gt_label)
                    else:

                        has_segments = video_list[vid_idx].get('segments') is not None
                        video_cls_labels.append(1.0 if has_segments else 0.0)


        if (iter_idx != 0) and iter_idx % (print_freq) == 0:

            torch.cuda.synchronize()
            batch_time.update((time.time() - start) / print_freq)
            start = time.time()


            print('Test: [{0:05d}/{1:05d}]\t'
                  'Time {batch_time.val:.2f} ({batch_time.avg:.2f})'.format(
                  iter_idx, len(val_loader), batch_time=batch_time))


    if len(video_cls_preds) > 0 and len(video_cls_labels) > 0:
        video_cls_preds = np.array(video_cls_preds)
        video_cls_labels = np.array(video_cls_labels)


        best_threshold = 0.5
        best_accuracy = 0.0
        best_f1 = 0.0

        thresholds_to_try = np.concatenate([
            np.linspace(0.001, 0.1, 10),
            np.linspace(0.1, 0.9, 17),
        ])

        threshold_results = []
        for thresh in thresholds_to_try:
            pred_binary = (video_cls_preds >= thresh).astype(float)
            tp_t = np.sum((pred_binary == 1) & (video_cls_labels == 1))
            fp_t = np.sum((pred_binary == 1) & (video_cls_labels == 0))
            tn_t = np.sum((pred_binary == 0) & (video_cls_labels == 0))
            fn_t = np.sum((pred_binary == 0) & (video_cls_labels == 1))

            acc_t = (tp_t + tn_t) / (tp_t + tn_t + fp_t + fn_t + 1e-8)
            prec_t = tp_t / (tp_t + fp_t + 1e-8)
            rec_t = tp_t / (tp_t + fn_t + 1e-8)
            f1_t = 2 * prec_t * rec_t / (prec_t + rec_t + 1e-8)

            threshold_results.append({
                'threshold': thresh, 'accuracy': acc_t, 'f1': f1_t,
                'precision': prec_t, 'recall': rec_t,
                'tp': int(tp_t), 'fp': int(fp_t), 'tn': int(tn_t), 'fn': int(fn_t)
            })

            if acc_t > best_accuracy:
                best_accuracy = acc_t
                best_threshold = thresh
                best_f1 = f1_t


        best_result = [r for r in threshold_results if r['threshold'] == best_threshold][0]
        tp, fp, tn, fn = best_result['tp'], best_result['fp'], best_result['tn'], best_result['fn']
        accuracy = best_result['accuracy']
        precision = best_result['precision']
        recall = best_result['recall']
        f1_score = best_result['f1']


        fixed_result = [r for r in threshold_results if abs(r['threshold'] - 0.5) < 0.01][0]


        try:
            from sklearn.metrics import roc_auc_score
            if len(np.unique(video_cls_labels)) > 1:
                auc_roc = roc_auc_score(video_cls_labels, video_cls_preds)
            else:
                auc_roc = 0.0
        except:
            auc_roc = 0.0

        print("\n" + "="*70)
        print("Video-level Classification Metrics (Adaptive Threshold)")
        print("="*70)
        print(f"  Total Videos: {len(video_cls_labels)}")
        print(f"  Positive (Fake): {int(np.sum(video_cls_labels))}")
        print(f"  Negative (Real): {int(len(video_cls_labels) - np.sum(video_cls_labels))}")
        print("-"*70)
        print(f"  [Fixed Threshold=0.5]")
        print(f"    Accuracy: {fixed_result['accuracy']*100:.2f}%, F1: {fixed_result['f1']*100:.2f}%")
        print(f"    TP: {fixed_result['tp']}, FP: {fixed_result['fp']}, TN: {fixed_result['tn']}, FN: {fixed_result['fn']}")
        print("-"*70)
        print(f"  [Best Adaptive Threshold={best_threshold:.2f}]")
        print(f"    TP: {int(tp)}, FP: {int(fp)}, TN: {int(tn)}, FN: {int(fn)}")
        print(f"    Accuracy:  {accuracy*100:.2f}%  (best)")
        print(f"    Precision: {precision*100:.2f}%")
        print(f"    Recall:    {recall*100:.2f}%")
        print(f"    F1-Score:  {f1_score*100:.2f}%")
        print(f"    AUC-ROC:   {auc_roc*100:.2f}%")
        print("="*70 + "\n")


        if tb_writer is not None:
            tb_writer.add_scalar('validation/video_cls_accuracy', accuracy, curr_epoch)
            tb_writer.add_scalar('validation/video_cls_accuracy_fixed', fixed_result['accuracy'], curr_epoch)
            tb_writer.add_scalar('validation/video_cls_precision', precision, curr_epoch)
            tb_writer.add_scalar('validation/video_cls_recall', recall, curr_epoch)
            tb_writer.add_scalar('validation/video_cls_f1', f1_score, curr_epoch)
            tb_writer.add_scalar('validation/video_cls_auc', auc_roc, curr_epoch)
            tb_writer.add_scalar('validation/best_threshold', best_threshold, curr_epoch)


    if len(results['t-start']) > 0:
        results['t-start'] = torch.cat(results['t-start']).numpy()
        results['t-end'] = torch.cat(results['t-end']).numpy()
        results['label'] = torch.cat(results['label']).numpy()
        results['score'] = torch.cat(results['score']).numpy()
    else:
        results['t-start'] = np.array([])
        results['t-end'] = np.array([])
        results['label'] = np.array([])
        results['score'] = np.array([])

    mAP = 0.0
    if evaluator is not None:
        if ext_score_file is not None and isinstance(ext_score_file, str):
            results = postprocess_results(results, ext_score_file)

        if len(results['t-start']) > 0:
            _, mAP, _ = evaluator.evaluate(results, verbose=True)
    elif 'json' in output_file:
        if len(results['t-start']) > 0:
            mAP,_ = run_evaluation(results, gt_file,output_file, max_avg_nr_proposal=max_avg_nr_proposal,tiou_thre=tiou_thre,subset=subset,cls_score_file=ext_score_file)
        else:
            print("No localization results to evaluate (all videos have no detected segments)")
    else:

        with open(output_file, "wb") as f:
            pickle.dump(results, f)
        mAP = 0.0



    if tb_writer is not None:
        tb_writer.add_scalar('validation/mAP', mAP, curr_epoch)
    return mAP
