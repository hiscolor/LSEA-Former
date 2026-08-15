import numpy as np
import matplotlib.pyplot as plt
from .eval_proposal import ANETproposal
from .eval_detection import ANETdetection

import os
from joblib import Parallel, delayed
import json
import pandas as pd
















def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data

def plot_metric(args, average_nr_proposals, average_recall, recall, tiou_thresholds=np.linspace(0.5, 1.0, 11)):
    fn_size = 14
    plt.figure(num=None, figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    colors = ['k', 'r', 'yellow', 'b', 'c', 'm', 'b', 'pink', 'lawngreen', 'indigo']
    area_under_curve = np.zeros_like(tiou_thresholds)
    for i in range(recall.shape[0]):
        area_under_curve[i] = np.trapz(recall[i], average_nr_proposals)

    for idx, tiou in enumerate(tiou_thresholds[::2]):
        ax.plot(average_nr_proposals, recall[2 * idx, :], color=colors[idx + 1],
                label="tiou=[" + str(tiou) + "],area=" + str(int(area_under_curve[2 * idx] * 100) / 100.),
                linewidth=4, linestyle='-', marker=None)

    ax.plot(average_nr_proposals, average_recall, color=colors[0],
            label="tiou=0.5:0.1:1.0," + "area=" + str(int(np.trapz(average_recall, average_nr_proposals) * 100) / 100.),
            linewidth=4, linestyle='-', marker=None)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend([handles[-1]] + handles[:-1], [labels[-1]] + labels[:-1], loc='best')

    plt.ylabel('Average Recall', fontsize=fn_size)
    plt.xlabel('Average Number of Proposals per Video', fontsize=fn_size)
    plt.grid(b=True, which="both")
    plt.ylim([0, 1.0])
    plt.setp(plt.axes().get_xticklabels(), fontsize=fn_size)
    plt.setp(plt.axes().get_yticklabels(), fontsize=fn_size)
    plt.savefig(os.path.join(args.output["work_dir"],args.model_name,args.output["output_path"], args.eval["save_fig_path"]))


def evaluation_proposal(gt_filename,pred_filename,tious,subset,max_avg_nr_proposal=100):
    anet_proposal = ANETproposal(gt_filename,pred_filename,
                                tiou_thresholds=tious, max_avg_nr_proposals=max_avg_nr_proposal,
                                subset=subset, verbose=True, check_status=False)

    anet_proposal.evaluate()

    recall = anet_proposal.recall
    average_recall = anet_proposal.avg_recall
    average_nr_proposal = anet_proposal.proposals_per_video





    result = f'Proposal: AR@10 {np.mean(recall[:, 9])*100:.3f} \t'
    result+=f'AR@20 {np.mean(recall[:, 19])*100:.3f} \t'
    result+=f'AR@50 {np.mean(recall[:, 49])*100:.3f} \t'
    result+=f'AR@100 {np.mean(recall[:, 99])*100:.3f} \t'
    with open(pred_filename.replace('.json','.txt'), 'a') as fobj:
        fobj.write(f'{result}\n')
    return (np.mean(recall[:, 9])+np.mean(recall[:, 19])+np.mean(recall[:, 49])+np.mean(recall[:, 99]))/4*100

def evaluation_detection(gt_filename,pred_filename,tious,subset):
    anet_detection = ANETdetection(
    ground_truth_filename=gt_filename,
    prediction_filename=pred_filename,
    tiou_thresholds=tious,
    subset=subset, verbose=True, check_status=False)
    anet_detection.evaluate()

    mAP_at_tIoU = [f'mAP@{t:.2f} {mAP*100:.3f}' for t, mAP in zip(anet_detection.tiou_thresholds, anet_detection.mAP)]
    results = f'Detection: average-mAP {anet_detection.average_mAP*100:.3f} {" ".join(mAP_at_tIoU)}'
    print(results)
    with open(pred_filename.replace('.json','.txt'), 'a') as fobj:
        fobj.write(f'{results}\n')
    return np.mean(anet_detection.mAP)*100


def evaluation_video_level(gt_filename, pred_filename, subset, score_threshold='auto'):
    """
    视频级别的分类评估: 判断视频是真实还是伪造
    返回 Precision, Recall, F1-score, Accuracy

    参数:
        score_threshold: float或'auto'. 如果是'auto'，搜索最优 F1 阈值
    """
    with open(gt_filename, 'r') as f:
        gt_data = json.load(f)

    if 'database' in gt_data:
        gt_data = gt_data['database']

    with open(pred_filename, 'r') as f:
        pred_data = json.load(f)

    pred_results = pred_data['results']


    video_gt_labels = []
    video_max_scores = []

    for vid_key, vid_info in gt_data.items():
        vid_split = vid_info.get('split', vid_info.get('subset', '')).lower()
        if vid_split != subset:
            continue


        annotations = vid_info.get('annotations', vid_info.get('fake_periods', []))
        gt_has_fake = 1 if (annotations and len(annotations) > 0) else 0
        video_gt_labels.append(gt_has_fake)


        if vid_key in pred_results and pred_results[vid_key]:
            max_score = max([p['score'] for p in pred_results[vid_key]])
        else:
            max_score = 0.0
        video_max_scores.append(max_score)

    video_gt_labels = np.array(video_gt_labels)
    video_max_scores = np.array(video_max_scores)


    if score_threshold == 'auto':

        best_f1 = 0
        best_thresh = 0.5


        candidate_thresholds = np.unique(video_max_scores)
        candidate_thresholds = np.concatenate([[0], candidate_thresholds, [1.0]])

        for thresh in candidate_thresholds:
            pred_labels = (video_max_scores > thresh).astype(int)
            tp = np.sum((pred_labels == 1) & (video_gt_labels == 1))
            fp = np.sum((pred_labels == 1) & (video_gt_labels == 0))
            fn = np.sum((pred_labels == 0) & (video_gt_labels == 1))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh

        score_threshold = best_thresh
        print(f'[Info] 使用最优 F1 阈值: {score_threshold:.6f} (best F1={best_f1:.4f})')

    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    for vid_key, vid_info in gt_data.items():
        vid_split = vid_info.get('split', vid_info.get('subset', '')).lower()
        if vid_split != subset:
            continue


        annotations = vid_info.get('annotations', vid_info.get('fake_periods', []))
        gt_has_fake = len(annotations) > 0 if annotations else False


        pred_has_fake = False
        if vid_key in pred_results and pred_results[vid_key]:
            max_score = max([p['score'] for p in pred_results[vid_key]])
            pred_has_fake = max_score > score_threshold


        if gt_has_fake and pred_has_fake:
            true_positives += 1
        elif gt_has_fake and not pred_has_fake:
            false_negatives += 1
        elif not gt_has_fake and pred_has_fake:
            false_positives += 1
        else:
            true_negatives += 1


    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)

    print(f'\n[Video-Level Classification Metrics] (threshold={score_threshold:.6f})')
    print(f'  TP={true_positives}, FP={false_positives}, TN={true_negatives}, FN={false_negatives}')
    print(f'  Precision: {precision*100:.2f}%')
    print(f'  Recall: {recall*100:.2f}%')
    print(f'  F1-score: {f1_score*100:.2f}%')
    print(f'  Accuracy: {accuracy*100:.2f}%')

    video_level_results = f'Video-Level: Precision {precision*100:.3f} Recall {recall*100:.3f} F1 {f1_score*100:.3f} Accuracy {accuracy*100:.3f} (threshold={score_threshold:.6f})'
    with open(pred_filename.replace('.json','.txt'), 'a') as fobj:
        fobj.write(f'{video_level_results}\n')

    return precision, recall, f1_score, accuracy


def detection_thread(vid,pred_data,cls_data_cls):
    proposal_list = []
    old_df = pred_data[pred_data.video_name == vid]

    df = pd.DataFrame()
    df['score'] = old_df.score.values[:]
    df['label'] = old_df.label.values[:]
    df['xmin'] = old_df.xmin.values[:]
    df['xmax'] = old_df.xmax.values[:]
    best_score=np.max(cls_data_cls[vid])
    for j in range(min(100, len(df))):
            tmp_proposal = {}
            tmp_proposal["label"] = 'Fake'
            tmp_proposal["score"] = float(df.score.values[j])*best_score
            tmp_proposal["segment"] = [max(0, df.xmin.values[j]),
                                    df.xmax.values[j]]
            proposal_list.append(tmp_proposal)
    return {vid: proposal_list}

def post_process_multi(pred_data,output_file,cls_score_file=None):

    pred_videos = list(pred_data.video_name.values[:])
    pred_videos = set(pred_videos)
    cls_data_cls = {}
    if cls_score_file is not None:
        best_cls = load_json(cls_score_file)

        for idx, vid in enumerate(pred_videos):
            if vid in pred_videos:
                cls_data_cls[vid] = best_cls[vid]
    else:
        for idx, vid in enumerate(pred_videos):
            if vid in pred_videos:
                cls_data_cls[vid] = [1,1]

    parallel = Parallel(n_jobs=16, prefer="processes")
    detection = parallel(delayed(detection_thread)(vid, pred_data,cls_data_cls)
                        for vid in pred_videos)
    detection_dict = {}
    [detection_dict.update(d) for d in detection]

    output_dict = {"version": "ANET v1.3, Lavdf", "results": detection_dict, "external_data": {}}

    with open(output_file, "w") as out:
        json.dump(output_dict, out)


def run_evaluation(preds, ground_truth_file, proposal_file, dataset_name='',
                   max_avg_nr_proposal=100,
                   tiou_thre=np.linspace(0.5, 1.0, 11), subset='test',cls_score_file=None):
    preds = pd.DataFrame({
                'video_name' : preds['video-id'],
                'xmin' : preds['t-start'].tolist(),
                'xmax': preds['t-end'].tolist(),
                'label': preds['label'].tolist(),
                'score': preds['score'].tolist()
            })
    print("saving detection results...")
    post_process_multi(preds,proposal_file,cls_score_file)
    print("evaluion detection results...")
    mAP=evaluation_detection(ground_truth_file,proposal_file,tiou_thre,subset)
    print("evaluion proposal results...")
    mAR=evaluation_proposal(ground_truth_file,proposal_file,tiou_thre,subset,max_avg_nr_proposal)
    return mAP,mAR
