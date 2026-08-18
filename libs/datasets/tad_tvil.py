import os
import json
import h5py
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.nn import functional as F

from .datasets import register_dataset
from .data_utils import truncate_feats
from ..utils import remove_duplicate_annotations

@register_dataset("tad_tvil")
class TAD_TVILDataset(Dataset):
    def __init__(
        self,
        is_training,
        split,
        feat_folder,
        audio_feat_folder,
        json_file,
        feat_stride,
        num_frames,
        default_fps,
        downsample_rate,
        max_seq_len,
        trunc_thresh,
        crop_ratio,
        input_dim,
        audio_input_dim,
        num_classes,
        file_prefix,
        file_ext,
        audio_file_ext,
        force_upsampling,

        feb_feat_folder=None,
        enable_feb=False,
        source_filter=None,
        video_allowlist=None,
    ):

        assert os.path.exists(feat_folder) and os.path.exists(json_file)
        assert isinstance(split, tuple) or isinstance(split, list)
        assert crop_ratio == None or len(crop_ratio) == 2
        self.feat_folder = feat_folder
        self.audio_feat_folder= audio_feat_folder
        if file_prefix is not None:
            self.file_prefix = file_prefix
        else:
            self.file_prefix = ''
        self.file_ext = file_ext
        self.audio_file_ext=audio_file_ext
        self.json_file = json_file


        self.force_upsampling = force_upsampling


        self.split = split
        self.is_training = is_training


        self.feat_stride = feat_stride
        self.num_frames = num_frames
        self.input_dim = input_dim
        self.audio_input_dim=audio_input_dim
        self.default_fps = default_fps
        self.downsample_rate = downsample_rate
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.num_classes = num_classes
        self.label_dict = {'Fake':0}
        self.crop_ratio = crop_ratio


        self.feb_feat_folder = feb_feat_folder
        self.enable_feb = enable_feb
        self._feb_warning_shown = False
        self.source_filter = source_filter
        self.video_allowlist = None
        if video_allowlist:
            if isinstance(video_allowlist, (list, tuple, set)):
                self.video_allowlist = set(video_allowlist)
            elif os.path.isfile(video_allowlist):
                if video_allowlist.endswith('.json'):
                    with open(video_allowlist) as f:
                        obj = json.load(f)
                    if isinstance(obj, dict) and 'videos' in obj:
                        self.video_allowlist = set(obj['videos'])
                    elif isinstance(obj, list):
                        self.video_allowlist = set(obj)
                    else:
                        raise ValueError(f'Unsupported allowlist json: {video_allowlist}')
                else:
                    with open(video_allowlist) as f:
                        self.video_allowlist = {
                            ln.strip() for ln in f if ln.strip() and not ln.startswith('#')
                        }
            print(f"video_allowlist: {len(self.video_allowlist)} ids from {video_allowlist}")
        self._vid_to_source = None
        if self.source_filter:
            info_path = os.path.join(os.path.dirname(json_file), '..', 'dataset_source_info.json')
            info_path = os.path.normpath(info_path)
            if os.path.isfile(info_path):
                with open(info_path, 'r') as f:
                    src_info = json.load(f)
                self._vid_to_source = {}
                for src, splits in src_info.items():
                    for split_name, vids in splits.items():
                        for vid in vids:
                            self._vid_to_source[vid] = src
                print(f"source_filter={self.source_filter!r}, mapped {len(self._vid_to_source)} video IDs")
            else:
                print(f"[WARN] source_filter set but missing {info_path}")


        dict_db = self._load_json_db(self.json_file)

        assert (num_classes == 1)
        self.data_list = dict_db


        self.db_attributes = {
            'dataset_name': 'VIL',
            'tiou_thresholds': np.linspace(0.5, 0.95, 10),
            'empty_label_ids': []
        }
        print("{} subset has {} videos".format(self.split,len(self.data_list)))
        if self.enable_feb:
            print(f"FEB enabled, loading from: {self.feb_feat_folder}")
    def get_attributes(self):
        return self.db_attributes

    def _load_json_db(self, json_file):

        with open(json_file, 'r') as fid:
            json_data = json.load(fid)


        if 'database' in json_data:
            json_db = json_data['database']
        else:
            json_db = json_data

        dict_db = tuple()

        for key, value in json_db.items():

            video_split = value.get('subset', value.get('split', '')).lower()
            if video_split not in self.split:
                continue


            video_id = key if 'file' not in value else value['file'][:-4]

            if self.source_filter and self._vid_to_source is not None:
                if self._vid_to_source.get(video_id) != self.source_filter:
                    continue

            if (
                self.is_training
                and self.video_allowlist is not None
                and video_id not in self.video_allowlist
            ):
                continue

            if isinstance(self.file_prefix, list):
                assert len(self.file_prefix) == 2
                feat_file = os.path.join(self.feat_folder, self.file_prefix[0], video_split,
                                        video_id + self.file_ext)
            else:
                feat_file = os.path.join(self.feat_folder, self.file_prefix, video_split,
                                        video_id + self.file_ext)
                if not os.path.exists(feat_file):
                    feat_file = os.path.join(self.feat_folder, video_id + self.file_ext)
            if not os.path.exists(feat_file):
                continue


            if 'fps' in value:
                fps = value['fps']
            elif self.default_fps is not None:
                fps = self.default_fps
            elif 'video_frames' in value:
                fps = value['video_frames'] / value['duration']
            else:
                assert False, "Unknown video FPS."
            duration = value['duration']


            annotations = value.get('annotations')
            if annotations and len(annotations) > 0:
                num_acts = len(annotations)
                segments = np.zeros([num_acts, 2], dtype=np.float32)
                labels = np.zeros([num_acts, ], dtype=np.int64)
                for idx, act in enumerate(annotations):
                    if isinstance(act, dict):
                        segments[idx][0] = act['segment'][0]
                        segments[idx][1] = act['segment'][1]
                    else:
                        segments[idx][0] = act[0]
                        segments[idx][1] = act[1]
                    labels[idx] = 0

                video_label = 1.0
            else:
                segments = None
                labels = None

                video_label = 0.0

            dict_db += ({'id': video_id,
                         'fps': fps,
                         'duration': duration,
                         'split': video_split,
                         'segments': segments,
                         'labels': labels,
                         'video_label': video_label,
            }, )

        return dict_db

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):



        video_item = self.data_list[idx]



        if isinstance(self.file_prefix, list):
            filename1 = os.path.join(self.feat_folder, self.file_prefix[0], video_item['split'],
                                    video_item['id'] + self.file_ext)
            feats1 = np.load(filename1).astype(np.float32)
            filename2 = os.path.join(self.feat_folder, self.file_prefix[1], video_item['split'],
                                    video_item['id'] + self.file_ext)
            feats2 = np.load(filename2).astype(np.float32)
            if feats1.shape[0] != feats2.shape[0]:
                feature_length = max(feats1.shape[0], feats2.shape[0])
                feats1 = np.resize(feats1, (feature_length, feats1.shape[1]))
                feats2 = np.resize(feats2, (feature_length, feats2.shape[1]))
            feats = np.concatenate((feats1, feats2), axis=1)
        else:
            filename = os.path.join(self.feat_folder, self.file_prefix, video_item['split'],
                                    video_item['id'] + self.file_ext)
            if not os.path.exists(filename):
                filename = os.path.join(self.feat_folder, video_item['id'] + self.file_ext)
            feats = np.load(filename).astype(np.float32)
        audio_feats= None
        if self.audio_feat_folder is not None:
            audio_filename = os.path.join(self.audio_feat_folder,video_item['split'],
                        video_item['id'] + self.file_ext)
            audio_feats = np.load(audio_filename)




        if self.feat_stride > 0 and (not self.force_upsampling):

            feat_stride, num_frames = self.feat_stride, self.num_frames

            if self.downsample_rate > 1:
                feats = feats[::self.downsample_rate, :]
                feat_stride = self.feat_stride * self.downsample_rate

        elif self.feat_stride > 0 and self.force_upsampling:
            feat_stride = float(
                (feats.shape[0] - 1) * self.feat_stride + self.num_frames
            ) / self.max_seq_len

            num_frames = feat_stride

        else:

            seq_len = feats.shape[0]
            assert seq_len <= self.max_seq_len
            if self.force_upsampling:

                seq_len = self.max_seq_len
            feat_stride = video_item['duration'] * video_item['fps'] / seq_len

            num_frames = feat_stride
        feat_offset = 0.5 * num_frames / feat_stride


        feats = torch.from_numpy(np.ascontiguousarray(feats.transpose()))


        if (feats.shape[-1] != self.max_seq_len) and self.force_upsampling:
            resize_feats = F.interpolate(
                feats.unsqueeze(0),
                size=self.max_seq_len,
                mode='linear',
                align_corners=False
            )
            feats = resize_feats.squeeze(0)

        if (self.audio_feat_folder is not None):
            audio_feats = torch.from_numpy(np.ascontiguousarray(audio_feats.transpose()))
            resize_audio_feats = F.interpolate(
                audio_feats.unsqueeze(0),
                size=feats.shape[1],
                mode='linear',
                align_corners=False
            )
            audio_feats = resize_audio_feats.squeeze(0)
            feats=torch.cat([feats,audio_feats],dim=0)


        if video_item['segments'] is not None:
            segments = torch.from_numpy(
                video_item['segments'] * video_item['fps'] / feat_stride - feat_offset
            )
            labels = torch.from_numpy(video_item['labels'])


            if self.is_training:
                vid_len = feats.shape[1] + feat_offset
                valid_seg_list, valid_label_list = [], []
                for seg, label in zip(segments, labels):
                    if seg[0] >= vid_len:

                        print(f"[DEBUG] {video_item['id']}: seg[0]={seg[0].item():.2f} >= vid_len={vid_len:.2f}, skipped")
                        continue

                    ratio = (
                        (min(seg[1].item(), vid_len) - seg[0].item())
                        / (seg[1].item() - seg[0].item())
                    )
                    if ratio >= self.trunc_thresh:
                        valid_seg_list.append(seg.clamp(max=vid_len))

                        valid_label_list.append(label.view(1))
                    else:
                        print(f"[DEBUG] {video_item['id']}: ratio={ratio:.4f} < trunc_thresh={self.trunc_thresh}, seg=[{seg[0].item():.2f}, {seg[1].item():.2f}], vid_len={vid_len:.2f}")

                if len(valid_seg_list) > 0:
                    segments = torch.stack(valid_seg_list, dim=0)
                    labels = torch.cat(valid_label_list)
                else:
                    print(f"[WARN] {video_item['id']}: all segments filtered! original_segments={video_item['segments']}, feat_shape={feats.shape}, vid_len={vid_len:.2f}, feat_stride={feat_stride:.2f}, feat_offset={feat_offset:.2f}")
                    segments, labels = None, None
        else:
            segments, labels = None, None




        data_dict = {'video_id'        : video_item['id'],
                     'feats'           : feats,
                     'segments'        : segments,
                     'labels'          : labels,
                     'fps'             : video_item['fps'],
                     'duration'        : video_item['duration'],
                     'feat_stride'     : feat_stride,
                     'feat_num_frames' : num_frames,
                     'video_label'     : video_item['video_label'],
        }


        if self.enable_feb and self.feb_feat_folder is not None:
            feb_path = os.path.join(self.feb_feat_folder, video_item['id'] + '.npy')
            if os.path.exists(feb_path):
                feb_feats = np.load(feb_path).astype(np.float32)


                vmae_L = feats.shape[1]
                feb_L = feb_feats.shape[0]


                if feb_L != vmae_L:
                    P, E = feb_feats.shape[1], feb_feats.shape[2]

                    feb_flat = torch.from_numpy(
                        feb_feats.reshape(feb_L, -1).transpose().copy()
                    ).unsqueeze(0).float()
                    feb_resized = F.interpolate(
                        feb_flat, size=vmae_L, mode='linear', align_corners=False
                    )
                    feb_feats = feb_resized.squeeze(0).T.numpy().reshape(vmae_L, P, E)

                data_dict['freq_evidence'] = torch.from_numpy(np.ascontiguousarray(feb_feats))
            else:
                if not self._feb_warning_shown:
                    print(f"[WARNING] FEB feature not found for {video_item['id']}: {feb_path}")
                    self._feb_warning_shown = True
                vmae_L = feats.shape[1]
                data_dict['freq_evidence'] = torch.zeros(vmae_L, 196, 14, dtype=torch.float32)



        if self.is_training and (segments is not None):
            data_dict = truncate_feats(
                data_dict, self.max_seq_len, self.trunc_thresh, feat_offset, self.crop_ratio
            )

        return data_dict
