# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import numpy as np
import os
from PIL import Image
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as tv_F
from torchvision.utils import save_image

class DatasetFromClipPaths(Dataset):
    def __init__(self, clip_paths, with_labels):
        super().__init__()
        #TODO currently doesn't support loading of annotations
        self.with_labels = with_labels
        if self.with_labels:
            self.clip_paths, self.clip_labels = clip_paths
        else:
            self.clip_paths = clip_paths
        
        self.normalize_stats = {'mean' : [0.500, 0.436, 0.396], 'std' : [0.145, 0.143, 0.138]} # orbit mean train frame
        
    def __getitem__(self, index):
        clip = []
        for frame_path in self.clip_paths[index]:
            frame = self.load_and_transform_frame(frame_path)
            clip.append(frame)
    
        if self.with_labels:
            return torch.stack(clip, dim=0), self.clip_labels[index]
        else:
            return torch.stack(clip, dim=0)
    
    def load_and_transform_frame(self, frame_path):
        """
        Function to load and transform frame.
        :param frame_path: (str) Path to frame.
        :return: (torch.Tensor) Loaded and transformed frame.
        """
        frame = Image.open(frame_path)
        frame = tv_F.to_tensor(frame)
        return tv_F.normalize(frame, mean=self.normalize_stats['mean'], std=self.normalize_stats['std'])

    def __len__(self):
        return len(self.clip_paths)

def get_clip_loader(clips, batch_size, with_labels=False):
    if isinstance(clips[0], np.ndarray):
        clips_dataset = DatasetFromClipPaths(clips, with_labels=with_labels)
        return DataLoader(clips_dataset,
                      batch_size=batch_size,
                      num_workers=8,
                      pin_memory=True,
                      prefetch_factor=8,
                      persistent_workers=True)

    elif isinstance(clips[0], torch.Tensor):
        if with_labels:
            return list(zip(clips[0].split(batch_size), clips[1].split(batch_size)))
        else: 
            return clips.split(batch_size)

def attach_frame_history(frames, history_length):
    
    if isinstance(frames, np.ndarray):
        return attach_frame_history_paths(frames, history_length)
    elif isinstance(frames, torch.Tensor):
        return attach_frame_history_tensor(frames, history_length)

def attach_frame_history_paths(frame_paths, history_length):
    """
    Function to attach the immediate history of history_length frames to each frame in an array of frame paths.
    :param frame_paths: (np.ndarray) Frame paths.
    :param history_length: (int) Number of frames of history to append to each frame.
    :return: (np.ndarray) Frame paths with attached frame history.
    """
    # pad with first frame so that frames 0 to history_length-1 can be evaluated
    frame_paths = np.concatenate([np.repeat(frame_paths[0], history_length-1), frame_paths])
    
    # for each frame path, attach its immediate history of history_length frames
    frame_paths = [ frame_paths ]
    for l in range(1, history_length):
        frame_paths.append( np.roll(frame_paths[0], shift=-l, axis=0) )
    frame_paths_with_history = np.stack(frame_paths, axis=1) # of size num_clips x history_length
    
    if history_length > 1:
        return frames_with_history[:-(history_length-1)] # frames have wrapped around, remove last (history_length - 1) frames
    else:
        return frames_with_history

def attach_frame_history_tensor(frames, history_length):
    """
    Function to attach the immediate history of history_length frames to each frame in a tensor of frame data.
    param frames: (torch.Tensor) Frames.
    :param history_length: (int) Number of frames of history to append to each frame.
    :return: (torch.Tensor) Frames with attached frame history.
    """
    # pad with first frame so that frames 0 to history_length-1 can be evaluated
    frame_0 = frames.narrow(0, 0, 1)
    frames = torch.cat((frame_0.repeat(history_length-1, 1, 1, 1), frames), dim=0)

    # for each frame, attach its immediate history of history_length frames
    frames = [ frames ]
    for l in range(1, history_length):
        frames.append( frames[0].roll(shifts=-l, dims=0) )
    frames_with_history = torch.stack(frames, dim=1) # of size num_clips x history_length
    
    if history_length > 1:
        return frames_with_history[:-(history_length-1)] # frames have wrapped around, remove last (history_length - 1) frames
    else:
        return frames_with_history

def unpack_task(task_dict, device, context_to_device=True, target_to_device=False, preload_clips=False, remove_target_frames_without_object=False):
    #max_context = 17
    context_clips = task_dict['context_clips']#[0:max_context]
    context_paths = task_dict['context_paths']#[0:max_context]
    context_labels = task_dict['context_labels']#[0:max_context]
    context_annotations = task_dict['context_annotations']
    target_clips = task_dict['target_clips']
    target_paths = task_dict['target_paths']
    target_labels = task_dict['target_labels']
    target_annotations = task_dict['target_annotations']
    object_list = task_dict['object_list']

    if context_to_device and isinstance(context_labels, torch.Tensor):
        context_labels = context_labels.to(device)
    if target_to_device and isinstance(target_labels, torch.Tensor):
        target_labels = target_labels.to(device)

    # Remove frames from the target set if they're annotated as not having the target object in frame
    if remove_target_frames_without_object:
        num_videos = len(target_clips)
        for v in range(num_videos):
            target_annotations_per_video = target_annotations[v]
            assert 'object_not_present_issue' in target_annotations_per_video.keys()
            object_in_frame_mask = (target_annotations_per_video['object_not_present_issue'] == 0).squeeze()
            target_clips[v] = target_clips[v][object_in_frame_mask]
            target_paths[v] = target_paths[v][object_in_frame_mask]
            # Labels are already just one per video, no need to shorten
  
    if preload_clips:
        return context_clips, context_paths, context_labels, target_clips, target_paths, target_labels, object_list
    else:
        return context_paths, context_paths, context_labels, target_paths, target_paths, target_labels, object_list

def process_annotations_dict(annotations_dict):
    noise_tensor = None
    for key in annotations_dict.keys():
        not_nan_annotation = torch.nan_to_num(annotations_dict[key], nan=0.0)
        if noise_tensor is None:
            noise_tensor = not_nan_annotation
        else:
            noise_tensor += not_nan_annotation
    return noise_tensor

#keep_indices, "keep")

def save_selected_frames(context_clip_paths, annotations_dict, context_labels, object_list, output_dir, index_list=None, selection_name=""):
    if selection_name != "":
        output_path = os.path.join(output_dir, selection_name)
        if not os.path.exists(output_path):
            os.mkdir(output_path)
    else:
        output_path = output_dir
    if index_list is None:
        index_list = torch.tensor(range(len(context_clip_paths)), dtype=torch.int64)

    # Save out the entire selection
    # object_list[context_labels[index_list]]
    save_frame_grid(context_clip_paths[index_list].squeeze(axis=1), output_path, "sample")

    # For each noise type, save out all those frames as a grid:
    for key in annotations_dict.keys():
        # Handle nan values
        not_nan_annotations = torch.nan_to_num(annotations_dict[key], nan=0.0)
        # Preserve the batch dimension, in case there is only image with this noise type
        not_nan_annotations = not_nan_annotations.squeeze(dim=1).squeeze(dim=1)
        # Filter out any annotations that aren't in our subselection:
        not_nan_annotations = not_nan_annotations[index_list]
        issue_mask = not_nan_annotations != 0 # True where issue exists
        # issue_mask can now be applied to anything that has been filtered by index_list
        issue_paths = (context_clip_paths[index_list]).squeeze(axis=1)[issue_mask]

        #issue_labels = context_labels[index_list][issue_mask]
        
        if issue_paths.size == 0:
            continue
        save_frame_grid(issue_paths,  output_path, key)

def save_frame_grid(clip_paths, output_path, file_prefix):
    all_frames = []
    for p in range(clip_paths.size):
        frame_path = clip_paths[p]
        user = frame_path.split('/')[-5]
        frame = Image.open(frame_path)
        frame = tv_F.to_tensor(frame)
        all_frames.append(frame)
    filename = "{}_{}.png".format(user, file_prefix)
    # Save them out
    save_image(all_frames, os.path.join(output_path, filename))


def sample_noisy_frames(context_clip_paths, annotations_dict, context_labels, object_list, output_dir):
    
    for key in annotations_dict.keys():
        # Handle nan values
        not_nan_annotations = torch.nan_to_num(annotations_dict[key], nan=0.0)
        issue_mask = (not_nan_annotations != 0).squeeze() # True where issue exists
        # Select 10 images that have this issu to save out
        #issue_frames = context_clips[issue_mask]
        #issue_frames = issue_frames[0:min(10, len(issue_frames))]
        issue_paths = context_clip_paths[issue_mask]
        if issue_paths.size <= 1:
            continue
        issue_paths = issue_paths[0:min(10, issue_paths.size)].squeeze()
        sub_context_labels = context_labels[issue_mask][0:issue_paths.size]
        for p in range(issue_paths.size):
            frame_path = issue_paths[p]
            user = frame_path.split('/')[-5]
            
            frame = Image.open(frame_path)
            frame = tv_F.to_tensor(frame)
            filename = "{}_{}_{}_{}.png".format(user, key, p, object_list[sub_context_labels[p]])
            
            # Save them out
            save_image(frame, output_dir + "/" + filename)


