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
import matplotlib.pyplot as plt
import matplotlib

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

def handle_nan_annotations(annotations_dict):
    for key in annotations_dict.keys():
        not_nan_annotations = torch.nan_to_num(annotations_dict[key], nan=0.0)
        not_nan_annotations = not_nan_annotations.squeeze(dim=1).squeeze(dim=1)
        annotations_dict[key] = not_nan_annotations
    # Check that if we're using hand annotations, then all of the data must have a hand label
    if 'bad' in annotations_dict.keys():
        assert (annotations_dict["bad"] + annotations_dict["medium"] + annotations_dict["good"]).sum() == annotations_dict["bad"].shape[0]
    return annotations_dict

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
        # not_nan_annotations = torch.nan_to_num(annotations_dict[key], nan=0.0)
        # Preserve the batch dimension, in case there is only image with this noise type
        # not_nan_annotations = not_nan_annotations.squeeze(dim=1).squeeze(dim=1)
        # Filter out any annotations that aren't in our subselection:
        not_nan_annotations = annotations_dict[key][index_list]
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
        #not_nan_annotations = torch.nan_to_num(annotations_dict[key], nan=0.0)
        issue_mask = (annotations_dict[key] != 0) # True where issue exists
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


gizmo_dict = {'blur_issue': 'b', 'framing_issue':'f', 'object_not_present_issue':'n', 'occlusion_issue':'o', 'overexposed_issue': '+', 'underexposed_issue': '-', 'viewpoint_issue': 'v', 'bad': 'B', 'medium': 'M', 'good': 'G'}
def visualize_context_clips(context_clip_paths, annotations_dict, context_labels, object_list, dropped_indices, output_dir, task_num=None):
    # So we have a grid of images, where each row is for a different class
    # We can get the class label from the object list
    num_rows = len(object_list)
    # Then we show each instance of the class in its own little column; I'm not sure if we have equal representation, so this may be jagged
    # Let's read in a grid of images
    image_infos = []
    for r in range(num_rows):
        image_infos.append([])
    for p in range(context_clip_paths.size):
        frame_path = context_clip_paths[p][0]
        user = frame_path.split('/')[-5]
        frame = Image.open(frame_path)
        dropped = np.isin(p, dropped_indices)
        annot_string = ""
        #hand_annot_string = ""
        for issue in annotations_dict.keys():
            #if issue == "label":
            #    hand_annot_string += annotations_dict[issue][p]
            #    continue
            if annotations_dict[issue][p] != 0:
                if issue in gizmo_dict:
                    annot_string += gizmo_dict[issue]
        image_info = (frame, dropped, annot_string) # + hand_annot_string)
        image_infos[context_labels[p]].append(image_info)
    num_cols = -1

    for r in range(num_rows):
        if len(image_infos[r]) > num_cols:
            num_cols = len(image_infos[r])
    # Great, now we have our grid of image_infos. Let's put them on matplotlib subfigures
    f, axarr = plt.subplots(num_rows, num_cols, figsize=(num_cols+1,num_rows+1))
    f.tight_layout()
    plt.axis('off')
    for r in range(num_rows):
        axarr[r][0].set_ylabel(object_list[r], rotation='vertical', size='large')
        for c in range(num_cols):
            # No more instances of this class, go to next class/row
            if c >= len(image_infos[r]):
                axarr[r,c].axis('off')
                continue
            # Set image
            image, dropped, annot_string = image_infos[r][c]
            axarr[r,c].imshow(image, aspect='auto')

            # We want the borders of the dropped images to be a different color from the selected images (red vs green)
            axis_color = '#db321f' if dropped else '#1fdb4b'
            # Set boundary if image is in list of dropped indices(?)
            for child in axarr[r,c].get_children():
                if isinstance(child, matplotlib.spines.Spine):
                    child.set_color(axis_color)
            # We want to display a little "gizmo" string of annotations on each image
            axarr[r,c].text(0, 0, annot_string)
            axarr[r,c].get_xaxis().set_ticks([])
            axarr[r,c].get_yaxis().set_ticks([])
    
    plt.subplots_adjust(wspace=0.01, hspace=0.3)
    if task_num is None:
        plt.savefig(os.path.join(output_dir, "{}_context.png".format(user)), bbox_inches='tight')
    else:
        plt.savefig(os.path.join(output_dir, "{}_{}_context.png".format(user, task_num)), bbox_inches='tight')

    plt.close()

def get_user_id_from_clip_path(context_clip_path):
    return context_clip_path[0].split('/')[-5]
