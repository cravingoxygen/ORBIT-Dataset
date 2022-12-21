# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import torch
import numpy as np
#import json
import pickle

def print_and_log(log_file, message):
    print(message)
    log_file.write(message + '\n')

def get_log_files(checkpoint_dir, model_path, experiment_name=None):
    """
    Function that takes a path to a checkpoint directory and returns a reference to a logfile and paths to the
    fully trained model and the model with the best validation score.
    """
    if model_path:
        model_dir = os.path.split(model_path)[0]
        verify_checkpoint_dir(model_dir)
    # If no experiment name specified, use timestamp
    if experiment_name is None:
        checkpoint_dir = os.path.join(checkpoint_dir, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    else:
    # Otherwise, use the experiment name instead
        checkpoint_dir = os.path.join(checkpoint_dir, experiment_name)

    if os.path.exists(checkpoint_dir):
        print("Warning, checkpoint dir {} already exists".format(checkpoint_dir))
    else:
        os.makedirs(checkpoint_dir)

    checkpoint_path_best = os.path.join(checkpoint_dir, 'best.pt')
    checkpoint_path_final = os.path.join(checkpoint_dir, 'final.pt')
    logfile_path = os.path.join(checkpoint_dir, 'log.txt')
    if os.path.isfile(logfile_path):
        logfile = open(logfile_path, "a", buffering=1)
    else:
        logfile = open(logfile_path, "w", buffering=1)

    return checkpoint_dir, logfile, checkpoint_path_best, checkpoint_path_final

def verify_checkpoint_dir(checkpoint_dir):
    # verify that the checkpoint directory and file exists
    if not os.path.exists(checkpoint_dir):
        print("Can't resume/test for checkpoint. Checkpoint directory ({}) does not exist.".format(checkpoint_dir), flush=True)
        sys.exit()

def stats_to_str(stats):
    s=''
    for stat, scores in stats.items():
        if isinstance(scores, list):
            s+='{0:}: {1:.6f} ({2:.6f}) '.format(stat, scores[0]*100, scores[1]*100)
        else:
            s+='{0:}: {1:.6f} '.format(stat, scores*100)
    return s

def convert_to_numpy(x):
    if torch.is_tensor(x):
        if x.is_cuda:
            return x.cpu().numpy()
        else:
            return x.numpy()
    else:
        return x


def plot_hist(x, bins, filename, checkpoint_dir, user=None, task_num=None, title='', x_label='', y_label='', density=False):
    x = convert_to_numpy(x)
    plt.hist(x, bins=bins, density=density)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    fname = ""
    if user != None:
        fname += "{}_".format(user)
    if task_num != None:
        fname += "{}_".format(task_num)
    fname += filename + ".png"
    plt.savefig(os.path.join(checkpoint_dir, fname))
    plt.close()

def save_task(task_dict, keep_mask, task_num, seed, checkpoint_dir):
    output_dir = os.path.join(checkpoint_dir, "tasks_for_seed_{}".format(seed))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_path = os.path.join(output_dir, "task_{}.pickle".format(task_num))

    with open(file_path, "wb") as file:
        # Basically the whole task dict, but including the task_num and without the clips.
        trimmed_task_dict = {  "object_list": task_dict["object_list"],
                               "context_paths": task_dict["context_paths"],
                               "target_paths": task_dict["target_paths"],
                               "context_labels": task_dict["context_labels"],
                               "target_labels": task_dict["target_labels"],
                               "context_annotations": task_dict["context_annotations"],
                               "target_annotations": task_dict["target_annotations"],
                               "user_id": task_dict["user_id"],
                               "task": task_num,
                               "keep_mask": keep_mask,
                          }
        pickle.dump(trimmed_task_dict, file)


def save_image_paths(context_clip_paths, target_paths_by_video, seed, checkpoint_dir, task_num):
    output_dir = os.path.join(checkpoint_dir, "paths_for_seed_{}".format(seed))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_path = os.path.join(output_dir, "paths_task_{}.pickle".format(task_num))

    with open(file_path, "wb") as file:
        clip_path_dict = { "context_clip_paths": context_clip_paths,
                           "target_paths_by_video": target_paths_by_video,
                            "task": task_num,
                          }
        pickle.dump(clip_path_dict, file)


def dump_context_paths_to_file(context_clip_paths, seed, checkpoint_dir):
    file_path = os.path.join(checkpoint_dir, "clip_paths_seed_{}.txt".format(seed))
    with open(file_path, "a") as file:
        #for vid in target_paths_by_video:
        #    file.writelines(s + "\n" for s in vid)
        for path_arr in context_clip_paths:
            path_list = path_arr.tolist()
            file.writelines(s + "\n" for s in path_list)
