# Now, we can create an instance of the dataset. This creates a queue of tasks from the dataset that can be divided between multiple workers.

# Add this repository root to the Python path.
from pathlib import Path
from data.queues import UserEpisodicDatasetQueue

DATA_ROOT = "dataset/orbit_benchmark_84/" #"orbit_benchmark"
DATA_SPLIT = "validation"

print("Creating data queue.")
data_queue = UserEpisodicDatasetQueue(
    root=Path(DATA_ROOT, DATA_SPLIT), # path to data
    way_method="max", # sample all objects per user
    object_cap="max", # do not cap number of objects per user
    shot_method=["max", "max"], # sample [all context videos, all target videos] per object
    shots=[5, 2], # only relevant if shot_method contains strings "specific" or "fixed"
    video_types=["clean", "clutter"], # sample clips from [clean context videos, clutter target videos]
    subsample_factor=1, # subsample rate for video frames
    num_clips=["random", "max"], # sample [a random number of clips per context video, all target clips per target video]; note if test_mode=True, target clips will be flattened into a list of frames
    clip_length=8, # sample 8 frames per clip
    preload_clips=True, # load clips into memory when sampling a task; if False, load each clip only when it is passed through model
    frame_size=84, # width and height of frame
    annotations_to_load=[], # do not load any frame annotations
    tasks_per_user=1, # sample 1 task per user; if >1 then only frame predictions from the final task per user will be saved
    test_mode=True, # sample test (rather than train) tasks
    with_cluster_labels=False, # use user's personalised object names as labels, rather than broader object categories
    with_caps=False, # do not impose any caps
    shuffle=False) # do not shuffle task data

print(f"Created data queue, queue uses {data_queue.num_workers} workers.")

'''
We now need to set up the model. For the starter task, we will use a few-shot learning model called Prototypical Networks (Snell et al., 2017)
which has been pretrained on the ORBIT train users for the CLUVE or Clutter Video Evaluation task (trained on 84x84 frame size, using LITE).
First, we download the checkpoint file that corresponds to this model.
We then create an instance of the model using the pretrained weights.
'''

checkpoint_path = Path("checkpoints/orbit_cleve_protonets_resnet18_84.pth")
if not checkpoint_path.exists():
    #checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    #print("downloading checkpoint file")
    #!wget -q -O orbit_pretrained_checkpoints/orbit_cleve_protonets_efficientnetb0_84_lite.pth https://github.com/microsoft/ORBIT-Dataset/raw/master/checkpoints/orbit_cleve_protonets_efficientnetb0_84_lite.pth
    #print(f"checkpoint saved to {checkpoint_path}!")
    print(f"failed to find checkpoint {checkpoints/orbit_cleve_protonets_resnet18_84.pth}")
    quit()
else:
    print(f"checkpoint already exists at {checkpoint_path}")

import torch
from models.few_shot_recognisers import SingleStepFewShotRecogniser

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    map_location = lambda storage, _: storage.cuda()
else:
    device = torch.device("cpu")
    map_location = lambda storage, _: storage.cpu()
print(f"Using device {device}")

model = SingleStepFewShotRecogniser(
    pretrained_extractor_path="features/pretrained/resnet18_imagenet_84.pth", # path to pretrained feature extractor trained on ImageNet
    feature_extractor="resnet18", # feature extractor is an EfficientNet-B0
    batch_normalisation="basic", # standard batch normalisation rather than task normalisation (Bronskill et al., 2020)
    adapt_features=False, # do not use FiLM Layers
    classifier="proto", # use a Prototypical Networks classifier head
    clip_length=8, # number of frames per clip; frame features are mean-pooled to get the clip feature
    batch_size=4, # number of clips within a task to process at a time
    learn_extractor=False, # only relevant when training
    feature_adaptation_method="generate", # only relevant when adapt_features = True
    use_two_gpus=False, # use only 1 GPU; if >1 model is parallelised over 2 GPUs
    num_lite_samples=8 # only relevant when training with LITE
)
model._set_device(device)
model._send_to_device()
model._register_extra_parameters()

checkpoint_path = Path("checkpoints", "orbit_cleve_protonets_resnet18_84.pth")
model.load_state_dict(torch.load(checkpoint_path, map_location=map_location), strict=False)
model.set_test_mode(True)
print("instance of SingleStepFewShotRecogniser created!")


'''
We are now going to run our data through our model.
We go through each task (which corresponds to a user from the dataset, since we specified tasks_per_user = 1 above)
and use the task's context clips to create a personalized model.
We then evaluate the personalized model on each frame in the user's target videos.

The results from each user will be saved to a JSON file (this is what should be submitted to the evaluation server)
and the aggregate stats will be printed to the console.
'''

from typing import Dict, Tuple
from utils.data import attach_frame_history
from utils.eval_metrics import TestEvaluator

output_dir = Path("output", DATA_SPLIT)
output_dir.mkdir(exist_ok=True, parents=True)

metrics = ['frame_acc', 'video_acc', 'frames_to_recognition']
evaluator = TestEvaluator(metrics, output_dir)
num_test_tasks = data_queue.num_users * data_queue.tasks_per_user

def get_stats_str(label: str, stats: Dict[str, Tuple[float, float]], dps: int=4) -> str:
    stats_str = f"{label}\t"
    stats_str += "\t".join([f"{stats[metric][0]:.{dps}f} ({stats[metric][1]:.{dps}f})" for metric in metrics])
    return stats_str

print("running evaluation")
print("         \tFrame Accuracy\tVideo Accuracy\tFrames to Recognition")
for step, task in enumerate(data_queue.get_tasks()):
    with torch.no_grad():
        context_set = task["context_clips"].to(device)          # Torch tensor of shape: (N, clip_length, C, H, W), dtype float32
        context_labels = task["context_labels"].to(device)      # Torch tensor of shape: (N), dtype int64
        object_list = task["object_list"]                       # List of str of length num_objects

        # personalise the pre-trained model to the current user
        model.personalise(context_set, context_labels)

        # loop through each of the user's target videos, and get predictions from the personalised model for every frame
        for video_frames, video_paths, video_label in zip(task['target_clips'], task["target_paths"], task['target_labels']):
            # video_frames is a Torch tensor of shape (frame_count, C, H, W), dtype float32
            # video_paths is a Torch tensor of shape (frame_count), dtype object (Path)
            # video_label is single int64

            # first, for each frame, attach a short history of its previous frames
            video_frames_with_history = attach_frame_history(video_frames, model.clip_length)      # Torch tensor of shape: (frame_count, clip_length, C, H, W), dtype float32
            # get predicted logits for each frame
            logits = model.predict(video_frames_with_history)                                      # Torch tensor of shape: (frame_count, num_objects), dtype float32
            evaluator.append_video(logits, video_label, video_paths, object_list)

        # reset model for next task
        model._reset()

        # check if the user has any more tasks; if tasks_per_user == 1, we reset every time.
        if (step+1) % data_queue.tasks_per_user == 0:
            _, current_user_stats = evaluator.get_mean_stats(current_user=True)
            print(get_stats_str(f"user {task['user_id']} ({evaluator.current_user+1}/{data_queue.num_users})", current_user_stats))
            if (step+1) < num_test_tasks:
                evaluator.next_user()

# Compute the aggregate statistics averaged over users and averged over videos. We use the video aggregate stats for the competition.
stats_per_user, stats_per_video = evaluator.get_mean_stats()
print(get_stats_str("User avg", stats_per_user))
print(get_stats_str("Video avg", stats_per_video))
evaluator.save()
print(f"results saved to {evaluator.json_results_path}")


