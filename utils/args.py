# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
import argparse

FRAME_ANNOTATION_OPTIONS = ["object_not_present_issue", "framing_issue", "viewpoint_issue", "blur_issue", "occlusion_issue", "overexposed_issue", "underexposed_issue", "good", "medium", "bad"]
BOUNDING_BOX_OPTIONS = ["object_bounding_box"]

def parse_args(learner='default'):

    parser = argparse.ArgumentParser()
   
    # dataset parameters
    parser.add_argument("--checkpoint_dir", default='./checkpoint', help="Directory to save checkpoint to.")
    parser.add_argument("--data_path", required=True, help="Path to ORBIT root directory.")
    parser.add_argument("--test_set", default='test', choices=['validation', 'test'], 
                        help="Test on validation or test users.")
    parser.add_argument("--load_from_path", default=None, type=str,
                        help="Path to saved out tasks which will be used instead of sampling new tasks.")
    parser.add_argument("--filter_task_by_saved_mask", action="store_true",
                        help="If true and load_from_path is set, then use any specified drop filters that are saved out with the task being loaded")
    parser.add_argument("--generate_new_target_set", action="store_true",
                        help="If true and load_from_path is set, then instead of using the target set at load_from_path, a new target set will be generated with settings specified by args")
    parser.add_argument("--save_out_tasks", action="store_true",
                        help="If True, save out noisy detection tasks for transfer.")
    parser.add_argument("--experiment_name", default=None, type=str,
                        help="This will be used to name the checkpoint directory if specified; otherwise the checkpoint directory is the timestamp")


    # model parameters
    parser.add_argument("--model_path", "-m", default=None,
                        help="Path to model to load and test.")
    parser.add_argument("--mode", choices=["train", "test", "train_test"], default="train_test",
                        help="Whether to run training only, testing only, or both training and testing.")
    parser.add_argument("--feature_extractor", type=str, default="resnet18", choices=["resnet18", "efficientnetb0"],
                        help="Feature extractor backbone (default: resnet18).")
    parser.add_argument("--learn_extractor", action="store_true",
                        help="If True, learns all parameters of feature extractor at 0.1 of learning rate.")
    parser.add_argument("--pretrained_extractor_path", type=str, default=None, 
                        help="Path to pretrained feature extractor model (default: None).")
    parser.add_argument("--adapt_features", action="store_true",
                        help="If True, learns FiLM layers for feature adaptation.")
    parser.add_argument("--feature_adaptation_method", default="generate", choices=["generate", "learn"],
                        help="Generate FiLM layers with hyper-networks or add-in and learn FiLM layers directly (default: generate).")
    parser.add_argument("--classifier", default="linear", choices=["linear", "versa", "proto", "mahalanobis"],
                        help="Classifier head to use (default: linear).")
    parser.add_argument("--batch_normalisation", choices=["basic",  "task_norm"], default="basic", 
                        help="Normalisation layer to use (default: basic).")

    # task parameters
    parser.add_argument("--task_type", type=str, default="noisy_shots", choices=["noisy_shots", "test", "none"],
                        help="What LOO task we're doing (always in test mode)")
    parser.add_argument("--spread_constraint", choices=["by_class", "nonempty", "none"], default="none",
                        help="Spread coresets over classes (by_class) or by not allowing it to drop the last instance of a class (nonempty) or no constraint (none)")
    parser.add_argument("--drop_rate", type=float, default=0.1, help="Percentage of points to drop (as float in [0,1])")
    parser.add_argument("--importance_calculator", type=str, default="loo_loss", choices=["loo_loss", "random"],
                        help="What method we're using to calculate importance of a point")
    parser.add_argument("--fraction_of_target_for_loo", type=float, default=1.0,
                        help="What fraction of the target set can be used for LOO importance calculations")
    parser.add_argument("--random_drop_seed", type=int, default=12345, 
                        help="Random seed used when selecting points to drop at random (different from other seed) (default: 12345).")



    # data parameters
    parser.add_argument("--train_way_method", type=str, default="random", choices=["random", "max"],
                        help="Method to sample classes for a train task (default: random).")
    parser.add_argument("--test_way_method", type=str, default="max", choices=["random", "max"],
                        help="Method to sample classes for a test/validation task (default: max).")
    parser.add_argument("--train_object_cap", type=int, default=15,
                        help="Cap on objects sampled per train task (default: 15).")
    parser.add_argument("--train_context_shot_method", type=str, default="random", choices=["specific", "fixed", "random", "max"],
                        help="Method to sample context shots for a train task (default: random).")
    parser.add_argument("--train_target_shot_method", type=str, default="random", choices=["specific", "fixed", "random", "max"],
                        help="Method to sample target shots for a train task (default: random).")
    parser.add_argument("--test_context_shot_method", type=str, default="max", choices=["specific", "fixed", "random", "max"],
                        help="Method to sample context shots for a test/validation task (default: max).")
    parser.add_argument("--test_target_shot_method", type=str, default="max", choices=["specific", "fixed", "random", "max"],
                        help="Method to sample target shots for a test/validation task (default: max).")
    parser.add_argument("--context_shot", type=int, default=5,
                        help="If train/test_context_shot_method = specific/fixed, number of videos per object for context set (default: 5).")
    parser.add_argument("--target_shot", type=int, default=2,
                        help="If train/test_context_shot_method = specific/fixed, number of videos per object for target set (default: 2).")
    parser.add_argument("--with_train_shot_caps", action="store_true",
                        help="Caps videos per objects sampled in train tasks.")
    parser.add_argument("--context_video_type", type=str, default='clean', choices=['clean'],
                        help="Video type for context set (default: clean).")
    parser.add_argument("--target_video_type", type=str, default='clutter', choices=['clutter', 'clean'],
                        help="Video type for target set (default: clutter).")
    parser.add_argument("--subsample_factor", type=int, default=1,
                        help="Factor to subsample video by before sampling clips (default: 1).")
    parser.add_argument("--train_context_num_clips", type=str, default='random', choices=['random', 'max'],
                        help="Number of clips to sample per context video for a train task (default: random).")
    parser.add_argument("--train_target_num_clips", type=str, default='random', choices=['random', 'max'],
                        help="Number of clips to sample per target video for a train task (default: random).")
    parser.add_argument("--test_context_num_clips", type=str, default='random', choices=['random', 'max'],
                        help="Number of clips to sample per context video for a test/validation task (default: random).")
    parser.add_argument("--test_target_num_clips", type=str, default='max', choices=['max'],
                        help="Number of overlapping to sample per target video for a test/validation task (default: max).")
    parser.add_argument("--clip_length", type=int, default=8,
                        help="Number of frames to sample per clip (default: 8).")
    parser.add_argument("--no_preload_clips", action="store_true",
                        help="Do not preload clips per task from disk. Use if CPU memory is limited, but will mean slower training/testing.")
    parser.add_argument("--frame_size", type=int, default=224, choices=[84, 224],
                        help="Frame size (default: 224).")
    parser.add_argument("--annotations_to_load", nargs='+', type=str, default=[], choices=FRAME_ANNOTATION_OPTIONS+BOUNDING_BOX_OPTIONS,
                        help="Annotations to load per frame (default: None).")
    parser.add_argument("--train_task_type", type=str, default="user_centric", choices=["user_centric", "object_centric"],
                        help="Sample train tasks as user-centric or object-centric.")
    parser.add_argument("--train_tasks_per_user", type=int, default=50,
                        help="Number of train tasks per user per epoch (default: 50).")
    parser.add_argument("--test_tasks_per_user", type=int, default=5,
                        help="Number of test tasks per user (default: 5).")
    
    # training parameters
    parser.add_argument("--seed", type=int, default=1991, 
                        help="Random seed (default: 1991).")
    parser.add_argument("--epochs", "-e", type=int, default=10, 
                        help="Number of training epochs (default: 10).")
    parser.add_argument("--validation_on_epoch", type=int, default=5,
                        help="Epoch to turn on validation (default: 5).")
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.0001,
                        help="Learning rate (default: 0.0001).")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size when processing context and target set. Used for training/testing FineTuner, testing MAML, and training/testing CNAPs/ProtoNets with LITE (default: 256).")
    parser.add_argument("--tasks_per_batch", type=int, default=16,
                        help="Number of tasks between parameter optimization.")
    parser.add_argument("--with_lite", action="store_true",
                        help="If True, trains with LITE.")
    parser.add_argument("--num_lite_samples", type=int, default=8, 
                        help="Number of context clips per task to back-propagate with LITE training (default: 8)")
    parser.add_argument("--gpu", type=int, default=0, 
                        help="gpu id to use (default: 0, cpu: <0)")
    parser.add_argument("--use_two_gpus", dest="use_two_gpus", default=False, action="store_true",
                        help="If True, do model parallelism over 2 GPUs.")
    parser.add_argument("--print_by_step", action="store_true",
                        help="Print training by step (otherwise print by epoch).")

    # specific parameters
    if learner == 'gradient-learner':
        parser.add_argument("--num_grad_steps", type=int, required=True,
                        help="Number of inner loop (MAML, typically 15) or fine-tuning (FineTuner, typically 50) steps.")
        parser.add_argument("--inner_learning_rate", "--inner_lr", type=float, default=0.1,
                        help="Learning rate for inner loop (MAML) or fine-tuning (FineTuner) (default: 0.1).")
             

    args = parser.parse_args()
    args.preload_clips = not args.no_preload_clips
    verify_args(learner, args)
    return args

def verify_args(learner, args):
    # define some print out colours
    cred = "\33[31m"
    cyellow = "\33[33m"
    cend = "\33[0m"
    
    if args.test_tasks_per_user > 1:
        print('{:}warning: --test_tasks_per_user > 1 means multiple predictions are made per target frame. Only the last prediction is saved to JSON{:}'.format(cyellow, cend))
    
    if len(args.annotations_to_load) and args.no_preload_clips:
        sys.exit('{:}error: loading annotations with --annotations_to_load is currently not supported with --no_preload_clips{:}'.format(cred, cend))

    if 'train' in args.mode and not args.learn_extractor and not args.adapt_features:
        sys.exit('{:}error: at least one of "--learn_extractor" and "--adapt_features" must be used during training{:}'.format(cred, cend))

    if args.frame_size == 84:
        if 'resnet18' in args.feature_extractor:
            args.feature_extractor = "{:}_84".format(args.feature_extractor)
        else:
            sys.exit('{:}error: --frame_size 84 not implemented for {:}{:}'.format(cred, args.feature_extractor, cend))

    if args.filter_task_by_saved_mask:
        if args.load_from_path is None:
            sys.exit('{:}error: --filter_task_by_saved_mask cannot be applied if --load_from_path is not specified{:}'.format(cred, cend))

    if learner == 'gradient-learner':
        if args.with_lite:
            print('{:}warning: --with_lite is not relevant for MAML and FineTuner. Normal batching is used instead{:}'.format(cyellow, cend))
        
        if args.adapt_features and args.feature_adaptation_method == 'generate':
            sys.exit('{:}error: MAML/FineTuner are not generation-based methods; use --feature_adaptation_method learn{:}'.format(cred, cend))
