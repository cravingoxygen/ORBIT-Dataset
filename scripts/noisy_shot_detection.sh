#!/bin/bash
CUDA_VISIBLE_DEVICES=2

python3 single-step-learner.py --data_path dataset/orbit_benchmark_84 --frame_size 84 \
                              --model_path checkpoints/orbit_cleve_protonets_resnet18_84.pth \
                              --feature_extractor resnet18 \
                              --pretrained_extractor_path features/pretrained/resnet18_imagenet_84.pth \
                              --classifier proto \
                              --context_video_type clean \
                              --target_video_type clean \
                              --train_object_cap 10 \
                              --with_train_shot_caps \
                              --mode test \
                              --batch_normalisation basic \
                              --clip_length 1 \
                              --batch_size 4 \
                              --drop_rate 0.5 \
                              --importance_calculator loo_loss \
                              --test_set validation \
                              --test_tasks_per_user 1 \
                              --annotations_to_load "object_not_present_issue" "framing_issue" "viewpoint_issue" "blur_issue" "occlusion_issue" "overexposed_issue" "underexposed_issue"

#                              --test_context_num_clips max \

# For real experiment:
# Remove context_num_clips_max
# Remove test_set validation
# Make sure the test_tasks_per_user is more than 2 (probably 5?)
# Make sure we're loading full list of annotations:
#  --annotations_to_load "object_not_present_issue" "framing_issue" "viewpoint_issue" "blur_issue" "occlusion_issue" "overexposed_issue" "underexposed_issue"
# Make sure we've removed the random weights from metaloo!
