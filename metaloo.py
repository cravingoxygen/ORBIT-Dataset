import torch
import numpy as np
from utils.data import attach_frame_history
from utils.optim import cross_entropy


def calculate_loo(model, context_clips, context_labels, target_frames_by_video, target_paths_by_video, target_labels_by_video, object_list, ops_counter, clip_length):
    context_classes = context_labels.unique()
    num_videos = len(target_labels_by_video)
    loss_per_video = torch.zeros((len(context_labels), num_videos))
    with torch.no_grad():
        for i in range(context_clips.shape[0]):
            if i == 0:
                context_clips_loo = context_clips[i + 1:]
                context_labels_loo = context_labels[i + 1:]
            else:
                context_clips_loo = torch.cat((context_clips[0:i], context_clips[i + 1:]), 0)
                context_labels_loo = torch.cat((context_labels[0:i], context_labels[i + 1:]), 0)
                
            # Handle the case where we're dropping the only instance of a class by heavy penalty
            if len(context_labels_loo.unique()) < len(context_classes):
                for v in range(num_videos):
                    loss_per_video[i][v] = 1000
                continue
                
            # fit model to LOO context set
            model.personalise(context_clips_loo, context_labels_loo, ops_counter=ops_counter)
            
            # evaluate model (on all videos)
            # TODO: Maybe only evaluate on some? Or on some frames from each?
            for v, video_frames, video_paths, video_label in zip(range(num_videos), target_frames_by_video, target_paths_by_video, target_labels_by_video):
                video_clips = attach_frame_history(video_frames, clip_length)
                video_logits = model.predict(video_clips)
                # What to do with these logits
                loss_per_video[i][v] = calculate_loss(video_logits, video_label, video_paths, object_list)
            #print(f'Loss per video {i} : {loss_per_video[i]}')
    weights = loss_per_video.sum(dim=1)
    if len(weights) != context_clips.shape[0]:
        import pdb; pdb.set_trace()
        print("Problem with weights shape")

    #weights = torch.from_numpy(np.random.random(len(context_labels)))

    return weights

def calculate_loss(frame_logits, video_label, frame_paths, object_list):

    # remove any duplicate frames added due to padding to a multiple of clip_length
    frame_paths, unique_idxs = np.unique(frame_paths, return_index=True)
    frame_logits = frame_logits[unique_idxs]

    assert frame_paths.shape[0] == frame_logits.shape[0]
    # repeat this video's label for all frames in video
    video_labels = video_label.repeat_interleave(frame_logits.shape[0]).cuda()
    return cross_entropy(frame_logits, video_labels)
    #frame_predictions = frame_logits.argmax(dim=-1).detach().cpu().numpy()
    #video_label = video_label.clone().cpu().numpy()
 
 
def drop_worst(weights, drop_rate=None, num_to_drop=None, spread_constraint=None, class_labels=None):
    assert drop_rate != None or num_to_drop != None
    if drop_rate != None:
        num_to_keep = int(len(weights) * (1 - drop_rate))
    else:
        num_to_keep = max(0, len(weights) - num_to_drop)

    ranking = torch.argsort(weights, descending=True)

    if num_to_keep == 0:
        import pdb; pdb.set_trace()
        return [], ranking

    if spread_constraint == None or spread_constraint == "none":
        return ranking[0:num_to_keep], ranking[num_to_keep:]
        
    if spread_constraint == "nonempty":
        # Only dropping 1 point
        if num_to_keep != len(ranking)-1:
            print("nonempty spread constraint currenlty only support dropping 1 point at a time")
            return None
        drop_index = num_to_keep
        keep_mask = torch.ones(len(ranking), dtype=bool)
        keep_mask[drop_index] = False

        total_num_classes = len(class_labels.unique())
        num_represented_classes = len(class_labels[ranking[keep_mask]].unique())
        while num_represented_classes != total_num_classes:
            # We shouldn't reach the end of the loop as long as the way/shot setup makes sense
            keep_mask[drop_index] = True
            drop_index -= 1
            keep_mask[drop_index] = False

            num_represented_classes = len(class_labels[ranking[keep_mask]].unique())
        drop_mask = torch.logical_not(keep_mask)
        return ranking[keep_mask], ranking[drop_mask]
        
def select_top_k(number, weights, spread_constraint, class_labels=None):
    if spread_constraint == "by_class":
        classes = torch.unique(class_labels)
        candidate_indices = torch.zeros((len(classes), number), dtype=torch.long)
        for c in classes:
            c_indices = utils.extract_class_indices(class_labels, c)
            sub_weights = weights[c_indices]
            sub_ranking = torch.argsort(sub_weights, descending=True)
            class_candidate_indices = c_indices[sub_ranking[0:number]]
            candidate_indices[c] = class_candidate_indices
        return candidate_indices.flatten()
    elif spread_constraint == "none":
        ranking = torch.argsort(weights, descending=True)
        return ranking[0:number], ranking[number:]
    elif spread_constraint == "nonempty":
        print("By class spread constraint not yet supported when selecting by dropping x")
        return None
