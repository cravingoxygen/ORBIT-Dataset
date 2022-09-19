import numpy as np
import torch

class NoiseTracker():
    def __init__(self, annotations_to_load):
        self.current_user = 0
        self.issue_types = annotations_to_load
        # For each user, for each task, the annotations for each frame
        # Start with one element in the list, since we only call next_user after the first user
        self.all_context_frame_annotations = [[]]
        # What we ultimately want to know is how many type-X-error frames did we find out of the total
        # We can keep this split by user
        self.issues_per_user = [[]]
        self.noisy_discarded_per_user = [[]]
        self.total_frames_per_user = [[]]

    def get_confidence_interval(self, scores):
        return (1.96 * np.std(scores)) / np.sqrt(len(scores))

    def next_user(self):
        self.current_user += 1
        self.all_context_frame_annotations.append([])
        self.issues_per_user.append([])
        self.noisy_discarded_per_user.append([])
        self.total_frames_per_user.append([])

    # Can track with int or list
    def make_empty_issues_dict(self, count_type="int"):
        new_dict = {}
        for key in self.issue_types:
            if count_type == "int":
                new_dict[key] = 0
            elif count_type == "list":
                new_dict[key] = []
            else:
                return None
        return new_dict

    def append_video(self, video_annotations, drop_mask):
        # Clean up the annotations, which may have NaNs
        video_issues = {}
        for key in video_annotations.keys():
            video_issues[key] = torch.nan_to_num(video_annotations[key], nan=0.0).squeeze()
        self.all_context_frame_annotations[self.current_user].append(video_issues)
        self.issues_per_user[self.current_user].append(self.make_empty_issues_dict())
        self.noisy_discarded_per_user[self.current_user].append(self.make_empty_issues_dict())
        video_num = len(self.all_context_frame_annotations[self.current_user])-1

        for key in video_issues.keys():
            self.issues_per_user[self.current_user][video_num][key] = (video_issues[key]).sum().item() #TODO: do squeeze already
            self.noisy_discarded_per_user[self.current_user][video_num][key] = (video_issues[key])[drop_mask].sum().item()

        self.total_frames_per_user[self.current_user].append(len(drop_mask))


    def get_mean_stats(self):
        # Questions we want to answer with these stats:
        # For each user, how many of each annotation type did we get correct?
        # For each user, how much of each video consisted of the annotation type? i.e. had that kind of noise?
        # On average for each video, how many issues of each type did we get correct?
        # On average for each video, how much of of the video consisted of the annotation type? i.e. had that kind of noise?
        # To match how eval_metrics does it, we're going to average per user, per video 
        # i.e. calculate the stat for each individual video, and then average as we like
        print("Range over users when printing: {}".format(self.current_user+1))
        frac_noisy_all_users = self.make_empty_issues_dict(count_type="list")
        frac_detected_all_users = self.make_empty_issues_dict(count_type="list")
        user_stats = []
        for user in range(self.current_user+1):
            frac_noisy_for_user = self.make_empty_issues_dict(count_type="list")
            frac_detected_for_user = self.make_empty_issues_dict(count_type="list")
            for video in range(len(self.all_context_frame_annotations[user])):
                for annot in self.issue_types:
                    # Fraction of frames with this noise type
                    frac_noisy = self.issues_per_user[user][video][annot]/self.total_frames_per_user[user][video]
                    # Fraction of frames with this noise type that were chosen to be discarded
                    if self.issues_per_user[user][video][annot] == 0:
                        frac_detected = 0
                    else:
                        frac_detected = self.noisy_discarded_per_user[user][video][annot]/self.issues_per_user[user][video][annot]
                    frac_noisy_for_user[annot].append(frac_noisy)
                    frac_detected_for_user[annot].append(frac_detected)

            # We can now average frac_noisy and frac_detecteed over all videos for this user
            user_summary = {}
            for annot in self.issue_types:
                user_summary[annot] = {'frac_noisy': (np.mean(frac_noisy_for_user[annot]), self.get_confidence_interval(frac_noisy_for_user[annot])),
                                   'frac_detected': (np.mean(frac_detected_for_user[annot]), self.get_confidence_interval(frac_detected_for_user[annot]))}
                frac_noisy_all_users[annot].extend(frac_noisy_for_user[annot])
                frac_detected_all_users[annot].extend(frac_detected_for_user[annot])
            user_stats.append(user_summary)

        import pdb; pdb.set_trace

        # We can now average frac_noisy and frac_detected over all users
        summary_stats = {}
        for annot in self.issue_types:
            summary_stats[annot] = { 'frac_noisy_all_videos': (np.mean(frac_noisy_all_users[annot]), self.get_confidence_interval(frac_noisy_all_users[annot])),
                                    'frac_detected_all_videos': (np.mean(frac_detected_all_users[annot]), self.get_confidence_interval(frac_detected_all_users[annot]))}
        summary_stats['user_stats'] = user_stats

        return summary_stats
