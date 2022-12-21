"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license

This code was based on the file run_cnaps.py (https://github.com/cambridge-mlg/cnaps/blob/master/src/run_cnaps.py)
from the cambridge-mlg/cnaps library (https://github.com/cambridge-mlg/cnaps).

The original license is included below:

Copyright (c) 2019 John Bronskill, Jonathan Gordon, and James Requeima.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE. 
"""

import os
import time
import torch
import random
import numpy as np
import torch.backends.cudnn as cudnn

from data.dataloaders import DataLoader
from models.few_shot_recognisers import SingleStepFewShotRecogniser
from utils.args import parse_args
from utils.ops_counter import OpsCounter
from utils.optim import cross_entropy, init_optimizer
from utils.data import get_clip_loader, unpack_task, attach_frame_history, save_selected_frames, visualize_context_clips, visualize_target_clips
from utils.data import handle_nan_annotations, get_user_id_from_clip_path
from utils.logging import print_and_log, get_log_files, stats_to_str, plot_hist, save_image_paths, save_task
from utils.eval_metrics import TrainEvaluator, ValidationEvaluator, TestEvaluator
from utils.noise_tracker import NoiseTracker

import metaloo

torch.multiprocessing.set_sharing_strategy('file_system')

def main():
 
    learner = Learner()
    learner.run()

class Learner:
    def __init__(self):
        self.args = parse_args()

        self.checkpoint_dir, self.logfile, self.checkpoint_path_validation, self.checkpoint_path_final \
            = get_log_files(self.args.checkpoint_dir, self.args.model_path, experiment_name=self.args.experiment_name)

        print_and_log(self.logfile, "Options: %s\n" % self.args)
        print_and_log(self.logfile, "Checkpoint Directory: %s\n" % self.checkpoint_dir) 

        random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        device_id='cpu'
        self.map_location='cpu'
        if torch.cuda.is_available() and self.args.gpu>=0:
            cudnn.enabled = True
            cudnn.benchmark = False
            cudnn.deterministic = True
            device_id = 'cuda:' + str(self.args.gpu)
            torch.cuda.manual_seed_all(self.args.seed)
            self.map_location=lambda storage, loc: storage.cuda()
        
        self.device = torch.device(device_id)
        self.ops_counter = OpsCounter()
        self.init_dataset() 
        self.init_model()
        self.init_evaluators()
        self.loss = cross_entropy
        self.train_task_fn = self.train_task_with_lite if self.args.with_lite else self.train_task
    
    def init_dataset(self):
        
        dataset_info = {
            'mode': self.args.mode,
            'data_path': self.args.data_path,
            'train_object_cap': self.args.train_object_cap,
            'with_train_shot_caps': self.args.with_train_shot_caps,
            'with_cluster_labels': False,
            'train_way_method' : self.args.train_way_method,
            'test_way_method' : self.args.test_way_method,
            'train_shot_methods' : [self.args.train_context_shot_method, self.args.train_target_shot_method],
            'test_shot_methods' : [self.args.test_context_shot_method, self.args.test_target_shot_method],
            'train_tasks_per_user': self.args.train_tasks_per_user,
            'test_tasks_per_user': self.args.test_tasks_per_user,
            'train_task_type' : self.args.train_task_type,
            'test_set': self.args.test_set,
            'shots' : [self.args.context_shot, self.args.target_shot],
            'video_types' : [self.args.context_video_type, self.args.target_video_type],
            'clip_length': self.args.clip_length,
            'train_num_clips': [self.args.train_context_num_clips, self.args.train_target_num_clips],
            'test_num_clips': [self.args.test_context_num_clips, self.args.test_target_num_clips],
            'subsample_factor': self.args.subsample_factor,
            'frame_size': self.args.frame_size,
            'annotations_to_load': self.args.annotations_to_load,
            'preload_clips': self.args.preload_clips,
            'load_from_path': self.args.load_from_path,
            'filter_task_by_saved_mask': self.args.filter_task_by_saved_mask,
            'generate_new_target_set': self.args.generate_new_target_set,
        }
        
        dataloader = DataLoader(dataset_info)
        self.train_queue = dataloader.get_train_queue()
        self.validation_queue = dataloader.get_validation_queue()
        self.test_queue = dataloader.get_test_queue()
        
    def init_model(self):
        self.model = SingleStepFewShotRecogniser(
                        self.args.pretrained_extractor_path, self.args.feature_extractor, self.args.batch_normalisation,
                        self.args.adapt_features, self.args.classifier, self.args.clip_length, self.args.batch_size,
                        self.args.learn_extractor, self.args.feature_adaptation_method, self.args.use_two_gpus, self.args.num_lite_samples
                    )
        self.model._register_extra_parameters()
        self.model._set_device(self.device)
        self.model._send_to_device()
        
    def init_evaluators(self):
        self.train_metrics = ['frame_acc']
        self.evaluation_metrics = ['frame_acc', 'frames_to_recognition', 'video_acc'] 
        self.train_evaluator = TrainEvaluator(self.train_metrics)
        self.validation_evaluator = ValidationEvaluator(self.evaluation_metrics)
        self.test_evaluator = TestEvaluator(self.evaluation_metrics, self.checkpoint_dir)
    
    def run(self):
        if self.args.mode == 'train' or self.args.mode == 'train_test':
            
            extractor_scale_factor=0.1 if self.args.pretrained_extractor_path else 1.0
            self.optimizer = init_optimizer(self.model, self.args.learning_rate, extractor_scale_factor=extractor_scale_factor)
            
            for epoch in range(self.args.epochs):
                losses = []
                since = time.time()
                torch.set_grad_enabled(True)
                self.model.set_test_mode(False)
                
                train_tasks = self.train_queue.get_tasks()
                total_steps = len(train_tasks)
                for step, task_dict in enumerate(train_tasks):

                    t1 = time.time()
                    task_loss = self.train_task_fn(task_dict)
                    task_time = time.time() - t1
                    losses.append(task_loss.detach())
                    
                    if self.args.print_by_step:
                        current_stats_str = stats_to_str(self.train_evaluator.get_current_stats())
                        print_and_log(self.logfile, f'epoch [{epoch+1}/{self.args.epochs}][{step+1}/{total_steps}], train loss: {task_loss.item():.7f}, {current_stats_str.strip()}, time/task: {int(task_time/60):d}m{int(task_time%60):02d}s')

                    if ((step + 1) % self.args.tasks_per_batch == 0) or (step == (total_steps - 1)):
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                
                mean_stats = self.train_evaluator.get_mean_stats()
                mean_epoch_loss = torch.Tensor(losses).mean().item()
                seconds = time.time() - since
                # print
                print_and_log(self.logfile, '-'*150)
                print_and_log(self.logfile, f'epoch [{epoch+1}/{self.args.epochs}] train loss: {mean_epoch_loss:.7f} {stats_to_str(mean_stats)} time/epoch: {int(seconds/60):d}m{int(seconds%60):02d}s')
                print_and_log(self.logfile, '-'*150)
                self.train_evaluator.reset()
                self.save_checkpoint(epoch + 1)

                # validate
                if (epoch + 1) >= self.args.validation_on_epoch:
                    self.validate()
            
            # save the final model
            torch.save(self.model.state_dict(), self.checkpoint_path_final)

        if self.args.mode == 'train_test':
            self.test(self.checkpoint_path_final)
            self.test(self.checkpoint_path_validation)

        if self.args.mode == 'test':
            if self.args.task_type == 'noisy_shots':
                self.detect_noisy_shots(self.args.model_path)
            else:
                self.test(self.args.model_path)

        self.logfile.close()

    def train_task(self, task_dict):
        context_clips, context_paths, context_labels, target_clips, target_paths, target_labels, object_list = unpack_task(task_dict, self.device, target_to_device=True, preload_clips=self.args.preload_clips)

        self.model.personalise(context_clips, context_labels)
        target_logits = self.model.predict(target_clips)
        self.train_evaluator.update_stats(target_logits, target_labels)
        
        task_loss = self.loss(target_logits, target_labels) / self.args.tasks_per_batch
        task_loss += 0.001 * self.model.feature_adapter.regularization_term(switch_device=self.args.use_two_gpus) 
        task_loss.backward(retain_graph=False)        
       
        # reset task's params
        self.model._reset()

        return task_loss

    def train_task_with_lite(self, task_dict):
        context_clips, context_paths, context_labels, target_clips, target_paths, target_labels, object_list = unpack_task(task_dict, self.device, preload_clips=self.args.preload_clips)

        # compute and save personalise outputs of whole context set with back-propagation disabled
        self.model._cache_context_outputs(context_clips)

        task_loss = 0
        target_logits = []
        target_clip_loader = get_clip_loader((target_clips, target_labels), self.args.batch_size, with_labels=True)
        for batch_target_clips, batch_target_labels in target_clip_loader:
            self.model.personalise_with_lite(context_clips, context_labels)
            batch_target_clips = batch_target_clips.to(device=self.device)
            batch_target_labels = batch_target_labels.to(device=self.device)
            batch_target_logits = self.model.predict_a_batch(batch_target_clips)  
            target_logits.extend(batch_target_logits.detach())
           
            loss_scaling = len(context_labels) / (self.args.num_lite_samples * self.args.tasks_per_batch)
            batch_loss = loss_scaling * self.loss(batch_target_logits, batch_target_labels)
            batch_loss += 0.001 * self.model.feature_adapter.regularization_term(switch_device=self.args.use_two_gpus) 
            batch_loss.backward(retain_graph=False)
            task_loss += batch_loss.detach()
            
            # reset task's params
            self.model._reset()

        target_logits = torch.stack(target_logits)
        self.train_evaluator.update_stats(target_logits, target_labels)
        return task_loss
    
    def validate(self):
        
        self.model.set_test_mode(True) 
        with torch.no_grad():
            # loop through validation tasks (num_validation_users * num_test_tasks_per_user)
            num_val_tasks = self.validation_queue.num_users * self.args.test_tasks_per_user
            for step, task_dict in enumerate(self.validation_queue.get_tasks()):
                context_clips, context_clip_paths, context_labels, target_frames_by_video, target_paths_by_video, target_labels_by_video, object_list = unpack_task(task_dict, self.device, preload_clips=self.args.preload_clips)

                # if this is a user's first task, cache their target videos (as they remain constant for all their tasks - ie. num_test_tasks_per_user)
                if step % self.args.test_tasks_per_user == 0:
                    cached_target_frames_by_video, cached_target_paths_by_video, cached_target_labels_by_video = target_frames_by_video, target_paths_by_video, target_labels_by_video

                self.model.personalise(context_clips, context_labels)

                # loop through cached target videos for the current task
                for video_frames, video_paths, video_label in zip(cached_target_frames_by_video, cached_target_paths_by_video, cached_target_labels_by_video):
                    video_clips = attach_frame_history(video_frames, self.args.clip_length)
                    video_logits = self.model.predict(video_clips)
                    self.validation_evaluator.append_video(video_logits, video_label, video_paths, object_list)

                # reset task's params
                self.model._reset()

                # if this is the user's last task, get the average performance for the user
                if (step+1) % self.args.test_tasks_per_user == 0:
                    _, current_user_stats = self.validation_evaluator.get_mean_stats(current_user=True)
                    print_and_log(self.logfile, f'validation user {task_dict["user_id"]} ({self.validation_evaluator.current_user+1}/{self.validation_queue.num_users}) stats: {stats_to_str(current_user_stats)}')
                    if (step+1) < num_val_tasks:
                        self.validation_evaluator.next_user()
                    
            stats_per_user, stats_per_video = self.validation_evaluator.get_mean_stats()
            stats_per_user_str, stats_per_video_str = stats_to_str(stats_per_user), stats_to_str(stats_per_video)

            print_and_log(self.logfile, f'validation\n per-user stats: {stats_per_user_str}\n per-video stats: {stats_per_video_str}\n')
            # save the model if validation is the best so far
            if self.validation_evaluator.is_better(stats_per_video):
                self.validation_evaluator.replace(stats_per_video)
                torch.save(self.model.state_dict(), self.checkpoint_path_validation)
                print_and_log(self.logfile, 'best validation model was updated.\n')
            
            self.validation_evaluator.reset()

    def test(self, path):

        self.init_model()
        self.model.load_state_dict(torch.load(path, map_location=self.map_location)) 
        self.model.set_test_mode(True)
        self.ops_counter.set_base_params(self.model)

        with torch.no_grad():
            # loop through test tasks (num_test_users * num_test_tasks_per_user)
            num_test_tasks = self.test_queue.num_users * self.args.test_tasks_per_user

            remove_target_frames_without_object = True 
            if self.args.load_from_path is not None and not self.args.generate_new_target_set:
                remove_target_frames_without_object = False # Do exactly the saved out task

            for step, task_dict in enumerate(self.test_queue.get_tasks()):
                context_clips, context_clip_paths, context_labels, target_frames_by_video, target_paths_by_video, target_labels_by_video, object_list = unpack_task(task_dict, self.device, \
                                                                                                                preload_clips=self.args.preload_clips, remove_target_frames_without_object=remove_target_frames_without_object)
                # Something went wrong if these don't match
                assert task_dict["task"] == step
                # if this is a user's first task, cache their target videos (as they remain constant for all their tasks - ie. num_test_tasks_per_user)
                if step % self.args.test_tasks_per_user == 0:
                    cached_target_frames_by_video, cached_target_paths_by_video, cached_target_labels_by_video = target_frames_by_video, target_paths_by_video, target_labels_by_video

                # dummy warm-up to get correct timing
                self.model.personalise(context_clips, context_labels, ops_counter=False)
                torch.cuda.synchronize()
                self.model.personalise(context_clips, context_labels, ops_counter=self.ops_counter)

                # loop through cached target videos for the current task
                for video_frames, video_paths, video_label in zip(cached_target_frames_by_video, cached_target_paths_by_video, cached_target_labels_by_video):
                    video_clips = attach_frame_history(video_frames, self.args.clip_length)
                    video_logits = self.model.predict(video_clips)
                    self.test_evaluator.append_video(video_logits, video_label, video_paths, object_list)

                # reset task's params
                self.model._reset()
                # add task's ops to self.ops_counter
                self.ops_counter.task_complete()

                # if this is the user's last task, get the average performance for the user
                if (step+1) % self.args.test_tasks_per_user == 0:
                    _, current_user_stats = self.test_evaluator.get_mean_stats(current_user=True)
                    print_and_log(self.logfile, f'{self.args.test_set} user {task_dict["user_id"]} ({self.test_evaluator.current_user+1}/{self.test_queue.num_users}) stats: {stats_to_str(current_user_stats)}')
                    if (step+1) < num_test_tasks:
                        self.test_evaluator.next_user()
                    
            self.save_and_reset_stats(self.test_evaluator, path, descriptor="full")

    def detect_noisy_shots(self, path):
        
        self.init_model()
        self.model.load_state_dict(torch.load(path, map_location=self.map_location))
        self.model.set_test_mode(True)
        self.ops_counter.set_base_params(self.model)
        reduced_evaluator = TestEvaluator(self.evaluation_metrics, self.checkpoint_dir, save_file="reduced_results.json")
        noise_tracker = NoiseTracker(self.args.annotations_to_load)

        # Make random its own generator to use
        seed = self.args.random_drop_seed
        if self.args.importance_calculator == 'random':
            rng = np.random.default_rng(seed)
        else:
            rng = None

        with torch.no_grad():
            # loop through test tasks (num_test_users * num_test_tasks_per_user)
            num_test_tasks = self.test_queue.num_users * self.args.test_tasks_per_user
            # We want to remove bad target frames if we aren't sampling the target set
            # We sample the target set if we aren't loading from path or if we are loading from path, but generate_new_target_set = True
            remove_target_frames_without_object = True
            if self.args.load_from_path is not None and not self.args.generate_new_target_set:
                remove_target_frames_without_object = False
            # How are user tasks defined? Surely they must range over multiple classes/videos or else evaluating on the whole test set doesn't make sense
            for step, task_dict in enumerate(self.test_queue.get_tasks()):
                context_clips, context_clip_paths, context_labels, target_frames_by_video, target_paths_by_video, target_labels_by_video, object_list = unpack_task(task_dict, self.device, 
                                                                                                                                                        preload_clips=self.args.preload_clips, 
                                                                                                                                                        remove_target_frames_without_object=remove_target_frames_without_object )
                #save_image_paths(context_clip_paths, target_paths_by_video, seed, self.checkpoint_dir, step)
                #continue
                user = get_user_id_from_clip_path(context_clip_paths[0])
                annotations_dict = handle_nan_annotations(task_dict['context_annotations'])

                plot_hist(context_labels, list(range(len(object_list)+1)), "context_distrib", self.checkpoint_dir, user=user,
                         x_label='Classes', y_label='Count', title="Context Class Distribution") #task_num=step%self.args.test_tasks_per_user
                # if this is a user's first task, cache their target videos (as they remain constant for all their tasks - ie. num_test_tasks_per_user)
                if step % self.args.test_tasks_per_user == 0:
                    cached_target_frames_by_video, cached_target_paths_by_video, cached_target_labels_by_video = target_frames_by_video, target_paths_by_video, target_labels_by_video
                del target_frames_by_video, target_paths_by_video, target_labels_by_video

                # dummy warm-up to get correct timing
                self.model.personalise(context_clips, context_labels, ops_counter=False)
                torch.cuda.synchronize()
                self.model.personalise(context_clips, context_labels, ops_counter=self.ops_counter)

                # loop through cached target videos for the current task
                for video_frames, video_paths, video_label in zip(cached_target_frames_by_video, cached_target_paths_by_video, cached_target_labels_by_video):
                    video_clips = attach_frame_history(video_frames, self.args.clip_length)
                    video_logits = self.model.predict(video_clips)
                    self.test_evaluator.append_video(video_logits, video_label, video_paths, object_list)

                if self.args.importance_calculator == 'random':
                    weights = rng.random(len(context_clips))
                    weights = torch.Tensor(weights)
                elif self.args.importance_calculator == 'loo_loss':
                    # calculate loo weights
                    weights = metaloo.calculate_loo(self.model, context_clips, context_labels, cached_target_frames_by_video, cached_target_paths_by_video, cached_target_labels_by_video,
                                                         object_list, self.ops_counter, self.args.clip_length)
                # use these weights to do something
                keep_indices, drop_indices = metaloo.drop_worst(weights, drop_rate=self.args.drop_rate, spread_constraint=self.args.spread_constraint, class_labels=context_labels)
                drop_mask = np.zeros(len(context_clips), dtype=bool)
                drop_mask[drop_indices] = True
                noise_tracker.append_video(annotations_dict, drop_mask)

                if self.args.save_out_tasks:
                    save_task(task_dict, ~drop_mask, step, seed, self.checkpoint_dir)


                reduced_context_clips, reduced_context_labels = context_clips[keep_indices], context_labels[keep_indices]
                self.model.personalise(reduced_context_clips, reduced_context_labels, ops_counter=self.ops_counter)
                
                # loop through cached target videos for the current task, now using pared down context set base on metaloo outcome
                for video_frames, video_paths, video_label in zip(cached_target_frames_by_video, cached_target_paths_by_video, cached_target_labels_by_video):
                    video_clips = attach_frame_history(video_frames, self.args.clip_length)
                    video_logits = self.model.predict(video_clips)
                    reduced_evaluator.append_video(video_logits, video_label, video_paths, object_list)

                # Save out selected (dropped) frames
                # Save out sample of noisy frames, ideally of most noisy frames (what if there are many?)
                #save_selected_frames(context_clip_paths, task_dict['context_annotations'], context_labels, object_list, self.checkpoint_dir, keep_indices, "keep")
                #save_selected_frames(context_clip_paths, task_dict['context_annotations'], context_labels, object_list, self.checkpoint_dir, drop_indices, "drop")
                visualize_context_clips(context_clip_paths, annotations_dict, context_labels, object_list, drop_indices, self.checkpoint_dir)
                visualize_target_clips(cached_target_paths_by_video, cached_target_labels_by_video, object_list, self.checkpoint_dir)
                plot_hist(context_labels[keep_indices], list(range(len(object_list)+1)), "keep_distrib", self.checkpoint_dir, user=user, 
                         x_label='Classes', y_label='Count', title="Reduced Class Distribution") #task_num=step%self.args.test_tasks_per_user
                
                # TODO: should we be resetting between each call?
                # reset task's params
                self.model._reset()
                # add task's ops to self.ops_counter
                self.ops_counter.task_complete()
                # if this is the user's last task, get the average performance for the user
                if (step+1) % self.args.test_tasks_per_user == 0:
                    self.print_current_user_stats(self.test_evaluator, task_dict["user_id"], descriptor="full") # Save out clean stats using default test evaluator
                    self.print_current_user_stats(reduced_evaluator, task_dict["user_id"], descriptor="reduced") # Save out stats when training using reduced context set
                    # Notify evaluators that we're moving to next user
                    if (step+1) < num_test_tasks:
                        self.test_evaluator.next_user()
                        reduced_evaluator.next_user()
                        noise_tracker.next_user()

        friendly_str, data_dict = noise_tracker.get_summary_stats_str(noise_tracker.get_mean_stats())
        print_and_log(self.logfile, friendly_str)
        self.test_evaluator.save_user_stats_to_df(os.path.join(self.checkpoint_dir, "full_results.csv"))
        reduced_evaluator.save_user_stats_to_df(os.path.join(self.checkpoint_dir, "{}_results.csv".format(self.args.importance_calculator)))
        self.save_and_reset_stats(self.test_evaluator, path, descriptor="full")
        self.save_and_reset_stats(reduced_evaluator, path, descriptor="reduced")

        for key in data_dict.keys():
            self.logfile.write(key + '\n')
            self.logfile.write(data_dict[key] + '\n')

            
    def print_current_user_stats(self, evaluator, user_id, descriptor=""):
        _, current_user_stats = evaluator.get_mean_stats(current_user=True)
        print_and_log(self.logfile, f'{self.args.test_set} user {user_id} ({evaluator.current_user+1}/{self.test_queue.num_users})  {descriptor} stats: {stats_to_str(current_user_stats)}')

    def save_and_reset_stats(self, evaluator, path, descriptor=""):
        table_str = f'{descriptor}\n'
        for stat in evaluator.stats_to_compute:
            table_str += "," + stat
        table_str += "\n"
        for user in range(0, evaluator.current_user+1):
            table_str += f'{self.args.test_set} user {evaluator.user2userid[user]} ({user+1}/{self.test_queue.num_users})'
            for stat in evaluator.stats_to_compute:
                _, mean, var = evaluator.get_user_stats(user, stat)
                table_str += f',{mean*100:.3f}'
            table_str += '\n'

        self.logfile.write(table_str + '\n')
        stats_per_user, stats_per_video = evaluator.get_mean_stats()
        stats_per_user_str, stats_per_video_str = stats_to_str(stats_per_user), stats_to_str(stats_per_video)
        mean_ops_stats = self.ops_counter.get_mean_stats()
        print_and_log(self.logfile, f'{descriptor} stats\n')
        print_and_log(self.logfile, f'{self.args.test_set} [{path}]\n per-user stats: {stats_per_user_str}\n per-video stats: {stats_per_video_str}\n model stats: {mean_ops_stats}\n')
        evaluator.save(enforce_sequential_frames=False) #We may have removed some target points, don't enforce sequential-ness
        evaluator.reset()

    def save_checkpoint(self, epoch):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_stats': self.validation_evaluator.get_current_best_stats()
            }, os.path.join(self.checkpoint_dir, 'checkpoint.pt'))

    def load_checkpoint(self):
        checkpoint = torch.load(os.path.join(self.checkpoint_dir, 'checkpoint.pt'))
        self.start_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.validation_evaluator.replace(checkpoint['best_stats'])

 
if __name__ == "__main__":
    main()
