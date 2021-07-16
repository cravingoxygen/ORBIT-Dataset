"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license

This code was based on the file model.py (https://github.com/cambridge-mlg/cnaps/blob/master/src/model.py) and
config_networks.py (https://github.com/cambridge-mlg/cnaps/blob/master/src/config_networks.py)
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

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from features import extractors
from feature_adapters import FilmAdapter, NullAdapter
from models.poolers import MeanPooler
from models.normalisation_layers import TaskNorm
from models.set_encoder import SetEncoder, NullSetEncoder
from models.classifiers import LinearClassifier, VersaClassifier, PrototypicalClassifier, MahalanobisClassifier
from utils.optim import init_optimizer
from utils.data import ListBatcher, FrameLoader, attach_frame_history

class FewShotRecogniser(nn.Module):
    """
    Generic few-shot classification model.
    """
    def __init__(self, args):
        """
        Creates instance of FewShotRecogniser.
        :param args: (argparse.NameSpace) Command line arguments to configure model.
        :return: Nothing.
        """
        super(FewShotRecogniser, self).__init__()
        self.args = args
        pretrained=True if self.args.pretrained_extractor_path else False
        
        # configure feature extractor
        extractor_fn = extractors[ self.args.feature_extractor ]
        self.feature_extractor = extractor_fn(
            pretrained=pretrained,
            pretrained_model_path=self.args.pretrained_extractor_path,
            batch_norm=self.args.batch_normalisation,
            with_film=self.args.adapt_features
        )
        if not self.args.learn_extractor:
            self._freeze_extractor()
     
        # configure feature adapter
        if self.args.adapt_features:     
            if self.args.feature_adaptation_method == 'generate':
                self.set_encoder = SetEncoder(self.args.batch_normalisation)
                adaptation_layer = self.feature_extractor._get_adaptation_layer(generatable=True)
            else:
                self.set_encoder = NullSetEncoder()
                adaptation_layer = self.feature_extractor._get_adaptation_layer(generatable=False)
            self.feature_adapter = FilmAdapter(
                layer=adaptation_layer,
                adaptation_config = self.feature_extractor._get_adaptation_config(),
                task_dim=self.set_encoder.output_size
            ) 
        else:
            self.set_encoder = NullSetEncoder()
            self.feature_adapter = NullAdapter() 
         
        # configure classifier
        if self.args.classifier == 'linear': 
            # classifier head will instead be appended per-task during train/test
            self.classifier = LinearClassifier(self.feature_extractor.output_size)
        elif self.args.classifier == 'versa':
            self.classifier = VersaClassifier(self.feature_extractor.output_size)
        elif self.args.classifier == 'proto':
            self.classifier = PrototypicalClassifier()
        elif self.args.classifier == 'mahalanobis':
            self.classifier = MahalanobisClassifier() 
            
        # configure frame pooler
        self.frame_pooler = MeanPooler(T=self.args.clip_length)

        # configure batchers
        self.inner_batcher = ListBatcher(self.args.batch_size)
        self.outer_batcher = ListBatcher(self.args.batch_size)

        # configure frame loader
        self.frame_loader = FrameLoader(self.args.clip_length, self.args.frame_size)

    def _set_device(self, device):
        self.device = device
    
    def _send_to_device(self):
        """
        Function that moves whole model to self.device.
        :return: Nothing.
        """
        self.to(self.device)
        if self.args.use_two_gpus:
            self._distribute_model()

    def _distribute_model(self):
        """
        Function that moves the feature extractor and feature adapter to the second GPU.
        :return: Nothing.
        """
        self.feature_extractor.cuda(1)
        self.feature_adapter.cuda(1)
    
    def batch_predict(self, target_clip_paths):
        """
        Function that processes target clips in batches to get logits over object classes.
        :param target_clip_paths: (np.ndarray) Target clip paths, each composed of self.args.clip_length contiguous frames.
        :return: (torch.Tensor) Logits over object classes for each clip in target_clips_paths.
        """
        num_batches = self.outer_batcher._get_number_of_batches(len(target_clip_paths))
        target_logits = []
        for batch_id in range(num_batches):
            batch_range = self.outer_batcher._get_batch_indices(batch_id)
            #batch_clips = self.frame_loader(target_clip_paths[batch_range], device=self.device)
            batch_logits = self.predict(target_clip_paths[batch_range])
            target_logits.append(batch_logits)
        return torch.cat(target_logits, dim=0)
    
    def _get_features(self, clip_paths, feature_adapter_params, ops_counter=None, context=False):
        """
        Function that passes clips through an adapted feature extractor to get adapted (and flattened) frame features.
        :param clip_paths: (np.ndarray) Clip paths, each composed of self.args.clip_length contiguous frames.
        :param feature_adapter_params: (list::dict::torch.Tensor or list::dict::list::torch.Tensor) Parameters of all FiLM layers.
        :param ops_counter: (utils.OpsCounter or None) Object that counts operations performed.
        :param context: (bool) True if clips are from context videos, otherwise False.
        :return: (torch.Tensor) Adapted frame features flattened across all clips.
        """ 
        features = []
        num_batches = self.inner_batcher._get_number_of_batches(len(clip_paths))
        self._set_model_state(context)

        for batch_id in range(num_batches):
            batch_range = self.inner_batcher._get_batch_indices(batch_id)
            batch_clips = self.frame_loader(clip_paths[batch_range], device=self.device)
            if self.args.use_two_gpus:
                batch_clips = batch_clips.cuda(1)
                batch_features = self.feature_extractor(batch_clips, feature_adapter_params).cuda(0)
            else:
                batch_features = self.feature_extractor(batch_clips, feature_adapter_params)

            if ops_counter:
                ops_counter.compute_macs(self.feature_extractor, batch_clips, feature_adapter_params)
            
            features.append(batch_features)

        return torch.cat(features, dim=0)

    def _get_feature_adapter_params(self, task_embedding, ops_counter=None):
        """
        Function that processes a task embedding to obtain parameters of the feature adapter.
        :param task_embedding: (torch.Tensor or None) Embedding of a whole task's context set.
        :param ops_counter: (utils.OpsCounter or None) Object that counts operations performed.
        :return: (list::dict::torch.Tensor or list::dict::list::torch.Tensor or None) Parameters of all FiLM layers.
        """ 
        if ops_counter:
            ops_counter.compute_macs(self.feature_adapter, task_embedding)
        
        return self.feature_adapter(task_embedding)
   
    def _get_task_embedding(self, context_clip_paths, ops_counter=None, reduction='mean'):
        """
        Function that passes all of a task's context set through the set encoder to get a task embedding.
        :param context_clip_paths: (np.ndarray) Context clip paths, each composed of self.args.clip_length contiguous frames.
        :param ops_counter: (utils.OpsCounter or None) Object that counts operations performed.
        :param reduction: (str) Method to aggregate clip encodings from self.set_encoder.
        :return: (torch.Tensor or None) Task embedding.
        """ 
        reps = []
        num_batches = self.inner_batcher._get_number_of_batches(len(context_clip_paths))
        
        for batch_id in range(num_batches):
            batch_range = self.inner_batcher._get_batch_indices(batch_id)
            batch_clips = self.frame_loader(context_clip_paths[batch_range], device=self.device)
            batch_reps = self.set_encoder(batch_clips)

            if ops_counter:
                ops_counter.compute_macs(self.set_encoder, batch_clips)

            reps.append(batch_reps)

        return self.set_encoder.aggregate(reps, reduction=reduction, switch_device=self.args.use_two_gpus)

    def _pool_features(self, features, ops_counter=None):
        """
        Function that pools frame features per clip.
        :param features: (torch.Tensor) Frame features i.e. flattened as (num_clips*self.args.clip_length) x (feat_dim).
        :param ops_counter: (utils.OpsCounter or None) Object that counts operations performed.
        :return: (torch.Tensor) Frame features pooled per clip i.e. as (num_clips) x (feat_dim).
        """ 
        if ops_counter:
            ops_counter.add_macs(features.size(0) * features.size(1) * self.args.clip_length)
        return self.frame_pooler(features)
    
    def _freeze_extractor(self):
        """
        Function that freezes all parameters in the feature extractor.
        :return: Nothing.
        """
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def _register_extra_parameters(self):
        """
        Function that registers TaskNorm layers as parameters if self.args.batch_normalisation == 'task_norm'
        :return: Nothing.
        """
        for module in self.modules():
            if isinstance(module, TaskNorm):
                module.register_extra_weights()
 
    def _set_model_state(self, context=False):
        """
        Function that sets modules to appropriate train() or eval() states. Note, only modules that use batch norm (self.set_encoder, self.feature_extractor) and dropout (none) are affected.
        :param context: (bool) True if a context set is being processed, otherwise False.
        :return: Nothing.
        """
        self.set_encoder.train() # set encoder always in train mode (it processes context data)
        self.feature_extractor.eval()
        if self.args.batch_normalisation == 'basic':
            if self.args.learn_extractor and not self.test_mode:
                self.feature_extractor.train() # compute batch statistics in extractor if unfrozen and train mode
        elif self.args.batch_normalisation == 'task_norm':
            if context:
                self.feature_extractor.train() # compute batch statistics when processing context set
    
    def set_test_mode(self, test_mode):
        """
        Function that flags if model is being evaluated. Relevant for self._set_model_state().
        :param test_mode: (bool) True if model is being evaluated, otherwise True.
        :return: Nothing.
        """
        self.test_mode = test_mode
    
    def _reset(self):
        """
        Function that resets model's classifier after a task is processed.
        :return: Nothing.
        """
        self.classifier.reset() 

class MultiStepFewShotRecogniser(FewShotRecogniser):
    """
    Few-shot classification model that is personalised in multiple forward-backward steps (e.g. MAML, FineTuner).
    """
    def __init__(self, args):
        """
        Creates instance of MultiStepFewShotRecogniser.
        :param args: (argparse.NameSpace) Command line arguments to configure model.
        :return: Nothing.
        """
        FewShotRecogniser.__init__(self, args)

    def personalise(self, context_clip_paths, context_labels, learning_args, ops_counter=None):
        """
        Function that learns a new task by taking a fixed number of gradient steps on the task's context set. For each task, a new linear classification layer is added (and FiLM layers if self.args.adapt_features == True).
        :param context_clip_paths: (np.ndarray) Context clip paths, each composed of self.args.clip_length contiguous frames.
        :param context_labels: (torch.Tensor) Video-level labels for each context clip.
        :param learning_args: (float, func, str, float) Learning hyper-parameters including learning rate, loss function, optimiser type and factor to scale the extractor's learning rate.
        :param ops_counter: (utils.OpsCounter or None) Object that counts operations performed.
        :return: Nothing.
        """ 
        lr, loss_fn, optimizer_type, extractor_scale_factor = learning_args
        num_classes = len(torch.unique(context_labels))
        self.configure_classifier(num_classes, init_zeros=True)
        self.configure_feature_adapter()
        inner_loop_optimizer = init_optimizer(self, lr, optimizer_type, extractor_scale_factor)
        num_context_batches = self.outer_batcher._get_number_of_batches(len(context_clip_paths))

        for _ in range(self.args.num_grad_steps): 
            for batch_id in range(num_context_batches):
                batch_range = self.outer_batcher._get_batch_indices(batch_id)
                batch_context_clips = self.frame_loader(context_clip_paths[batch_range], device=self.device)
                batch_context_labels = context_labels[batch_range].to(self.device)
                batch_context_logits = self.predict(batch_context_clips, ops_counter=ops_counter, context=True)
                batch_context_loss = loss_fn(batch_context_logits, batch_context_labels)
                batch_context_loss.backward()
                 
            inner_loop_optimizer.step()
            inner_loop_optimizer.zero_grad()
    
    def predict(self, clip_paths, ops_counter=None, context=False):
        """
        Function that processes a batch of clips to get logits over object classes for each clip.
        :param clip_paths: (np.ndarray) Clip paths, each composed of self.args.clip_length contiguous frames.
        :param ops_counter: (utils.OpsCounter or None) Object that counts operations performed. 
        :param context: (bool) True if a context set is being processed, otherwise False.
        :return: (torch.Tensor) Logits over object classes for each clip in clips.
        """
        task_embedding = self._get_task_embedding(clip_paths, ops_counter)
        self.feature_adapter_params = self._get_feature_adapter_params(task_embedding, ops_counter)
        features = self._get_features(clip_paths, self.feature_adapter_params, ops_counter, context=context)
        features = self._pool_features(features, ops_counter)
        return self.classifier.predict(features)
    
    def personalise_with_lite(self, context_clip_paths, context_labels):
        NotImplementedError

    def configure_classifier(self, num_classes, init_zeros=False):
        """
        Function that initialises and appends a linear classification layer to the model.
        :param num_classes: (int) Number of classes in classification layer.
        :init_zeros: (bool) If True, initialise classification layer with zeros, otherwise use Kaiming uniform.
        :return: Nothing.
        """
        self.classifier.configure(num_classes, self.device, init_zeros=init_zeros)
    
    def configure_feature_adapter(self):
        """
        Function that initialises learnable FiLM layers if self.args.adapt_features == True.
        :return: Nothing.
        """
        if self.args.adapt_features and self.args.feature_adaptation_method == 'learn':
            self.feature_adapter._init_layers()
            self.feature_adapter.to(self.device)
     
class SingleStepFewShotRecogniser(FewShotRecogniser):
    """
    Few-shot classification model that is personalised in a single forward step (e.g. CNAPs, ProtoNets).
    """
    def __init__(self, args):
        """
        Creates instance of SingleStepFewShotRecogniser.
        :param args: (argparse.NameSpace) Command line arguments to configure model.
        :return: Nothing.
        """
        FewShotRecogniser.__init__(self, args)

    def personalise(self, context_clip_paths, context_labels, ops_counter=None):
        """
        Function that learns a new task by performing a forward pass of the task's context set.
        :param context_clip_paths: (np.ndarray) Context clip paths, each composed of self.args.clip_length contiguous frames.
        :param context_labels: (torch.Tensor) Video-level labels for each context clip.
        :param ops_counter: (utils.OpsCounter or None) Object that counts operations performed.
        :return: Nothing.
        """ 
        task_embedding = self._get_task_embedding(context_clip_paths, ops_counter)
        self.feature_adapter_params = self._get_feature_adapter_params(task_embedding, ops_counter)
        context_features = self._get_features(context_clip_paths, self.feature_adapter_params, ops_counter, context=True)
        context_features = self._pool_features(context_features, ops_counter)
        self.classifier.configure(context_features, context_labels, ops_counter)
       
    def personalise_with_lite(self, context_clip_paths, context_labels):
        """
        Function that learns a new task by performning a forward pass of the task's context set with LITE. Namely a random subset of the context set (self.args.num_lite_samples) is processed with back-propagation enabled, while the remainder is processed with back-propagation disabled.
        :param context_clip_paths: (np.ndarray) Context clip paths, each composed of self.args.clip_length contiguous frames.
        :param context_labels: (torch.Tensor) Video-level labels for each context clip.
        :return: Nothing.
        """ 
        shuffled_idxs = np.random.permutation(len(context_clip_paths))
        task_embedding = self._get_task_embedding_with_lite(context_clip_paths, shuffled_idxs)
        self.feature_adapter_params = self._get_feature_adapter_params(task_embedding)
        context_features = self._get_pooled_features_with_lite(context_clip_paths, shuffled_idxs) 
        self.classifier.configure(context_features, context_labels[shuffled_idxs])
    
    def _cache_context_outputs(self, context_clip_paths):
        """
        Function that performs a forward pass with a task's entire context set with back-propagation disabled and caches the individual 1) encodings from the set encoder and 2) adapted features from the adapted feature extractor, for each clip.
        :param context_clip_paths: (np.ndarray) Context clip paths, each composed of self.args.clip_length contiguous frames.
        :return: Nothing.
        """ 
        with torch.set_grad_enabled(False):
            # cache encoding for each clip from self.set_encoder
            self.cached_set_encoder_reps = self._get_task_embedding(context_clip_paths, reduction='none')

            # get feature adapter parameters
            task_embedding = self.set_encoder.mean_pool(self.cached_set_encoder_reps)
            feature_adapter_params = self._get_feature_adapter_params(task_embedding)

            # cache adapted features for each clip
            context_features = self._get_features(context_clip_paths, feature_adapter_params, context=True)
            self.cached_context_features = self._pool_features(context_features)
       
    def _get_task_embedding_with_lite(self, context_clip_paths, idxs):
        """
        Function that passes all of a task's context set through the set encoder to get a task embedding with LITE.
        :param context_clip_paths: (np.ndarray) Context clip paths, each composed of self.args.clip_length contiguous frames.
        :param idxs: (torch.Tensor) Indicies of elements in context_clips to process with back-propagation enabled.
        :return: (torch.Tensor or None) Task embedding.
        """ 
        if isinstance(self.set_encoder, NullSetEncoder):
            return None
        H = self.args.num_lite_samples
        task_embedding_with_grads = self._get_task_embedding(context_clip_paths[idxs][:H], reduction='none')
        task_embedding_without_grads = self.cached_set_encoder_reps[idxs][H:]
        return torch.cat((task_embedding_with_grads, task_embedding_without_grads)).mean(dim=0)
        
    def _get_pooled_features_with_lite(self, context_clip_paths, idxs):
        """
        Function that gets adapted clip features for a task's context set with LITE.
        :param context_clip_paths: (np.ndarray) Context clip paths, each composed of self.args.clip_length contiguous frames.
        :param idxs: (torch.Tensor) Indicies of elements in context_clips to process with back-propagation enabled.
        :return: (torch.Tensor) Adapted frame features pooled per clip i.e. as (num_clips) x (feat_dim).
        """ 
        H = self.args.num_lite_samples
        context_features_with_grads = self._get_features(context_clip_paths[idxs][:H], self.feature_adapter_params, context=True)
        context_features_with_grads = self._pool_features(context_features_with_grads)
        context_features_without_grads = self.cached_context_features[idxs][H:]
        return torch.cat((context_features_with_grads, context_features_without_grads)) 
    
    def predict(self, target_clip_paths):
        """
        Function that processes a batch of target clips to get logits over object classes for each clip.
        :param target_clip_paths: (np.ndarray) Target clip paths, each composed of self.args.clip_length contiguous frames.
        :return: (torch.Tensor) Logits over object classes for each clip in target_clips.
        """
        target_features = self._get_features(target_clip_paths, self.feature_adapter_params)
        target_features = self._pool_features(target_features) 
        return self.classifier.predict(target_features) 