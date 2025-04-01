from torch.utils.data import Dataset
from random import random
import torch
import random
from data_loader import label2vectormerge, clips2vectormerge
from torchvision.io.video import read_video
import numpy as np
import torchvision.transforms.functional as F
import sys
import traceback

class HRNetMultiViewDataset(Dataset):
    def __init__(self, path, start, end, fps, split, num_views, transform=None, transform_model=None):

        if split != 'Chall':
            # To load the annotations
            self.labels_offence_severity, self.labels_action, self.distribution_offence_severity,self.distribution_action, not_taking, self.number_of_actions = label2vectormerge(path, split, num_views)
            self.clips = clips2vectormerge(path, split, num_views, not_taking)
            self.distribution_offence_severity = torch.div(self.distribution_offence_severity, len(self.labels_offence_severity))
            self.distribution_action = torch.div(self.distribution_action, len(self.labels_action))

            self.weights_offence_severity = torch.div(1, self.distribution_offence_severity)
            self.weights_action = torch.div(1, self.distribution_action)
        else:
            self.clips = clips2vectormerge(path, split, num_views, [])
            # Set default weights for challenge set
            self.weights_offence_severity = torch.ones(4)
            self.weights_action = torch.ones(8)
            self.distribution_offence_severity = torch.ones(4)
            self.distribution_action = torch.ones(8)

        self.split = split
        self.start = start
        self.end = end
        self.transform = transform
        self.transform_model = transform_model
        self.num_views = num_views

        self.factor = (end - start) / (((end - start) / 25) * fps)

        self.length = len(self.clips)

    def getDistribution(self):
        """
        Return distribution of offence severity and action classes
        """
        return self.distribution_offence_severity, self.distribution_action

    def getWeights(self):
        """
        Return weights for offence severity and action classes
        """
        return self.weights_offence_severity, self.weights_action

    def _select_representative_frame(self, video, clip_path):
        """
        Select a representative frame from the video sequence with extensive debugging
        """
        
        try:
            # Handle different possible input shapes
            if len(video.shape) == 4:
                # THWC format
                frames = video[self.start:self.end, :, :, :]
            elif len(video.shape) == 3:
                # HWC format
                frames = video[None, :, :, :]
            else:
                print(f"[ERROR] Unexpected video shape: {video.shape}")
                raise ValueError(f"Unexpected video shape: {video.shape}")
            
            # If total frames are less than expected, return the frame
            if frames.shape[0] <= 1:
                frame = frames[0]
            else:
                # Select middle frame
                mid_frame_index = frames.shape[0] // 2
                frame = frames[mid_frame_index]
            
            # Handle both numpy and torch tensors
            if isinstance(frame, np.ndarray):
                frame = torch.from_numpy(frame)
            
            # Ensure correct format and normalization
            frame = frame.permute(2, 0, 1).float() / 255.0
            
            
            return frame

        except Exception as e:
            print(f"[ERROR] Exception in frame selection: {e}")
            traceback.print_exc()
            raise

    def __getitem__(self, index):

        # Sanity check for clips
        if not self.clips[index]:
            print("[ERROR] No clips found for this index")
            # Fallback to a default behavior
            if self.split != 'Chall':
                return self.labels_offence_severity[index][0], self.labels_action[index][0], torch.zeros(5, 3, 224, 224), self.number_of_actions[index]
            else:
                return -1, -1, torch.zeros(5, 3, 224, 224), str(index)

        prev_views = []
        views_data = []

        # Track view processing attempts
        view_processing_attempts = 0
        view_processing_failures = 0

        max_views = min(len(self.clips[index]), 5)  # Limit to 5 views

        for num_view in range(max_views):
            index_view = num_view
            view_processing_attempts += 1

            if len(prev_views) == 2:
                continue

            # Randomize view selection during training
            cont = True
            if self.split == 'Train':
                while cont:
                    aux = random.randint(0, max_views-1)
                    if aux not in prev_views:
                        cont = False
                index_view = aux
                prev_views.append(index_view)

            # Read video with correct pts_unit
            clip_path = self.clips[index][index_view]
            try:
                video, _, _ = read_video(clip_path, output_format="THWC", pts_unit='sec')
            except Exception as e:
                print(f"[ERROR] Failed to read video {clip_path}: {e}")
                traceback.print_exc()
                view_processing_failures += 1
                continue
            
            # Select representative frame
            try:
                frame = self._select_representative_frame(video, clip_path)
            except Exception as e:
                print(f"[ERROR] Failed to select representative frame: {e}")
                traceback.print_exc()
                view_processing_failures += 1
                continue

            # Apply transforms if any
            if self.transform is not None:
                try:
                    frame = self.transform(frame)
                except Exception as e:
                    print(f"[ERROR] Custom transform failed: {e}")
                    view_processing_failures += 1
                    continue

            # Apply model-specific transforms
            try:
                frame = self.transform_model(frame)
            except Exception as e:
                print(f"[ERROR] Model transform failed: {e}")
                view_processing_failures += 1
                continue
            
            # Store processed view
            views_data.append(frame)

        # Ensure at least 5 views for multi-view processing
        while len(views_data) < 5:
            views_data.append(torch.zeros_like(views_data[0]))

        # Stack views
        try:
            videos = torch.stack(views_data)
        except Exception as e:
            print(f"[ERROR] Failed to stack views: {e}")
            # Fallback to a default tensor
            videos = torch.zeros(5, 3, 224, 224)

        # Permute if necessary to match expected input shape
        if videos.dim() == 5:
            videos = videos.permute(0, 2, 1, 3, 4)  # [V, C, T, H, W]
        
        # Return data
        if self.split != 'Chall':
            return self.labels_offence_severity[index][0], self.labels_action[index][0], videos, self.number_of_actions[index]
        else:
            return -1, -1, videos, str(index)

    def __len__(self):
        return self.length