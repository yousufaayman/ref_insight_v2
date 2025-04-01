from torch.utils.data import Dataset
from random import random
import torch
import random
from data_loader import label2vectormerge, clips2vectormerge
from torchvision.io.video import read_video
import numpy as np
import torchvision.transforms.functional as F

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

        self.split = split
        self.start = start
        self.end = end
        self.transform = transform
        self.transform_model = transform_model
        self.num_views = num_views

        self.factor = (end - start) / (((end - start) / 25) * fps)

        self.length = len(self.clips)
        print(self.length)

    def getDistribution(self):
        return self.distribution_offence_severity, self.distribution_action
    
    def getWeights(self):
        return self.weights_offence_severity, self.weights_action

    def _preprocess_video(self, video, start, end, factor):
        # Ensure video is in THWC format
        frames = video[start:end, :, :, :]
        
        # Select frames based on factor
        final_frames = []
        for j in range(len(frames)):
            if j % factor < 1:
                final_frames.append(frames[j])
        
        # Convert to tensor and permute to TCHW
        final_frames = torch.stack(final_frames)
        final_frames = final_frames.permute(0, 3, 1, 2)  # THWC -> TCHW
        
        # Convert to float and scale to [0, 1]
        final_frames = final_frames.float() / 255.0
        
        return final_frames

    def __getitem__(self, index):
        prev_views = []
        views_data = []

        for num_view in range(len(self.clips[index])):
            index_view = num_view

            if len(prev_views) == 2:
                continue

            # Randomize view selection during training
            cont = True
            if self.split == 'Train':
                while cont:
                    aux = random.randint(0, len(self.clips[index])-1)
                    if aux not in prev_views:
                        cont = False
                index_view = aux
                prev_views.append(index_view)

            # Read video with correct pts_unit
            video, _, _ = read_video(self.clips[index][index_view], output_format="THWC", pts_unit='sec')
            
            # Preprocess video frames
            final_frames = self._preprocess_video(video, self.start, self.end, self.factor)

            # Apply transforms
            if self.transform is not None:
                final_frames = self.transform(final_frames)

            # Apply model-specific transforms
            final_frames = self.transform_model(final_frames)
            
            # Store processed view
            views_data.append(final_frames)

        # Stack views
        if len(views_data) > 0:
            videos = torch.stack(views_data)
        else:
            raise ValueError("No valid views processed")

        # Adjust tensor shape if needed
        if self.num_views != 1 and self.num_views != 5:
            videos = videos.squeeze()   

        videos = videos.permute(0, 2, 1, 3, 4)  # Adjust dimensions if needed

        if self.split != 'Chall':
            return self.labels_offence_severity[index][0], self.labels_action[index][0], videos, self.number_of_actions[index]
        else:
            return -1, -1, videos, str(index)

    def __len__(self):
        return self.length