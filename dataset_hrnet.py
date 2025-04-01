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
        print(f"[DEBUG] Initializing HRNetMultiViewDataset")
        print(f"[DEBUG] Path: {path}")
        print(f"[DEBUG] Start: {start}, End: {end}")
        print(f"[DEBUG] Split: {split}")
        print(f"[DEBUG] Num Views: {num_views}")

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
        print(f"[DEBUG] Dataset length: {self.length}")

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

    def _debug_video_info(self, video, clip_path):
        """
        Print detailed information about the video
        """
        print(f"\n[VIDEO DEBUG] Clip path: {clip_path}")
        print(f"[VIDEO DEBUG] Video shape: {video.shape}")
        print(f"[VIDEO DEBUG] Video dtype: {video.dtype}")
        print(f"[VIDEO DEBUG] Slice details:")
        print(f"  Start: {self.start}")
        print(f"  End: {self.end}")
        
        try:
            sliced_video = video[self.start:self.end, :, :, :]
            print(f"[VIDEO DEBUG] Sliced video shape: {sliced_video.shape}")
        except Exception as e:
            print(f"[VIDEO DEBUG] Error slicing video: {e}")
            traceback.print_exc()

    def _select_representative_frame(self, video, clip_path):
        """
        Select a representative frame from the video sequence with extensive debugging
        """
        # Detailed video debugging
        print(f"\n[VIDEO DEBUG] Clip path: {clip_path}")
        print(f"[VIDEO DEBUG] Video shape: {video.shape}")
        print(f"[VIDEO DEBUG] Video dtype: {video.dtype}")
        print(f"[VIDEO DEBUG] Slice details:")
        print(f"  Start: {self.start}")
        print(f"  End: {self.end}")
        
        try:
            # Handle different possible input shapes
            if len(video.shape) == 4:
                # THWC format
                frames = video[self.start:self.end, :, :, :]
                print(f"[FRAME DEBUG] THWC frames shape: {frames.shape}")
            elif len(video.shape) == 3:
                # HWC format
                frames = video[None, :, :, :]
                print(f"[FRAME DEBUG] HWC frames shape: {frames.shape}")
            else:
                print(f"[ERROR] Unexpected video shape: {video.shape}")
                raise ValueError(f"Unexpected video shape: {video.shape}")
            
            # If total frames are less than expected, return the frame
            if frames.shape[0] <= 1:
                frame = frames[0]
                print("[FRAME DEBUG] Using first frame due to low frame count")
            else:
                # Select middle frame
                mid_frame_index = frames.shape[0] // 2
                frame = frames[mid_frame_index]
                print(f"[FRAME DEBUG] Selected middle frame (index {mid_frame_index})")
            
            # Handle both numpy and torch tensors
            if isinstance(frame, np.ndarray):
                frame = torch.from_numpy(frame)
            
            # Ensure correct format and normalization
            frame = frame.permute(2, 0, 1).float() / 255.0
            
            print(f"[FRAME DEBUG] Final frame shape: {frame.shape}")
            print(f"[FRAME DEBUG] Final frame dtype: {frame.dtype}")
            
            return frame

        except Exception as e:
            print(f"[ERROR] Exception in frame selection: {e}")
            traceback.print_exc()
            raise

    def __getitem__(self, index):
        print(f"\n[GETITEM DEBUG] Processing index: {index}")
        print(f"[GETITEM DEBUG] Clips for this index: {self.clips[index]}")

        prev_views = []
        views_data = []

        for num_view in range(len(self.clips[index])):
            index_view = num_view
            print(f"\n[VIEW DEBUG] Processing view {num_view}")

            if len(prev_views) == 2:
                print("[VIEW DEBUG] Skipping - already processed 2 views")
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
                print(f"[VIEW DEBUG] Randomly selected view {index_view}")

            # Read video with correct pts_unit
            clip_path = self.clips[index][index_view]
            print(f"[VIDEO DEBUG] Reading video: {clip_path}")
            try:
                video, _, _ = read_video(clip_path, output_format="THWC", pts_unit='sec')
            except Exception as e:
                print(f"[ERROR] Failed to read video {clip_path}: {e}")
                traceback.print_exc()
                continue
            
            # Select representative frame
            try:
                frame = self._select_representative_frame(video, clip_path)
            except Exception as e:
                print(f"[ERROR] Failed to select representative frame: {e}")
                traceback.print_exc()
                continue

            # Apply transforms if any
            if self.transform is not None:
                frame = self.transform(frame)
                print(f"[TRANSFORM DEBUG] After custom transform: {frame.shape}")

            # Apply model-specific transforms
            frame = self.transform_model(frame)
            print(f"[TRANSFORM DEBUG] After model transform: {frame.shape}")
            
            # Store processed view
            views_data.append(frame)

        # Stack views
        if len(views_data) > 0:
            videos = torch.stack(views_data)
            print(f"[STACK DEBUG] Stacked views shape: {videos.shape}")
        else:
            print("[ERROR] No valid views processed")
            raise ValueError("No valid views processed")

        # Adjust tensor shape if needed
        if self.num_views != 1 and self.num_views != 5:
            videos = videos.squeeze()   
            print(f"[SQUEEZE DEBUG] After squeeze: {videos.shape}")

        # Permute if necessary to match expected input shape
        if videos.dim() == 4:
            videos = videos.permute(0, 2, 1, 3)  # [V, C, T, H, W]
            print(f"[PERMUTE DEBUG] After permute: {videos.shape}")
        
        # Return data
        if self.split != 'Chall':
            print("[RETURN DEBUG] Returning training/validation data")
            return self.labels_offence_severity[index][0], self.labels_action[index][0], videos, self.number_of_actions[index]
        else:
            print("[RETURN DEBUG] Returning challenge data")
            return -1, -1, videos, str(index)

    def __len__(self):
        return self.length