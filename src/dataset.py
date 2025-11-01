import cv2
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset

class TennisDataset(Dataset):
    def __init__(self, dataset_path, split='train', transform=None, frame_size=(360, 640)):
        self.dataset_path = Path(dataset_path)
        self.transform = transform
        self.frame_size = frame_size
        self.split = split
        self.sequence_data = []
        
        # Load all sequences
        self._load_sequences()
        
    def _load_sequences(self):
        """Load all valid sequences from the dataset"""
        all_sequences = []
        for game_dir in sorted(self.dataset_path.glob('game*')):
            for clip_dir in sorted(game_dir.glob('Clip*')):
                labels_path = clip_dir / 'Label.csv'
                if not labels_path.exists():
                    continue
                    
                df = pd.read_csv(labels_path)
                df.columns = ['File_Name', 'Visibility_Class', 'X', 'Y', 'Trajectory_Pattern']
                
                # Create sequences of 3 consecutive frames
                for i in range(len(df) - 2):
                    if df.iloc[i:i+3]['Visibility_Class'].all() == 1:  # All frames have visible ball
                        all_sequences.append({
                            'clip_dir': clip_dir,
                            'frames': df.iloc[i:i+3],
                            'target_frame': df.iloc[i+1]  # Middle frame is target
                        })
        
        # Split sequences into train and validation
        np.random.seed(42)  # For reproducibility
        np.random.shuffle(all_sequences)
        
        split_idx = int(len(all_sequences) * 0.8)  # 80% for training
        
        if self.split == 'train':
            self.sequence_data = all_sequences[:split_idx]
        else:  # validation
            self.sequence_data = all_sequences[split_idx:]
    
    def __len__(self):
        return len(self.sequence_data)
    
    def create_gaussian_heatmap(self, size, center, sigma=5):
        """Generate a Gaussian heatmap."""
        height, width = size
        x = np.arange(0, width, 1, float)
        y = np.arange(0, height, 1, float)
        y = y[:, np.newaxis]
        
        x0, y0 = center
        heatmap = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        # Add normalization to ensure peaks are visible
        heatmap = heatmap / heatmap.max()  # Normalize to [0,1]
        return heatmap

    def __getitem__(self, idx):
        sequence = self.sequence_data[idx]
        
        # Load 3 consecutive frames
        frames = []
        for _, frame_data in sequence['frames'].iterrows():
            frame_path = sequence['clip_dir'] / f"{frame_data['File_Name']}"
            frame = cv2.imread(str(frame_path))
            frame = cv2.resize(frame, (self.frame_size[1], self.frame_size[0]))
            frames.append(frame)
        
        # Stack frames into input tensor
        input_tensor = np.concatenate(frames, axis=2)  # Stack along channel dimension
        
        # Create target heatmap using Gaussian
        target = np.zeros((self.frame_size[0], self.frame_size[1]), dtype=np.float32)
        x_scale = self.frame_size[1] / 1280
        y_scale = self.frame_size[0] / 720
        target_x = int(sequence['target_frame']['X'] * x_scale)
        target_y = int(sequence['target_frame']['Y'] * y_scale)
        target_x = np.clip(target_x, 0, self.frame_size[1] - 1)
        target_y = np.clip(target_y, 0, self.frame_size[0] - 1)
        target = self.create_gaussian_heatmap(
            size=(self.frame_size[0], self.frame_size[1]),
            center=(target_x, target_y),
            sigma=5
        )
        
        # Apply transformations if any
        if self.transform:
            transformed = self.transform(image=input_tensor, mask=target)
            input_tensor = transformed['image']
            target = transformed['mask']
        
        # Convert to torch tensors
        input_tensor = torch.from_numpy(input_tensor.transpose(2, 0, 1)).float() / 255.0
        target = torch.from_numpy(target).unsqueeze(0).float()
        
        return input_tensor, target