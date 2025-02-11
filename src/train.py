import os
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adadelta
import numpy as np
from tqdm import tqdm
import logging
from datetime import datetime



from model import TrackNetModel
from dataset import TennisDataset
from utils.visualization import save_training_plot, save_sample_predictions
from config import INPUT_HEIGHT, INPUT_WIDTH, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, STEPS_PER_EPOCH

def setup_logging(output_dir):
    """Setup logging configuration"""
    log_file = output_dir / 'training.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def create_output_dirs(base_dir):
    """Create necessary output directories"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(base_dir) / f'training_{timestamp}'
    
    dirs = {
        'root': output_dir,
        'checkpoints': output_dir / 'checkpoints',
        'visualizations': output_dir / 'visualizations'
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
        
    return dirs

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, logger, vis_dir):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_acc = 0
    
    progress_bar = tqdm(enumerate(train_loader), total=min(STEPS_PER_EPOCH, len(train_loader)),
                       desc=f'Epoch {epoch+1}/{NUM_EPOCHS}')
    
    for batch_idx, (inputs, targets) in progress_bar:
        if batch_idx >= STEPS_PER_EPOCH:
            break
            
        # Move data to device
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        accuracy = calculate_accuracy(outputs, targets)
        
        # Update metrics
        total_loss += loss.item()
        total_acc += accuracy
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'accuracy': f'{accuracy:.4f}'
        })
        
        # Save sample predictions periodically
        if batch_idx % 50 == 0:
            save_sample_predictions(inputs, outputs, targets, epoch, batch_idx, vis_dir)
        
        # Add debugging information periodically
        if batch_idx % 50 == 0:
            pred_max = outputs.max().item()
            pred_min = outputs.min().item()
            target_max = targets.max().item()
            target_min = targets.min().item()
            logger.info(f"Pred range: [{pred_min:.4f}, {pred_max:.4f}], Target range: [{target_min:.4f}, {target_max:.4f}]")
    
    # Calculate epoch metrics
    avg_loss = total_loss / min(STEPS_PER_EPOCH, len(train_loader))
    avg_acc = total_acc / min(STEPS_PER_EPOCH, len(train_loader))
    
    return avg_loss, avg_acc

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    total_acc = 0
    val_steps = min(STEPS_PER_EPOCH // 5, len(val_loader))
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            if batch_idx >= val_steps:
                break
                
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            accuracy = calculate_accuracy(outputs, targets)
            total_loss += loss.item()
            total_acc += accuracy
    
    avg_loss = total_loss / val_steps
    avg_acc = total_acc / val_steps
    
    return avg_loss, avg_acc

def calculate_accuracy(outputs, targets, radius=30):
    """
    Calculate detection accuracy.
    Args:
        outputs: Model predictions [B, 1, H, W]
        targets: Ground truth heatmaps [B, 1, H, W]
        radius: Radius in pixels for considering a prediction correct
    """
    batch_size = outputs.size(0)
    correct = 0
    
    for i in range(batch_size):
        pred = outputs[i, 0]  # [H, W]
        target = targets[i, 0]  # [H, W]
        
        # Get predicted position (max value in predicted heatmap)
        _, pred_idx = torch.max(pred.view(-1), 0)
        pred_y = pred_idx.item() // pred.size(1)
        pred_x = pred_idx.item() % pred.size(1)
        
        # Get target position (max value in target heatmap)
        _, target_idx = torch.max(target.view(-1), 0)
        target_y = target_idx.item() // target.size(1)
        target_x = target_idx.item() % target.size(1)
        
        # Calculate Euclidean distance
        distance = torch.sqrt(torch.tensor(
            (pred_x - target_x) ** 2 + (pred_y - target_y) ** 2
        ).float())
        
        # Consider prediction correct if within radius
        if distance <= radius:
            correct += 1
            
        # Add debugging info
        if i == 0:  # Print info for first item in batch
            print(f"Pred pos: ({pred_x}, {pred_y}), Target pos: ({target_x}, {target_y})")
            print(f"Distance: {distance}")
    
    accuracy = correct / batch_size
    return accuracy

def init_weights(m):
    """Initialize model weights as per paper specifications"""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.uniform_(m.weight, -0.05, 0.05)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.1)  # Small positive bias to avoid dead neurons

def focal_bce_loss(pred, target, gamma=1):
    """BCE loss with focal term to prevent class imbalance"""
    bce_loss = nn.BCELoss(reduction='none')(pred, target)
    pt = torch.exp(-bce_loss)
    focal_loss = ((1 - pt) ** gamma * bce_loss).mean()
    return focal_loss

def train(args):
    # Create output directories
    dirs = create_output_dirs(args.output_dir)
    logger = setup_logging(dirs['root'])
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Create datasets
    train_dataset = TennisDataset(
        args.data_dir,
        split='train',
        frame_size=(INPUT_HEIGHT, INPUT_WIDTH)
    )
    
    val_dataset = TennisDataset(
        args.data_dir,
        split='val',
        frame_size=(INPUT_HEIGHT, INPUT_WIDTH)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    model = TrackNetModel().to(device)
    model.apply(init_weights)
    
    # Loss and optimizer
    criterion = focal_bce_loss
    optimizer = Adadelta(model.parameters(), lr=LEARNING_RATE, rho=0.95, eps=1e-8, weight_decay=1e-4)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Training loop
    logger.info('Starting training...')
    logger.info(f'Training device: {device}')
    logger.info(f'Batch size: {BATCH_SIZE}')
    logger.info(f'Steps per epoch: {STEPS_PER_EPOCH}')
    
    for epoch in range(NUM_EPOCHS):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer,
            device, epoch, logger, dirs['visualizations']
        )
        
        # Validate
        logger.info('Running validation...')
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Log metrics
        logger.info(f'Epoch {epoch+1}/{NUM_EPOCHS}:')
        logger.info(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}')
        logger.info(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}')
        
        # Save checkpoint
        if (epoch + 1) % args.checkpoint_freq == 0:
            checkpoint_path = dirs['checkpoints'] / f'model_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'history': history
            }, checkpoint_path)
            logger.info(f'Saved checkpoint to {checkpoint_path}')
        
        # Save training plot
        plot_path = dirs['visualizations'] / 'training_history.png'
        save_training_plot(history, plot_path)
    
    return model, history

def main():
    parser = argparse.ArgumentParser(description='Train TrackNet model')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Path to output directory')
    parser.add_argument('--checkpoint_freq', type=int, default=10,
                      help='Frequency of saving checkpoints (epochs)')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of data loading workers')
    
    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()