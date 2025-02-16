import matplotlib.pyplot as plt
import torch
import numpy as np
from pathlib import Path
import cv2
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

def save_training_plot(history, output_path):
    """
    Plot and save training metrics history.
    
    Args:
        history (dict): Dictionary containing training metrics
        output_path (str or Path): Path to save the plot
    """
    plt.figure(figsize=(15, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def save_sample_predictions(inputs, outputs, targets, epoch, batch_idx, output_dir):
    """
    Save visualization of model predictions.
    Args:
        inputs: Input tensor [B, C, H, W]
        outputs: Model predictions [B, 1, H, W]
        targets: Ground truth heatmaps [B, 1, H, W]
        epoch: Current epoch number
        batch_idx: Current batch index
        output_dir: Directory to save visualizations
    """
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Get first sample from batch and detach from computation graph
    input_frames = inputs[0].detach().cpu().numpy()  # [C, H, W]
    pred_heatmap = outputs[0, 0].detach().cpu().numpy()  # [H, W]
    target_heatmap = targets[0, 0].detach().cpu().numpy()  # [H, W]
    
    # Get middle frame from input sequence
    middle_frame = input_frames[3:6].transpose(1, 2, 0)  # Get middle frame, convert to [H, W, C]
    
    # Plot heatmaps
    axes[0, 0].imshow(target_heatmap)
    axes[0, 0].set_title('Ground Truth Heatmap')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(pred_heatmap)
    axes[0, 1].set_title('Predicted Heatmap')
    axes[0, 1].axis('off')
    
    # Plot overlay of both heatmaps
    axes[0, 2].imshow(middle_frame)
    axes[0, 2].imshow(pred_heatmap, alpha=0.5, cmap='hot')
    axes[0, 2].set_title('Prediction Overlay')
    axes[0, 2].axis('off')
    
    # Get coordinates
    target_coords = np.unravel_index(target_heatmap.argmax(), target_heatmap.shape)
    pred_coords = np.unravel_index(pred_heatmap.argmax(), pred_heatmap.shape)
    
    # Plot positions
    axes[1, 1].imshow(middle_frame)
    axes[1, 1].scatter(target_coords[1], target_coords[0], c='r', marker='+', s=100, label='Ground Truth')
    axes[1, 1].set_title('Ground Truth Position')
    axes[1, 1].axis('off')
    axes[1, 1].legend()
    
    axes[1, 2].imshow(middle_frame)
    axes[1, 2].scatter(pred_coords[1], pred_coords[0], c='g', marker='+', s=100, label='Prediction')
    axes[1, 2].set_title('Predicted Position')
    axes[1, 2].axis('off')
    axes[1, 2].legend()
    
    # Original frame for comparison
    axes[1, 0].imshow(middle_frame)
    axes[1, 0].set_title('Original Frame')
    axes[1, 0].axis('off')
    
    # Add error information
    error = np.sqrt(((target_coords[0] - pred_coords[0]) ** 2) + 
                   ((target_coords[1] - pred_coords[1]) ** 2))
    plt.suptitle(f'Epoch {epoch+1}, Batch {batch_idx}\nPositioning Error: {error:.2f} pixels')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'epoch_{epoch}_batch_{batch_idx}.png')
    plt.close()

def generate_trajectory_visualization(frames, positions, output_path):
    """
    Generate visualization of ball trajectory across multiple frames.
    
    Args:
        frames (list): List of frames as numpy arrays
        positions (list): List of (x, y) ball positions
        output_path (str or Path): Path to save visualization
    """
    plt.figure(figsize=(20, 10))
    
    # Create a composite image showing trajectory
    composite = frames[0].copy()
    colors = plt.cm.rainbow(np.linspace(0, 1, len(positions)))
    
    for i, (pos, color) in enumerate(zip(positions, colors)):
        if pos is not None:  # Check if ball was detected
            x, y = pos
            cv2.circle(composite, (int(x), int(y)), 3, color[:3] * 255, -1)
            # Draw frame number
            cv2.putText(composite, str(i), (int(x+5), int(y+5)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color[:3] * 255, 1)
    
    plt.imshow(composite)
    plt.title('Ball Trajectory')
    plt.axis('off')
    
    # Add colorbar to show temporal progression
    sm = plt.cm.ScalarMappable(cmap=plt.cm.rainbow)
    sm.set_array([])
    plt.colorbar(sm, label='Frame Progression')
    
    plt.savefig(output_path)
    plt.close()

def visualize_heatmap_generation(frame, position, sigma=5):
    """
    Visualize how Gaussian heatmap is generated from ball position.
    
    Args:
        frame (numpy.ndarray): Input frame
        position (tuple): (x, y) ball position
        sigma (float): Standard deviation for Gaussian
    """
    height, width = frame.shape[:2]
    x, y = position
    
    # Generate Gaussian heatmap
    x_grid, y_grid = np.meshgrid(np.arange(width), np.arange(height))
    heatmap = np.exp(-((x_grid - x) ** 2 + (y_grid - y) ** 2) / (2 * sigma ** 2))
    heatmap = (heatmap * 255).astype(np.uint8)
    
    plt.figure(figsize=(15, 5))
    
    # Original frame
    axes = plt.subplots(1, 3, 1)[1]  # Get axes array from subplots
    plt.imshow(frame)
    plt.scatter(x, y, c='r', marker='x')
    plt.title('Original Frame with Ball Position')
    plt.axis('off')
    
    # Gaussian heatmap
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap='hot')
    plt.title('Generated Heatmap')
    plt.axis('off')
    
    # Overlay
    plt.subplot(1, 3, 3)
    plt.imshow(frame)
    plt.imshow(heatmap, cmap='hot', alpha=0.5)
    plt.title('Overlay')
    plt.axis('off')
    
    # Add scale bar
    scalebar = AnchoredSizeBar(axes[0].transData,
                              sigma*2, f'{sigma*2} pixels',
                              'lower right',
                              pad=0.5,
                              color='white',
                              frameon=False,
                              size_vertical=1)
    axes[0].add_artist(scalebar)
    
    plt.tight_layout()
    plt.show()

def plot_training_progress(step, train_loss, val_loss, train_acc, val_acc):
    """
    Create a real-time plot of training progress.
    
    Args:
        step (int): Current training step
        train_loss (float): Current training loss
        val_loss (float): Current validation loss
        train_acc (float): Current training accuracy
        val_acc (float): Current validation accuracy
    """
    plt.clf()
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.title('Loss vs. Step')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='Train Acc')
    plt.plot(val_acc, label='Val Acc')
    plt.title('Accuracy vs. Step')
    plt.xlabel('Step')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.pause(0.1)  # Short pause to update the plot

def create_tracking_video(model, video_path, output_path, frame_size=(360, 640)):
    """
    Create video visualization of ball tracking.
    Args:
        model: Trained model
        video_path: Path to input video
        output_path: Path to save output video
        frame_size: Size to resize frames to
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)
    
    frames_buffer = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.resize(frame, frame_size)
        frames_buffer.append(frame)
        
        if len(frames_buffer) == 3:
            # Prepare input
            input_tensor = np.concatenate(frames_buffer, axis=2)
            input_tensor = torch.from_numpy(input_tensor.transpose(2, 0, 1)).float() / 255.0
            input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
            
            # Get prediction
            with torch.no_grad():
                output = model(input_tensor)
            
            # Get predicted position
            pred_heatmap = output[0, 0].cpu().numpy()
            pred_coords = np.unravel_index(pred_heatmap.argmax(), pred_heatmap.shape)
            
            # Draw prediction on middle frame
            frame = frames_buffer[1].copy()
            cv2.drawMarker(frame, (pred_coords[1], pred_coords[0]), 
                          (0, 255, 0), markerType=cv2.MARKER_CROSS, 
                          markerSize=20, thickness=2)
            
            out.write(frame)
            frames_buffer.pop(0)
    
    cap.release()
    out.release()