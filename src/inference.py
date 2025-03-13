import torch
import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

from model import TrackNetModel
from utils.video import preprocess_video
from config import MODEL_DIR
from utils.homography import compute_homography, compute_velocity, filter_speeds, get_speed_stats, visualize_homography

def load_model(model_path, device):
    """Load trained TrackNet model"""
    model = TrackNetModel().to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    
    # Extract model state dict if it's wrapped in a checkpoint
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    model.load_state_dict(state_dict)
    model.eval()
    return model

def prepare_input_frames(frames, device):
    """Prepare sequence of 3 frames for model input"""
    # Stack 3 frames together
    stacked = np.concatenate(frames, axis=2)  # Shape: (H, W, 9)
    
    # Convert to tensor and normalize
    tensor = torch.from_numpy(stacked).permute(2, 0, 1).float() / 255.0
    
    # Add batch dimension
    return tensor.unsqueeze(0).to(device)  # Shape: (1, 9, H, W)

def get_ball_position(heatmap, threshold=0.5):
    """Extract ball position from heatmap"""
    heatmap = heatmap.squeeze().cpu().numpy()
    
    max_val = np.max(heatmap)
    if max_val < threshold:
        return None
    
    y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    return (int(x), int(y))

def draw_trajectory(frame, positions, radius=3):
    """Draw ball trajectory on frame"""
    # Draw detected positions
    for pos in positions:
        if pos is not None:
            cv2.circle(frame, pos, radius, (0, 255, 0), -1)
    
    # Draw lines connecting consecutive positions
    for i in range(len(positions)-1):
        if positions[i] is not None and positions[i+1] is not None:
            cv2.line(frame, positions[i], positions[i+1], (0, 255, 0), 2)
    
    return frame

def select_court_points(frame):
    """Interactive court point selection using service box and net area points"""
    display_frame = frame.copy()
    
    points = {}
    point_names = [
        # Original 6 points
        'near_service_left',    # Near court left service line intersection
        'near_service_right',   # Near court right service line intersection
        'near_service_center',  # Near court center service line intersection
        'far_service_left',     # Far court left service line intersection
        'far_service_right',    # Far court right service line intersection
        'far_service_center',   # Far court center service line intersection
        
        # Additional points for better homography
        'far_baseline_left',    # Far court left baseline corner
        'far_baseline_right',   # Far court right baseline corner
        'far_baseline_center',  # Far court center baseline point
        'net_left',             # Left net post area
        'net_right',            # Right net post area
        'net_center',           # Center net point
    ]
    
    # Define which points are required vs optional
    required_points = point_names[:6]  # First 6 points are required
    optional_points = point_names[6:]  # Remaining points are optional
    
    print("\nPlease select the following court points in order:")
    print("The first 6 points are required, the rest are optional but will improve accuracy:")
    print("\nREQUIRED POINTS:")
    for i, name in enumerate(required_points, 1):
        print(f"{i}. {name.replace('_', ' ').title()}")
    
    print("\nOPTIONAL POINTS (for better accuracy):")
    for i, name in enumerate(optional_points, len(required_points) + 1):
        print(f"{i}. {name.replace('_', ' ').title()}")
    
    print("\nPress 'c' to continue after selecting at least the required points")
    print("Press 'q' to quit point selection\n")
    
    current_point_idx = 0
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal current_point_idx
        if event == cv2.EVENT_LBUTTONDOWN:
            if current_point_idx < len(point_names):
                point_name = point_names[current_point_idx]
                points[point_name] = (x, y)
                print(f"Selected {point_name}: ({x}, {y})")
                
                # Draw point
                cv2.circle(display_frame, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(display_frame, str(current_point_idx + 1), (x+10, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.imshow('Select Court Points', display_frame)
                
                current_point_idx += 1
    
    # Display instructions with visual guide
    guide_frame = display_frame.copy()
    
    # Add text instructions
    cv2.putText(guide_frame, "Select court points as numbered below:", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(guide_frame, "Required: 1-6, Optional: 7-12", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Draw a simple diagram of the court points to select
    h, w = guide_frame.shape[:2]
    center_x, center_y = w // 2, h // 2
    
    # Draw near service line
    near_service_y = center_y + 100
    left_x = center_x - 100
    right_x = center_x + 100
    cv2.line(guide_frame, (left_x, near_service_y), (right_x, near_service_y), (0, 0, 255), 2)
    
    # Draw far service line
    far_service_y = center_y - 100
    cv2.line(guide_frame, (left_x, far_service_y), (right_x, far_service_y), (0, 0, 255), 2)
    
    # Draw singles lines
    cv2.line(guide_frame, (left_x, far_service_y - 100), (left_x, near_service_y), (0, 0, 255), 2)
    cv2.line(guide_frame, (right_x, far_service_y - 100), (right_x, near_service_y), (0, 0, 255), 2)
    
    # Draw center line
    center_x = (left_x + right_x) // 2
    cv2.line(guide_frame, (center_x, far_service_y - 100), (center_x, near_service_y), (0, 0, 255), 2)
    
    # Draw far baseline
    far_baseline_y = far_service_y - 100
    cv2.line(guide_frame, (left_x, far_baseline_y), (right_x, far_baseline_y), (0, 0, 255), 2)
    
    # Draw net line
    net_y = center_y
    cv2.line(guide_frame, (left_x, net_y), (right_x, net_y), (0, 0, 255), 2)
    
    # Mark required points to select
    # Near service line points
    cv2.circle(guide_frame, (left_x, near_service_y), 5, (255, 0, 0), -1)
    cv2.putText(guide_frame, "1", (left_x - 15, near_service_y - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    cv2.circle(guide_frame, (right_x, near_service_y), 5, (255, 0, 0), -1)
    cv2.putText(guide_frame, "2", (right_x + 10, near_service_y - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    cv2.circle(guide_frame, (center_x, near_service_y), 5, (255, 0, 0), -1)
    cv2.putText(guide_frame, "3", (center_x + 10, near_service_y - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Far service line points
    cv2.circle(guide_frame, (left_x, far_service_y), 5, (255, 0, 0), -1)
    cv2.putText(guide_frame, "4", (left_x - 15, far_service_y + 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    cv2.circle(guide_frame, (right_x, far_service_y), 5, (255, 0, 0), -1)
    cv2.putText(guide_frame, "5", (right_x + 10, far_service_y + 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    cv2.circle(guide_frame, (center_x, far_service_y), 5, (255, 0, 0), -1)
    cv2.putText(guide_frame, "6", (center_x + 10, far_service_y + 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Mark optional points
    # Far baseline points
    cv2.circle(guide_frame, (left_x, far_baseline_y), 5, (0, 255, 255), -1)
    cv2.putText(guide_frame, "7", (left_x - 15, far_baseline_y + 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    cv2.circle(guide_frame, (right_x, far_baseline_y), 5, (0, 255, 255), -1)
    cv2.putText(guide_frame, "8", (right_x + 10, far_baseline_y + 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    cv2.circle(guide_frame, (center_x, far_baseline_y), 5, (0, 255, 255), -1)
    cv2.putText(guide_frame, "9", (center_x + 10, far_baseline_y + 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    # Net points
    cv2.circle(guide_frame, (left_x, net_y), 5, (0, 255, 255), -1)
    cv2.putText(guide_frame, "10", (left_x - 20, net_y - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    cv2.circle(guide_frame, (right_x, net_y), 5, (0, 255, 255), -1)
    cv2.putText(guide_frame, "11", (right_x + 10, net_y - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    cv2.circle(guide_frame, (center_x, net_y), 5, (0, 255, 255), -1)
    cv2.putText(guide_frame, "12", (center_x + 10, net_y - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    cv2.imshow('Select Court Points', guide_frame)
    cv2.setMouseCallback('Select Court Points', mouse_callback)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Check if we have at least the required points
            if all(point in points for point in required_points):
                break
            else:
                print("Please select at least the required points (1-6) before continuing")
    
    cv2.destroyAllWindows()
    
    # Ensure we have all required points
    if not all(point in points for point in required_points):
        raise ValueError(f"All {len(required_points)} required court points must be selected")
    
    return points

def process_video(video_path, model_path, output_path, device='cuda', visualize_homography_flag=False, overlay_court=False):
    """Process video and detect ball trajectory"""
    # Load model
    model = load_model(model_path, device)
    
    # Get video properties
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
        
    # Get FPS from video
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30  # fallback to 30 if FPS cannot be determined
    cap.release()
    
    # Preprocess video
    frames = preprocess_video(video_path)
    if not frames:
        return
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (frames[0].shape[1], frames[0].shape[0])
    )
    
    # Get first frame for court point selection
    first_frame = frames[0].copy()
    print("Please select court points...")
    court_points = select_court_points(first_frame)
    
    # Compute homography matrix
    H = compute_homography(court_points, image=first_frame)
    
    # Visualize homography and save to file
    if visualize_homography_flag:
        homography_vis = visualize_homography(first_frame, H)
        homography_vis_path = str(Path(output_path).with_name(f"{Path(output_path).stem}_homography.jpg"))
        cv2.imwrite(homography_vis_path, homography_vis)
        print(f"Homography visualization saved to {homography_vis_path}")
        
        # Display homography visualization
        cv2.imshow("Homography Visualization", homography_vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Process frames
    positions = []
    timestamps = []
    frame_buffer = []
    speeds = []
    
    print("Processing frames...")
    for i, frame in enumerate(tqdm(frames)):
        # Store original frame for processing
        frame_buffer.append(frame)
        timestamps.append(i / fps)
        
        if len(frame_buffer) == 3:
            # Prepare input and get prediction
            model_input = prepare_input_frames(frame_buffer, device)
            with torch.no_grad():
                heatmap = model(model_input)
            
            # Get ball position
            position = get_ball_position(heatmap)
            positions.append(position)
            
            # Draw on frame
            output_frame = frame_buffer[1].copy()
            output_frame = draw_trajectory(output_frame, positions[-10:])
            
            # Overlay court lines if requested
            if overlay_court:
                output_frame = visualize_homography(output_frame, H, thickness=1)
            
            # Compute velocities
            if len(positions) >= 3:
                velocities = compute_velocity(positions[-3:], timestamps[-3:], H)
                if velocities and velocities[-1] is not None:
                    vx, vy = velocities[-1]
                    speed = np.sqrt(vx**2 + vy**2)
                    speeds.append(speed)
                    
                    # Display current speed in mph
                    speed_mph = speed * 2.23694
                    cv2.putText(output_frame, f"Speed: {speed_mph:.1f} mph", 
                               (10, output_frame.shape[0] - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Write frame
            out.write(output_frame)
            
            # Update buffer
            frame_buffer.pop(0)
    
    # Release video writer
    out.release()
    
    # Calculate speed statistics
    filtered_speeds = filter_speeds(speeds)
    max_speed, avg_speed, std_speed = get_speed_stats(filtered_speeds)
    
    # Convert to mph and print
    max_speed_mph = max_speed * 2.23694
    avg_speed_mph = avg_speed * 2.23694
    std_speed_mph = std_speed * 2.23694
    print(f"Maximum Speed: {max_speed_mph:.1f} mph")
    print(f"Average Speed: {avg_speed_mph:.1f} mph")
    print(f"Standard Deviation: {std_speed_mph:.1f} mph")

def main():
    parser = argparse.ArgumentParser(description='TrackNet Ball Detection')
    parser.add_argument('--video', type=str, required=True,
                      help='Path to input video file')
    parser.add_argument('--model', type=str, default=str(MODEL_DIR / 'model_best.pth'),
                      help='Path to trained model weights')
    parser.add_argument('--output', type=str, required=True,
                      help='Path to output video file')
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to use (cuda/cpu)')
    parser.add_argument('--visualize-homography', action='store_true',
                      help='Visualize and save homography to verify court mapping')
    parser.add_argument('--overlay-court', action='store_true',
                      help='Overlay court lines on output video frames')
    
    args = parser.parse_args()
    
    # Use CUDA if available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU instead")
        args.device = 'cpu'
    
    process_video(
        video_path=args.video,
        model_path=args.model,
        output_path=args.output,
        device=args.device,
        visualize_homography_flag=args.visualize_homography,
        overlay_court=args.overlay_court
    )

if __name__ == '__main__':
    main() 