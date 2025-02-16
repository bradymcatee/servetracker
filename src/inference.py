import torch
import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

from model import TrackNetModel
from utils.video import preprocess_video
from config import INPUT_HEIGHT, INPUT_WIDTH, MODEL_DIR
from utils.homography import compute_homography, compute_velocity, filter_speeds, get_speed_stats

def load_model(model_path, device):
    """Load trained TrackNet model"""
    model = TrackNetModel().to(device)
    state_dict = torch.load(model_path, map_location=device)
    if hasattr(state_dict, 'state_dict'):
        state_dict = state_dict.state_dict()
    model.load_state_dict(state_dict)
    model.eval()
    return model

def prepare_input_frames(frames, device):
    """Prepare sequence of 3 frames for model input"""
    # Stack 3 frames together (assuming RGB format)
    stacked = np.concatenate(frames, axis=2)  # Shape: (H, W, 9)
    
    # Convert to tensor and normalize
    tensor = torch.from_numpy(stacked).permute(2, 0, 1).float()  # Shape: (9, H, W)
    tensor = tensor / 255.0
    
    # Add batch dimension
    tensor = tensor.unsqueeze(0).to(device)  # Shape: (1, 9, H, W)
    return tensor

def get_ball_position(heatmap, threshold=0.5):
    """Extract ball position from heatmap"""
    # Convert heatmap to numpy array
    heatmap = heatmap.squeeze().cpu().numpy()
    
    # Find coordinates of maximum value
    max_val = np.max(heatmap)
    if max_val < threshold:
        return None
    
    y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    return (int(x), int(y))

def draw_trajectory(frame, positions, timestamps, homography, radius=3, max_speed=70):
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
    """Interactive court point selection for service box"""
    points = {}
    point_names = [
        'service_line_left',    # Left service line intersection
        'service_line_right',   # Right service line intersection
        'service_line_center',  # Center service line intersection
        'singles_line_left',    # Left singles line at baseline
        'singles_line_right',   # Right singles line at baseline
        'center_line_bottom'    # Center line at baseline
    ]
    
    print("\nPlease select the following points in order:")
    print("1. Left service line intersection")
    print("2. Right service line intersection")
    print("3. Center service line intersection")
    print("4. Left singles line at baseline")
    print("5. Right singles line at baseline")
    print("6. Center line at baseline")
    print("\nPress 'q' to quit point selection\n")
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < len(point_names):
                point_name = point_names[len(points)]
                points[point_name] = (x, y)
                print(f"Selected {point_name}: ({x}, {y})")
                
                # Draw point
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(frame, str(len(points)), (x+10, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.imshow('Select Service Box Points', frame)
    
    # Draw guide image
    guide_frame = frame.copy()
    cv2.putText(guide_frame, "Select 6 points as shown in the instructions", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Select Service Box Points', guide_frame)
    cv2.setMouseCallback('Select Service Box Points', mouse_callback)
    
    while len(points) < len(point_names):
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    return points

def draw_court_topdown(width=800, height=800):
    """Create a top-down view of tennis court with standard dimensions"""
    # Create blank white image
    court = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Convert real-world dimensions to pixels
    court_width = 10.97  # meters
    singles_width = 8.23  # meters
    singles_length = 23.77  # meters
    service_line = 6.40  # meters from net
    
    # Calculate scale to fit court in image with margins
    margin = 0.15  # 15% margin
    scale = min(
        width * (1 - 2*margin) / court_width,
        height * (1 - 2*margin) / singles_length
    )
    
    def to_pixels(x, y):
        # Center the court in the image
        px = int(width/2 + x * scale)
        py = int(height/2 + y * scale)
        return (px, py)
    
    # Singles court dimensions
    service_line = 6.40  # meters from net
    
    # Convert court points to pixel coordinates
    # Bottom half (near baseline, server's side)
    left_singles_bottom = to_pixels(-singles_width/2, singles_length/2)
    right_singles_bottom = to_pixels(singles_width/2, singles_length/2)
    left_service_bottom = to_pixels(-singles_width/2, service_line/2)
    right_service_bottom = to_pixels(singles_width/2, service_line/2)
    center_service_bottom = to_pixels(0, service_line/2)
    center_baseline = to_pixels(0, singles_length/2)
    
    # Top half (far side)
    left_singles_top = to_pixels(-singles_width/2, -singles_length/2)
    right_singles_top = to_pixels(singles_width/2, -singles_length/2)
    left_service_top = to_pixels(-singles_width/2, -service_line/2)
    right_service_top = to_pixels(singles_width/2, -service_line/2)
    center_service_top = to_pixels(0, -service_line/2)
    center_top = to_pixels(0, -singles_length/2)
    
    # Net line points
    net_left = to_pixels(-singles_width/2, 0)
    net_right = to_pixels(singles_width/2, 0)
    
    # Draw court lines (brown color)
    color = (139, 69, 19)
    
    # Draw baselines
    cv2.line(court, left_singles_bottom, right_singles_bottom, color, 2)  # Bottom baseline
    cv2.line(court, left_singles_top, right_singles_top, color, 2)  # Top baseline
    
    # Draw singles sidelines
    cv2.line(court, left_singles_bottom, left_singles_top, color, 2)  # Left singles line
    cv2.line(court, right_singles_bottom, right_singles_top, color, 2)  # Right singles line
    
    # Draw service lines
    cv2.line(court, left_service_bottom, right_service_bottom, color, 2)  # Bottom service line
    cv2.line(court, left_service_top, right_service_top, color, 2)  # Top service line
    
    # Draw center lines
    cv2.line(court, center_service_bottom, center_baseline, color, 2)  # Bottom center line
    cv2.line(court, center_service_top, center_top, color, 2)  # Top center line
    
    # Draw net line (dashed)
    cv2.line(court, net_left, net_right, color, 1, lineType=cv2.LINE_AA)
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(court, "Net", (width//2 - 20, height//2 + 15), font, 0.5, color, 1)
    cv2.putText(court, "Server", (width//2 - 25, height - 40), font, 0.5, color, 1)
    
    return court, scale

def update_topdown_view(court_image, positions, timestamps, H, scale):
    """Draw ball trajectory on top-down court view"""
    court = court_image.copy()
    height, width = court.shape[:2]
    
    # Convert pixel positions to real-world coordinates and draw
    real_positions = []
    for pos in positions:
        if pos is not None:
            # Convert to homogeneous coordinates
            px = np.array([[pos[0], pos[1], 1]], dtype=np.float32).T
            # Apply homography
            p_real = H.dot(px)
            p_real = p_real / p_real[2]
            # Store real-world coordinates
            real_positions.append((p_real[0][0], p_real[1][0]))
    
    # Draw trajectory on court
    for i in range(len(real_positions)):
        # Convert real-world coordinates to top-down pixel coordinates
        x, y = real_positions[i]
        px = int(width/2 + x * scale)
        py = int(height/2 + y * scale)
        
        # Draw point
        cv2.circle(court, (px, py), 3, (0, 0, 255), -1)
        
        # Draw line to previous point
        if i > 0:
            prev_x, prev_y = real_positions[i-1]
            prev_px = int(width/2 + prev_x * scale)
            prev_py = int(height/2 + prev_y * scale)
            cv2.line(court, (prev_px, prev_py), (px, py), (0, 0, 255), 1)
    
    return court

def process_video(video_path, model_path, output_path, device='cuda'):
    """Process video and detect ball trajectory"""
    # Load model
    model = load_model(model_path, device)
    
    # Get video properties first
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
        
    # Get actual FPS from video
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
        fps,  # Use actual video fps
        (frames[0].shape[1], frames[0].shape[0])
    )
    
    # Create top-down court view and video writer
    court_image, scale = draw_court_topdown(800, 800)
    topdown_path = str(Path(output_path).parent / f"{Path(output_path).stem}_topdown.mp4")
    out_topdown = cv2.VideoWriter(
        topdown_path,
        fourcc,
        fps,
        (court_image.shape[1], court_image.shape[0])
    )
    
    # Get first frame for court point selection
    first_frame = frames[0].copy()
    print("Please select court points...")
    court_points = select_court_points(first_frame)
    
    # Compute homography matrix
    H = compute_homography(court_points)
    
    # Process frames
    positions = []
    timestamps = []
    frame_buffer = []
    speeds = []
    
    print("Processing frames...")
    for i, frame in enumerate(tqdm(frames)):
        frame_buffer.append(frame)
        timestamps.append(i / fps)
        
        if len(frame_buffer) == 3:
            # Prepare input
            model_input = prepare_input_frames(frame_buffer, device)
            
            # Get prediction
            with torch.no_grad():
                heatmap = model(model_input)
            
            # Get ball position
            position = get_ball_position(heatmap)
            positions.append(position)
            
            # Draw on frame
            output_frame = frame_buffer[1].copy()
            output_frame = draw_trajectory(
                output_frame,
                positions[-10:],
                timestamps[-10:],
                H,
                radius=3,
                max_speed=70  # max speed in m/s
            )
            
            # Compute velocities with improved filtering
            if len(positions) >= 3:  # Reduced window for faster response
                velocities = compute_velocity(positions[-3:], timestamps[-3:], H)
                if velocities and velocities[-1] is not None:
                    vx, vy = velocities[-1]
                    speed = np.sqrt(vx**2 + vy**2)
                    speeds.append(speed)
            
            # Update and write top-down view
            if position is not None:  # Only update when ball is detected
                topdown_frame = update_topdown_view(
                    court_image, 
                    positions[-20:],  # Show longer trail
                    timestamps[-20:], 
                    H, 
                    scale
                )
                # Add frame number or timestamp
                cv2.putText(
                    topdown_frame,
                    f"Frame: {i}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1
                )
                out_topdown.write(topdown_frame)
            else:
                out_topdown.write(court_image)  # Write base court when no ball detected
            
            # Write frame
            out.write(output_frame)
            
            # Update buffer
            frame_buffer.pop(0)
    
    # Release both video writers
    out.release()
    out_topdown.release()
    
    # Apply improved filtering and compute statistics
    filtered_speeds = filter_speeds(speeds)
    max_speed, avg_speed, std_speed = get_speed_stats(filtered_speeds)
    
    # Convert to mph and print
    max_speed_mph = max_speed * 2.23694
    print(f"Maximum Speed: {max_speed_mph:.1f} mph")
    print(f"Average Speed: {avg_speed * 2.23694:.1f} mph")

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
    
    args = parser.parse_args()
    
    # Use CUDA if available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU instead")
        args.device = 'cpu'
    
    process_video(args.video, args.model, args.output, args.device)

if __name__ == '__main__':
    main() 