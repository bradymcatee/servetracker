import cv2
from pathlib import Path

def preprocess_video(video_path, output_path=None, target_fps=30):
    """Preprocess video for ball tracking. Handles both iPhone and YouTube videos."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None
    
    # Get actual FPS from video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        print("Warning: FPS could not be determined, defaulting to 30.")
        fps = target_fps  # Fallback to target FPS
    
    print(f"Video FPS: {fps}")  # Log the FPS for debugging
    
    # Get original video properties
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate resize dimensions (maintain aspect ratio)
    scale = min(640/orig_width, 360/orig_height)
    new_width = int(orig_width * scale)
    new_height = int(orig_height * scale)
    
    if output_path:
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            str(output_path), 
            fourcc, 
            fps,
            (new_width, new_height)
        )
    
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize maintaining aspect ratio
        frame = cv2.resize(frame, (new_width, new_height))
        
        if output_path:
            out.write(frame)
        frames.append(frame)
    
    cap.release()
    if output_path:
        out.release()
        
    return frames 