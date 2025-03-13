import numpy as np
import cv2
from scipy.signal import savgol_filter
from pathlib import Path

def get_court_coordinates():
    """Return real-world coordinates of tennis court service box points (in meters)"""
    # Standard tennis court dimensions
    COURT_WIDTH = 10.97  # meters
    SINGLES_WIDTH = 8.23  # meters
    SERVICE_LINE_NEAR = 6.40  # meters from net
    SERVICE_LINE_FAR = 12.80  # meters from net (6.40 + 6.40)
    NET_LINE = 0.0  # meters (reference point)
    BASELINE = 11.89  # meters from net
    
    # Define key points in real-world coordinates (meters)
    # Using both near and far court service lines for better homography
    return {
        # Near court service line points
        'near_service_left': ((COURT_WIDTH - SINGLES_WIDTH)/2, SERVICE_LINE_NEAR),
        'near_service_right': ((COURT_WIDTH + SINGLES_WIDTH)/2, SERVICE_LINE_NEAR),
        'near_service_center': (COURT_WIDTH/2, SERVICE_LINE_NEAR),
        
        # Far court service line points
        'far_service_left': ((COURT_WIDTH - SINGLES_WIDTH)/2, -SERVICE_LINE_NEAR),
        'far_service_right': ((COURT_WIDTH + SINGLES_WIDTH)/2, -SERVICE_LINE_NEAR),
        'far_service_center': (COURT_WIDTH/2, -SERVICE_LINE_NEAR),
        
        # Far baseline points
        'far_baseline_left': ((COURT_WIDTH - SINGLES_WIDTH)/2, -BASELINE),
        'far_baseline_right': ((COURT_WIDTH + SINGLES_WIDTH)/2, -BASELINE),
        'far_baseline_center': (COURT_WIDTH/2, -BASELINE),
        
        # Net line points
        'net_left': ((COURT_WIDTH - SINGLES_WIDTH)/2, NET_LINE),
        'net_right': ((COURT_WIDTH + SINGLES_WIDTH)/2, NET_LINE),
        'net_center': (COURT_WIDTH/2, NET_LINE),
        
        # T-junction points (where service line meets singles sideline)
        'near_t_left': ((COURT_WIDTH - SINGLES_WIDTH)/2, SERVICE_LINE_NEAR),
        'near_t_right': ((COURT_WIDTH + SINGLES_WIDTH)/2, SERVICE_LINE_NEAR),
        'far_t_left': ((COURT_WIDTH - SINGLES_WIDTH)/2, -SERVICE_LINE_NEAR),
        'far_t_right': ((COURT_WIDTH + SINGLES_WIDTH)/2, -SERVICE_LINE_NEAR)
    }

def visualize_homography(image, homography, thickness=2):
    """
    Visualize the homography by drawing tennis court lines on the image.
    
    Args:
        image: The input image to draw on
        homography: The homography matrix (3x3)
        thickness: Line thickness
        
    Returns:
        Image with court lines drawn
    """
    # Create a copy of the image to draw on
    vis_image = image.copy()
    
    # Get standard tennis court dimensions
    COURT_WIDTH = 10.97  # meters
    SINGLES_WIDTH = 8.23  # meters
    SERVICE_LINE_NEAR = 6.40  # meters from net
    SERVICE_LINE_FAR = 12.80  # meters from net (6.40 + 6.40)
    NET_LINE = 0.0  # meters (reference point)
    BASELINE = 11.89  # meters from net
    
    # Define court lines in real-world coordinates
    court_lines = [
        # Singles sidelines
        [((COURT_WIDTH - SINGLES_WIDTH)/2, -BASELINE), ((COURT_WIDTH - SINGLES_WIDTH)/2, BASELINE)],
        [((COURT_WIDTH + SINGLES_WIDTH)/2, -BASELINE), ((COURT_WIDTH + SINGLES_WIDTH)/2, BASELINE)],
        
        # Baselines
        [((COURT_WIDTH - SINGLES_WIDTH)/2, BASELINE), ((COURT_WIDTH + SINGLES_WIDTH)/2, BASELINE)],
        [((COURT_WIDTH - SINGLES_WIDTH)/2, -BASELINE), ((COURT_WIDTH + SINGLES_WIDTH)/2, -BASELINE)],
        
        # Service lines
        [((COURT_WIDTH - SINGLES_WIDTH)/2, SERVICE_LINE_NEAR), ((COURT_WIDTH + SINGLES_WIDTH)/2, SERVICE_LINE_NEAR)],
        [((COURT_WIDTH - SINGLES_WIDTH)/2, -SERVICE_LINE_NEAR), ((COURT_WIDTH + SINGLES_WIDTH)/2, -SERVICE_LINE_NEAR)],
        
        # Center service line
        [(COURT_WIDTH/2, SERVICE_LINE_NEAR), (COURT_WIDTH/2, -SERVICE_LINE_NEAR)],
        
        # Net
        [((COURT_WIDTH - SINGLES_WIDTH)/2, NET_LINE), ((COURT_WIDTH + SINGLES_WIDTH)/2, NET_LINE)],
        
        # Additional lines for better visualization
        # Center mark on baselines
        [(COURT_WIDTH/2, BASELINE), (COURT_WIDTH/2, BASELINE - 0.2)],
        [(COURT_WIDTH/2, -BASELINE), (COURT_WIDTH/2, -BASELINE + 0.2)],
        
        # Center mark on net
        [(COURT_WIDTH/2, NET_LINE - 0.1), (COURT_WIDTH/2, NET_LINE + 0.1)]
    ]
    
    # Define colors for different court lines
    colors = {
        'singles_sideline': (0, 0, 255),    # Red
        'baseline': (255, 0, 0),            # Blue
        'service_line': (0, 255, 0),        # Green
        'center_service': (255, 255, 0),    # Cyan
        'net': (255, 0, 255),               # Magenta
        'center_mark': (255, 255, 255)      # White
    }
    
    # Inverse homography to map from real-world to image coordinates
    H_inv = np.linalg.inv(homography)
    
    # Draw each court line
    for i, line in enumerate(court_lines):
        # Get line points in real-world coordinates
        p1_real = np.array([line[0][0], line[0][1], 1])
        p2_real = np.array([line[1][0], line[1][1], 1])
        
        # Transform to image coordinates
        p1_img = H_inv.dot(p1_real)
        p2_img = H_inv.dot(p2_real)
        
        # Normalize homogeneous coordinates
        p1_img = (int(p1_img[0] / p1_img[2]), int(p1_img[1] / p1_img[2]))
        p2_img = (int(p2_img[0] / p2_img[2]), int(p2_img[1] / p2_img[2]))
        
        # Choose color based on line type
        if i < 2:
            color = colors['singles_sideline']
        elif i < 4:
            color = colors['baseline']
        elif i < 6:
            color = colors['service_line']
        elif i < 7:
            color = colors['center_service']
        elif i < 8:
            color = colors['net']
        else:
            color = colors['center_mark']
        
        # Draw the line
        cv2.line(vis_image, p1_img, p2_img, color, thickness)
    
    # Add a legend
    legend_y = 30
    for name, color in colors.items():
        cv2.putText(vis_image, name.replace('_', ' ').title(), (10, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        legend_y += 25
    
    return vis_image

def filter_speeds(speeds, window_length=9, poly_order=2):
    """Apply Savitzky-Golay filtering to smooth speed measurements"""
    if len(speeds) < window_length:
        return speeds
    
    # Convert to numpy array and filter out None values
    valid_speeds = np.array([s for s in speeds if s is not None])
    
    # Apply Savitzky-Golay filter if we have enough points
    if len(valid_speeds) >= window_length:
        filtered = savgol_filter(valid_speeds, window_length, poly_order)
        return filtered.tolist()
    
    return valid_speeds.tolist()

def compute_velocity(positions, timestamps, homography):
    """Compute velocity between consecutive positions"""
    if len(positions) < 2:
        return []
    
    velocities = []
    
    for i in range(len(positions)-1):
        if positions[i] is None or positions[i+1] is None:
            velocities.append(None)
            continue
            
        # Convert positions to real-world coordinates
        p1 = np.array([positions[i][0], positions[i][1], 1])
        p2 = np.array([positions[i+1][0], positions[i+1][1], 1])
        
        p1_real = homography.dot(p1)
        p2_real = homography.dot(p2)
        
        p1_real = p1_real[:2] / p1_real[2]
        p2_real = p2_real[:2] / p2_real[2]
        
        dt = timestamps[i+1] - timestamps[i]
        
        # Compute velocity
        displacement = p2_real - p1_real
        velocity = displacement / dt
        
        # Apply reasonable bounds (70 m/s is ~155 mph)
        speed = np.sqrt(np.sum(velocity**2))
        if speed < 70:
            velocities.append(velocity)
        else:
            velocities.append(None)
    
    return velocities

def refine_points(image_points, image=None):
    """Refine the selected points to subpixel accuracy if possible"""
    refined_points = {}
    
    # If we have an image, use cornerSubPix for refinement
    if image is not None and isinstance(image, np.ndarray):
        gray = None
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        for key, point in image_points.items():
            # Convert to numpy array of points
            point_arr = np.array([point], dtype=np.float32)
            
            # Refine to subpixel accuracy
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            refined = cv2.cornerSubPix(gray, point_arr, (11, 11), (-1, -1), criteria)
            
            # Store refined point
            refined_points[key] = (float(refined[0][0]), float(refined[0][1]))
    else:
        # If no image is provided, just convert to float for better precision
        for key, point in image_points.items():
            refined_points[key] = (float(point[0]), float(point[1]))
    
    return refined_points

def compute_homography(image_points, image=None):
    """Compute homography matrix between image points and real-world coordinates"""
    court_points = get_court_coordinates()
    
    # Refine points to subpixel accuracy if possible
    refined_points = refine_points(image_points, image)
    
    # Get the common keys between image points and court points
    common_keys = [k for k in court_points.keys() if k in refined_points]
    
    if len(common_keys) < 4:
        raise ValueError("At least 4 corresponding points are needed to compute homography")
    
    # Convert dictionary points to arrays, using only the points that were selected
    src_pts = np.float32([refined_points[k] for k in common_keys])
    dst_pts = np.float32([court_points[k] for k in common_keys])
    
    # Print how many points are being used for homography
    print(f"Computing homography using {len(common_keys)} points")
    
    # Compute homography matrix with improved RANSAC parameters
    # Increase iterations and use a stricter threshold for better accuracy
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0, maxIters=2000, confidence=0.995)
    
    # Basic error checking
    if H is None:
        raise ValueError("Could not compute homography matrix")
    
    # Refine homography using Levenberg-Marquardt optimization
    H = refine_homography(src_pts, dst_pts, H, mask)
    
    # Validate homography quality
    error = estimate_homography_error(src_pts, dst_pts, H)
    if error > 0.5:  # Error threshold in meters
        print(f"Warning: Homography error is high ({error:.2f} meters). Court point selection may be inaccurate.")
    else:
        print(f"Homography error: {error:.2f} meters")
    
    return H

def refine_homography(src_pts, dst_pts, H, mask=None):
    """Refine homography using Levenberg-Marquardt optimization"""
    if mask is not None:
        # Use only inliers for refinement
        inlier_indices = np.where(mask.ravel() == 1)[0]
        if len(inlier_indices) >= 4:  # Need at least 4 points for homography
            src_inliers = src_pts[inlier_indices]
            dst_inliers = dst_pts[inlier_indices]
            # Use method=1 for Levenberg-Marquardt optimization
            H_refined, _ = cv2.findHomography(src_inliers, dst_inliers, method=cv2.LMEDS)
            if H_refined is not None:
                return H_refined
    
    return H

def estimate_homography_error(src_pts, dst_pts, H):
    """Estimate the error of the homography transformation in meters"""
    if len(src_pts) == 0:
        return float('inf')
    
    total_error = 0.0
    num_points = len(src_pts)
    
    for i in range(num_points):
        # Transform source point using homography
        p = np.array([src_pts[i][0], src_pts[i][1], 1.0])
        p_transformed = H.dot(p)
        p_transformed = p_transformed[:2] / p_transformed[2]
        
        # Calculate Euclidean distance to destination point
        error = np.sqrt(np.sum((p_transformed - dst_pts[i])**2))
        total_error += error
    
    # Return average error in meters
    return total_error / num_points

def get_speed_stats(speeds):
    """Compute speed statistics"""
    if not speeds:
        return 0.0, 0.0, 0.0
        
    # Remove None values
    valid_speeds = np.array([s for s in speeds if s is not None])
    if len(valid_speeds) == 0:
        return 0.0, 0.0, 0.0
    
    # Calculate simple statistics
    max_speed = np.max(valid_speeds)
    avg_speed = np.mean(valid_speeds)
    std_speed = np.std(valid_speeds)
    
    return max_speed, avg_speed, std_speed 