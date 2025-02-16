import numpy as np
import cv2
from scipy.signal import savgol_filter

def get_court_coordinates():
    """Return real-world coordinates of tennis court service box points (in meters)"""
    # Standard tennis court dimensions
    COURT_WIDTH = 10.97  # meters
    SINGLES_WIDTH = 8.23  # meters
    SERVICE_LINE = 6.40  # meters from net
    CENTER_LINE = SINGLES_WIDTH / 2  # 4.115 meters
    
    # Define key points in real-world coordinates (meters)
    # Using only service box points
    court_points = {
        'service_line_left': ((COURT_WIDTH - SINGLES_WIDTH)/2, SERVICE_LINE),
        'service_line_right': ((COURT_WIDTH + SINGLES_WIDTH)/2, SERVICE_LINE),
        'service_line_center': ((COURT_WIDTH)/2, SERVICE_LINE),
        'singles_line_left': ((COURT_WIDTH - SINGLES_WIDTH)/2, 0),
        'singles_line_right': ((COURT_WIDTH + SINGLES_WIDTH)/2, 0),
        'center_line_bottom': (COURT_WIDTH/2, 0)
    }
    
    return court_points

class KalmanSpeedFilter:
    """Kalman filter for speed estimation"""
    def __init__(self):
        # State: [speed, acceleration]
        self.state = np.array([0.0, 0.0])
        self.P = np.eye(2) * 100  # Initial uncertainty
        # Process noise
        self.Q = np.array([[0.1, 0],
                          [0, 0.1]])
        # Measurement noise
        self.R = np.array([[10.0]])
        
    def predict(self, dt):
        # State transition matrix
        F = np.array([[1, dt],
                     [0, 1]])
        # Predict state
        self.state = F.dot(self.state)
        # Predict covariance
        self.P = F.dot(self.P).dot(F.T) + self.Q
        
        return self.state[0]
    
    def update(self, measurement):
        if measurement is None:
            return self.state[0]
            
        # Measurement matrix
        H = np.array([[1.0, 0.0]])
        
        # Kalman gain
        S = H.dot(self.P).dot(H.T) + self.R
        K = self.P.dot(H.T) * (1.0/S)
        
        # Update state
        y = measurement - H.dot(self.state)
        self.state += K.flatten() * y
        
        # Update covariance
        self.P = (np.eye(2) - K.dot(H)).dot(self.P)
        
        return self.state[0]

def filter_speeds(speeds, window_length=7, poly_order=2):
    """Apply combined Kalman and Savitzky-Golay filtering"""
    if len(speeds) < window_length:
        return speeds
        
    # First apply Kalman filtering
    kalman = KalmanSpeedFilter()
    filtered_speeds = []
    
    for speed in speeds:
        if speed is not None:
            # Predict and update
            kalman.predict(1.0)  # Assume 1.0 time step
            filtered_speed = kalman.update(speed)
            filtered_speeds.append(filtered_speed)
        else:
            # Only predict when no measurement
            filtered_speed = kalman.predict(1.0)
            filtered_speeds.append(filtered_speed)
    
    # Then apply Savitzky-Golay for additional smoothing
    if len(filtered_speeds) >= window_length:
        filtered_speeds = savgol_filter(filtered_speeds, window_length, poly_order)
    
    return filtered_speeds.tolist()

def compute_velocity(positions, timestamps, homography, fps=None):
    """Improved velocity computation with Kalman filtering"""
    if len(positions) < 2:
        return []
    
    velocities = []
    speeds = []
    kalman_x = KalmanSpeedFilter()
    kalman_y = KalmanSpeedFilter()
    
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
        
        # Compute raw velocity
        displacement = p2_real - p1_real
        raw_velocity = displacement / dt
        
        # Filter velocity components
        kalman_x.predict(dt)
        kalman_y.predict(dt)
        
        vx = kalman_x.update(raw_velocity[0])
        vy = kalman_y.update(raw_velocity[1])
        
        filtered_velocity = np.array([vx, vy])
        speed = np.sqrt(np.sum(filtered_velocity**2))
        
        # Apply reasonable bounds
        if speed < 70:  # max speed in m/s
            velocities.append(filtered_velocity)
            speeds.append(speed)
        else:
            velocities.append(None)
    
    return velocities

def compute_homography(image_points, min_points=4):
    """Improved homography computation with RANSAC and error checking"""
    court_points = get_court_coordinates()
    
    # Convert dictionary points to arrays
    src_pts = np.float32([image_points[k] for k in court_points.keys()])
    dst_pts = np.float32([court_points[k] for k in court_points.keys()])
    
    # Compute homography matrix with RANSAC
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    # Check if homography is valid
    if H is None:
        raise ValueError("Could not compute homography matrix")
    
    # Check if homography is well-conditioned
    if np.linalg.cond(H) > 1e15:
        raise ValueError("Homography matrix is poorly conditioned")
        
    # Compute reprojection error
    error = compute_reprojection_error(src_pts, dst_pts, H)
    if error > 0.1:  # threshold in meters
        print(f"Warning: Large reprojection error: {error:.3f} meters")
    
    return H

def compute_reprojection_error(src_pts, dst_pts, H):
    """Compute average reprojection error"""
    # Transform source points
    src_transformed = cv2.perspectiveTransform(
        src_pts.reshape(-1, 1, 2), 
        H
    ).reshape(-1, 2)
    
    # Compute error
    errors = np.sqrt(np.sum((dst_pts - src_transformed) ** 2, axis=1))
    return np.mean(errors)

def get_speed_stats(speeds):
    """Compute reliable speed statistics using robust methods"""
    if not speeds:
        return 0.0, 0.0, 0.0
        
    # Remove None values
    valid_speeds = np.array([s for s in speeds if s is not None])
    if len(valid_speeds) == 0:
        return 0.0, 0.0, 0.0
    
    # Calculate median and MAD (Median Absolute Deviation)
    median = np.median(valid_speeds)
    mad = np.median(np.abs(valid_speeds - median))
    
    # Use MAD for outlier detection (more robust than standard deviation)
    modified_z_scores = 0.6745 * (valid_speeds - median) / mad
    filtered_speeds = valid_speeds[np.abs(modified_z_scores) < 3.5]
    
    if len(filtered_speeds) == 0:
        return 0.0, 0.0, 0.0
    
    # Use robust statistics for final calculations
    max_speed = np.max(filtered_speeds)
    median_speed = np.median(filtered_speeds)
    iqr = np.percentile(filtered_speeds, 75) - np.percentile(filtered_speeds, 25)
    
    return max_speed, median_speed, iqr 