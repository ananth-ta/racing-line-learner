import cv2
import numpy as np
import csv
from scipy.spatial import cKDTree
from scipy.interpolate import splprep, splev
from skimage.morphology import skeletonize

# F1TENTH YAML Data
RESOLUTION = 0.05
ORIGIN_X = -11.606540
ORIGIN_Y = -27.320793

def extract_racing_line():
    print("Loading map and computing boundaries...")
    img = cv2.imread('maps/vegas.png', cv2.IMREAD_GRAYSCALE)
    
    # 1. Threshold: Track is white (>200), walls are black
    _, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
    binary_bool = (binary == 255)
    
    # 2. Distance Transform (Distance to nearest wall)
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    
    # 3. Skeletonize to find the mathematical centerline
    print("Calculating medial axis...")
    skeleton = skeletonize(binary_bool)
    py, px = np.where(skeleton)
    points = np.column_stack((px, py))
    
    # 4. Order the points into a continuous loop (Nearest Neighbor)
    print("Ordering track coordinates (this will only take a few seconds now)...")
    ordered_points = [points[0]]
    unvisited = set(range(1, len(points)))
    tree = cKDTree(points)
    
    curr_idx = 0
    while unvisited:
        k = 10
        found = False
        
        # Dynamically expand the search radius if we hit a gap or branch
        while not found and k <= len(points):
            distances, indices = tree.query(points[curr_idx], k=k)
            
            # Ensure indices is iterable just in case k resolves weirdly
            if isinstance(indices, int): indices = [indices]
                
            for idx in indices:
                if idx in unvisited:
                    ordered_points.append(points[idx])
                    unvisited.remove(idx)
                    curr_idx = idx
                    found = True
                    break
            
            # If all k neighbors are visited, double the search radius to jump the gap!
            if not found:
                k *= 2 
                
        # Safety break if the loop is fully closed and no unvisited points remain in reach
        if not found:
            break
                
    ordered_points = np.array(ordered_points)
    
    # 5. Fit a mathematical B-Spline to smooth the line
    print("Applying B-Spline and calculating racing apexes...")
    # Reduce points slightly for a smoother spline
    sampled_points = ordered_points[::5] 
    
    # s is the smoothing factor. Adjust if the line is too jagged.
    tck, u = splprep([sampled_points[:,0], sampled_points[:,1]], s=200, per=True)
    
    # Evaluate the spline at 150 evenly spaced breadcrumbs
    u_new = np.linspace(0, 1, 150)
    smooth_x, smooth_y = splev(u_new, tck)
    
    # Evaluate first and second derivatives for curvature
    dx, dy = splev(u_new, tck, der=1)
    ddx, ddy = splev(u_new, tck, der=2)
    
    # 6. Translate to real-world F1TENTH meters and apply Apex Shift
    waypoints = []
    for i in range(len(smooth_x)):
        px, py = smooth_x[i], smooth_y[i]
        
        # Calculate normal vector (perpendicular to track)
        norm = np.sqrt(dx[i]**2 + dy[i]**2)
        nx, ny = -dy[i]/norm, dx[i]/norm
        
        # Calculate Curvature (kappa)
        curvature = (dx[i]*ddy[i] - dy[i]*ddx[i]) / (norm**3)
        
        # Apex Shift: Move inside the turn based on curvature
        # We limit the shift so it doesn't exceed 80% of the track width
        track_width_pixels = dist_transform[int(py), int(px)]
        shift_amount = np.clip(curvature * 200, -track_width_pixels * 0.8, track_width_pixels * 0.8)
        
        opt_px = px + (nx * shift_amount)
        opt_py = py + (ny * shift_amount)
        
        # Convert to ROS Meters using YAML math
        ros_x = (opt_px * RESOLUTION) + ORIGIN_X
        ros_y = ((img.shape[0] - opt_py) * RESOLUTION) + ORIGIN_Y
        waypoints.append((ros_x, ros_y))
        
    # Save the final trajectory
    csv_filename = 'maps/vegas_optimal_line.csv'
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y"])
        writer.writerows(waypoints)
        
    print(f"Success! {len(waypoints)} mathematically optimized breadcrumbs saved to {csv_filename}")

if __name__ == '__main__':
    extract_racing_line()