import cv2
import pandas as pd
import numpy as np

# F1TENTH YAML Data
RESOLUTION = 0.05
ORIGIN_X = -11.606540
ORIGIN_Y = -27.320793

def annotate_map():
    print("Loading image and CSV data...")
    # Load the map image in color so we can draw bright red/green lines
    img = cv2.imread('maps/vegas.png')
    
    if img is None:
        print("Error: Could not find vegas.png. Make sure it is in the same folder!")
        return
        
    height, width, _ = img.shape
    
    # Load the optimized breadcrumbs
    try:
        df = pd.read_csv('maps/vegas_optimal_line.csv')
    except FileNotFoundError:
        print("Error: Could not find vegas_optimal_line.csv.")
        return

    points = []
    
    print("Reversing YAML math (Meters -> Pixels)...")
    for index, row in df.iterrows():
        ros_x = row['x']
        ros_y = row['y']
        
        # Reverse the math to get exact pixel coordinates
        px = int((ros_x - ORIGIN_X) / RESOLUTION)
        py = int(height - ((ros_y - ORIGIN_Y) / RESOLUTION))
        
        points.append((px, py))
        
        # Draw a solid red dot for the specific waypoint
        cv2.circle(img, (px, py), radius=3, color=(0, 0, 255), thickness=-1)
        
        # Label the dot sequentially
        cv2.putText(img, str(index), (px + 5, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    print("Drawing continuous racing line...")
    # Draw a green line connecting the waypoints to visualize the exact path
    for i in range(len(points) - 1):
        cv2.line(img, points[i], points[i+1], color=(0, 255, 0), thickness=2)
        
    # Connect the final point back to the starting point to close the track loop
    if len(points) > 1:
        cv2.line(img, points[-1], points[0], color=(0, 255, 0), thickness=2)

    # Save the annotated image
    output_filename = 'maps/vegas_annotated.png'
    cv2.imwrite(output_filename, img)
    print(f"Success! Visual verification saved to {output_filename}")

if __name__ == '__main__':
    annotate_map()