
from skimage.measure import label, regionprops
from scipy.ndimage import convolve
import numpy as np

def order_segment(segment):
    """Orders a segment by sorting based on Euclidean distance from the first point."""
    if not segment:
        return segment
    
    segment = np.array(segment)
    distances = np.linalg.norm(segment - segment[0], axis=1)
    ordered_indices = np.argsort(distances)
    
    return segment[ordered_indices].tolist()

def extract_segments(skeleton):
    """
    Extracts vessel centerlines as ordered segments from a skeletonized image.
    
    Args:
        skeleton (ndarray): Binary skeletonized image.

    Returns:
        list of list of tuples: Each segment is a list of (y, x) coordinate pairs.
    """

    # Define 8-neighborhood kernel to find junctions
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]])

    # Count neighbors of each skeleton pixel
    neighbor_count = convolve(skeleton.astype(int), kernel, mode='constant', cval=0) - 10

    # Identify branch points (pixels with 3+ neighbors)
    branch_points = (neighbor_count >= 3) & skeleton

    # Label connected components
    labeled_skeleton, num_features = label(skeleton, return_num=True, connectivity=2)

    # Extract vessel segments as lists of (y, x) coordinate pairs
    vessel_segments = []
    for region in regionprops(labeled_skeleton):
        coords = np.array(region.coords)  # Extract (row, col) points

        # Split segments at branch points
        split_segments = []
        current_segment = []

        for point in coords:
            if branch_points[tuple(point)]:
                if current_segment:
                    split_segments.append(current_segment)
                    current_segment = []
            current_segment.append(tuple(point))  # Store (y, x) as tuple
        
        if current_segment:
            split_segments.append(current_segment)

        vessel_segments.extend(split_segments)

    # Order all segments
    ordered_segments = [order_segment(seg) for seg in vessel_segments]

    return ordered_segments  # List of ordered lists of (y, x) pairs