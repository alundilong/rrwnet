
from skimage.measure import label, regionprops
from scipy.ndimage import convolve
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

def find_neighbors(point, skeleton, visited):
    """Finds 8-connected neighbors of a point that are in the skeleton and not visited."""
    y, x = point
    neighbors = []
    
    # 8-connected neighbor offsets
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    
    for dy, dx in offsets:
        ny, nx = y + dy, x + dx
        if 0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1]:
            if skeleton[ny, nx] and (ny, nx) not in visited:  # Ensure it's unvisited
                neighbors.append((ny, nx))
    
    return neighbors

def traverse_segment(start, skeleton, visited, branch_points, endpoints):
    """Finds an ordered segment starting from a given point until hitting another branch or endpoint."""
    stack = [start]
    segment = []

    while stack:
        point = stack.pop()
        if point in visited:
            continue
        visited.add(point)
        segment.append(point)

        # Get unvisited neighbors
        neighbors = find_neighbors(point, skeleton, visited)

        # If the next neighbor is a branch point or endpoint, stop traversal
        if point in branch_points and point != start:
            break
        if point in endpoints:
            break

        # Continue traversal by adding all unvisited neighbors to stack
        for neighbor in neighbors:
            stack.append(neighbor)

    return segment

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
    branch_points = set(zip(*np.where((neighbor_count >= 3) & skeleton)))

    # Identify endpoints (pixels with exactly 1 neighbor)
    endpoints = set(zip(*np.where((neighbor_count == 1) & skeleton)))

    # Set of all skeleton points
    skeleton_points = set(zip(*np.where(skeleton)))

    # Visited pixels
    visited = set()

    # List of extracted vessel segments
    vessel_segments = []

    # Traverse from branch points first, exploring all possible directions
    for branch in branch_points:
        for neighbor in find_neighbors(branch, skeleton, visited):
            if neighbor not in visited:
                segment = traverse_segment(neighbor, skeleton, visited, branch_points, endpoints)
                if len(segment) > 2:  # Ensure segment has at least 2 points
                    vessel_segments.append(segment)

    # Traverse remaining unvisited endpoints
    for endpoint in endpoints:
        if endpoint not in visited:
            segment = traverse_segment(endpoint, skeleton, visited, branch_points, endpoints)
            if len(segment) > 2:
                vessel_segments.append(segment)

    print(f"Total vessel segments extracted: {len(vessel_segments)}")
    
    return vessel_segments  # List of ordered (y, x) coordinate lists