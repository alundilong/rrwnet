
from scipy.ndimage import convolve
import numpy as np

import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

def remove_duplicate_points(segment, max_distance = 2):
    """
    Removes duplicate points from a 3D segment while preserving order.
    
    Args:
        segment (list of tuples): A single 3D vessel segment [(x, y, z)].
    
    Returns:
        list of tuples: Segment without duplicate points, maintaining order.
    """
    seen = set()
    unique_segment = []
    
    for point in segment:
        if point not in seen:
            seen.add(point)
            unique_segment.append(point)

    return unique_segment

def smooth_and_resample_segment(segment, spacing=2, smoothing=0.9):
    """
    Smooths and resamples a 3D vessel segment using a spline curve, ensuring uniform spacing.

    Args:
        segment (list of tuples): A single 3D centerline segment [(x, y, z)].
        spacing (float): Desired spacing between resampled points.
        smoothing (float): Smoothing factor for spline fitting (0 = strict, higher = smoother).

    Returns:
        list of tuples: Smoothed and uniformly spaced 3D centerline segment.
    """
    segment = remove_duplicate_points(segment)  # Ensure uniqueness

    if len(segment) < 3:
        return segment  # Ignore short segments

    segment = np.array(segment)

    # Compute total length of the segment
    distances = np.linalg.norm(np.diff(segment, axis=0), axis=1)
    total_length = np.sum(distances)

    # Estimate the number of points based on the total length and desired spacing
    num_points = max(int(total_length / spacing), len(segment))  # Ensure at least as many points as original

    # Fit a spline curve to the segment
    tck, u = splprep(segment.T, s=smoothing, k=min(3, len(segment) - 1))  # Ensure valid spline order

    # Resample the curve with uniform spacing
    new_u = np.linspace(0, 1, num_points)  # Uniform parameterization
    new_points = np.array(splev(new_u, tck)).T  # Evaluate spline

    return [tuple(p) for p in new_points]

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

    stack = []
    segment = []

    # Get all unvisited neighbors of the start point
    neighbors = find_neighbors(start, skeleton, visited)

    # If there are multiple neighbors, choose only **one** to begin traversal
    if neighbors:
        stack.append(neighbors[0])  # Always follow only one path first

    visited.add(start)
    segment.append(start)

    while stack:
        point = stack.pop()
        if point in visited:
            continue
        visited.add(point)
        segment.append(point)

        # Get unvisited neighbors
        neighbors = find_neighbors(point, skeleton, visited)

        # Stop if reaching a branch point (excluding the start point)
        if point in branch_points and point != start:
            break
        if point in endpoints:
            break

        # Append only **one** next neighbor to continue in a single direction
        for nid in neighbors:
            stack.append(nid)

    return segment

def extract_segments(skeleton, distance_transform):
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

    # Visited pixels
    visited = set()

    # List of extracted vessel segments
    vessel_segments = []
    radius_list = []       # List of radius values per segment
    labels = []            # True if the segment originates from an endpoint

    # Traverse from branch points first, exploring all possible directions
    for branch in branch_points:
        for neighbor in find_neighbors(branch, skeleton, visited):
            if neighbor not in visited:
                segment = traverse_segment(neighbor, skeleton, visited, branch_points, endpoints)
                if len(segment) > 2:  # Ensure segment has at least 2 points
                    vessel_segments.append(segment)
                    
                    # Extract corresponding radius values
                    radii = [distance_transform[y, x] for y, x in segment]
                    radius_list.append(radii)

                    # Label as False (not starting from an endpoint)
                    labels.append(False)

    # Traverse remaining unvisited endpoints
    for endpoint in endpoints:
        if endpoint not in visited:
            segment = traverse_segment(endpoint, skeleton, visited, branch_points, endpoints)
            if len(segment) > 2:
                vessel_segments.append(segment)

                # Extract corresponding radius values
                radii = [distance_transform[y, x] for y, x in segment]
                radius_list.append(radii)

                # Label as True (originated from an endpoint)
                labels.append(True)
    
    print(f"Total vessel segments extracted: {len(vessel_segments)}")
    
    return vessel_segments , radius_list, labels  # Return all extracted data