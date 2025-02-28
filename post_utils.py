
from scipy.ndimage import convolve
import numpy as np

import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev, interp1d

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

def smooth_and_resample_segment(segment, radii, spacing=2, smoothing=0.9):
    """
    Smooths and resamples a 3D vessel segment using a spline curve, ensuring uniform spacing.
    Uses **linear interpolation** for radius values to avoid artifacts with non-monotonic data.

    Args:
        segment (list of tuples): A single 3D centerline segment [(x, y, z)].
        radii (list of floats): Corresponding radius values for each point in the segment.
        spacing (float): Desired spacing between resampled points.
        smoothing (float): Smoothing factor for spline fitting (0 = strict, higher = smoother).

    Returns:
        tuple: (new_segment, new_radii)
            - new_segment (list of tuples): Smoothed and uniformly spaced 3D centerline segment.
            - new_radii (list of floats): Interpolated radius values for the new points.
    """
    segment = remove_duplicate_points(segment)  # Ensure uniqueness

    if len(segment) < 3:
        return segment, radii  # Ignore short segments

    segment = np.array(segment)
    radii = np.array(radii)

    # Compute total length of the segment
    distances = np.linalg.norm(np.diff(segment, axis=0), axis=1)
    total_length = np.sum(distances)

    # Estimate the number of points based on the total length and desired spacing
    num_points = max(int(total_length / spacing), len(segment))  # Ensure at least as many points as original

    # Fit a spline curve to the segment (3D coordinates)
    tck, u = splprep(segment.T, s=smoothing, k=min(3, len(segment) - 1))  # Ensure valid spline order

    # Generate new uniform u values
    new_u = np.linspace(0, 1, num_points)  # Uniform parameterization

    # Resample the curve with uniform spacing
    new_points = np.array(splev(new_u, tck)).T  # Evaluate spline for 3D points

    # Ensure `u` and `radii` have matching lengths by recomputing `u` for radii
    u_radii = np.linspace(0, 1, len(radii))

    # Use linear interpolation for non-monotonic radii
    radius_interpolator = interp1d(u_radii, radii, kind='linear', fill_value="extrapolate")
    new_radii = radius_interpolator(new_u)
    # print(f'old {np.array(radii).min()} {np.array(radii).max()}')
    # print(f'new {np.array(new_radii).min()} {np.array(new_radii).max()}')
    # exit(1)

    return [tuple(p) for p in new_points], new_radii.tolist()

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

def is_terminal(segment, neighbor_count, endpoints):
    """
    Checks if a segment is terminal (one end is an endpoint).
    
    Args:
        segment (list of tuples): The extracted vessel segment (ordered list of (y, x) points).
        neighbor_count (ndarray): Precomputed neighbor count matrix.
        endpoints (set): Set of terminal endpoints.

    Returns:
        bool, tuple: (True if terminal, terminal endpoint)
    """
    start, end = segment[0], segment[-1]

    if start in endpoints and neighbor_count[start] == 1:
        return True, start
    elif end in endpoints and neighbor_count[end] == 1:
        return True, end
    return False, None

def extract_segments(skeleton, distance_transform):
    """
    Extracts vessel centerlines as ordered segments from a skeletonized image,
    while also extracting radius information and terminal branch labels.

    Args:
        skeleton (ndarray): Binary skeletonized image.
        distance_transform (ndarray): Euclidean distance transform of the binary vessel image.

    Returns:
        tuple: (vessel_segments, radius_list, labels)
            - vessel_segments: List of ordered lists of (y, x) coordinate pairs.
            - radius_list: List of ordered lists of radius values (from distance_transform).
            - labels: List of booleans indicating if a segment is a terminal branch.
    """

    # Define 8-neighborhood kernel to find junctions
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]])

    # Compute neighbor count for each skeleton pixel
    neighbor_count = convolve(skeleton.astype(int), kernel, mode='constant', cval=0) - 10

    # Identify branch points (pixels with 3+ neighbors)
    branch_points = set(zip(*np.where((neighbor_count >= 3) & skeleton)))

    # Identify endpoints (pixels with exactly 1 neighbor)
    endpoints = set(zip(*np.where((neighbor_count == 1) & skeleton)))

    # Visited pixels
    visited = set()

    # Lists to store extracted data
    vessel_segments = []
    radius_list = []
    labels = []  # True if the segment is a terminal branch

    # Process branch points first
    for branch in branch_points:
        for neighbor in find_neighbors(branch, skeleton, visited):
            if neighbor not in visited:
                segment = traverse_segment(neighbor, skeleton, visited, branch_points, endpoints)
                
                if len(segment) > 2:
                    # Check if the segment is terminal
                    is_term, terminal_endpoint = is_terminal(segment, neighbor_count, endpoints)

                    # Ensure the last point in terminal segments is the true terminal endpoint
                    if is_term and segment[-1] != terminal_endpoint:
                        segment.reverse()  # Reverse the order if needed

                    vessel_segments.append(segment)

                    # Extract ordered radius values
                    radii = [distance_transform[y, x] for y, x in segment]
                    radius_list.append(radii)

                    # Store terminal label
                    labels.append(is_term)

    # Process remaining endpoints
    for endpoint in endpoints:
        if endpoint not in visited:
            segment = traverse_segment(endpoint, skeleton, visited, branch_points, endpoints)
            
            if len(segment) > 2:
                # Check if the segment is terminal
                is_term, terminal_endpoint = is_terminal(segment, neighbor_count, endpoints)

                # Ensure the last point is the true terminal endpoint
                if is_term and segment[-1] != terminal_endpoint:
                    segment.reverse()

                vessel_segments.append(segment)

                # Extract ordered radius values
                radii = [distance_transform[y, x] for y, x in segment]
                radius_list.append(radii)

                # Store terminal label
                labels.append(is_term)

    print(f"Total vessel segments extracted: {len(vessel_segments)}")

    return vessel_segments, radius_list, labels

import vtk
import numpy as np
import pyvista as pv
from vtk.util import numpy_support
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkFiltersCore import vtkContourFilter

def load_vti_or_image(filename, image_data=None):
    """Load 3D image from VTI file if it exists, otherwise use the given image data."""
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(filename)
    if reader.CanReadFile(filename):
        reader.Update()
        return reader.GetOutput()
    else:
        return convert_numpy_to_vtk(image_data)

def convert_numpy_to_vtk(numpy_image):
    """Convert a NumPy 3D image to a VTK image."""
    vtk_image = vtkImageData()
    dims = numpy_image.shape
    vtk_image.SetDimensions(dims[2], dims[1], dims[0])
    vtk_image.AllocateScalars(vtk.VTK_FLOAT, 1)
    flat_data = numpy_image.flatten(order='C')
    vtk_array = numpy_support.numpy_to_vtk(flat_data, deep=True, array_type=vtk.VTK_FLOAT)
    vtk_image.GetPointData().SetScalars(vtk_array)
    return vtk_image

def extract_isosurface(vtk_image, iso_value=0):
    """Extract isosurface from a 3D VTK image."""
    contour = vtkContourFilter()
    contour.SetInputData(vtk_image)
    contour.SetValue(0, iso_value)
    contour.Update()
    return contour.GetOutput()

def get_largest_connected_region(surface):
    """Extracts the largest connected component from a VTK surface."""
    connectivity_filter = vtk.vtkConnectivityFilter()
    connectivity_filter.SetInputData(surface)
    connectivity_filter.SetExtractionModeToLargestRegion()
    connectivity_filter.Update()
    return connectivity_filter.GetOutput()

def display_3d_surface(surface, terminal_segments):
    """
    Displays the extracted surface along with terminal centerlines.
    
    Args:
        surface (vtkPolyData): Extracted isosurface.
        terminal_segments (list of lists): Terminal centerlines (list of 3D point lists).
    """
    plotter = pv.Plotter()
    plotter.add_mesh(pv.wrap(surface), color="white", opacity=0.5, show_edges=True)

    # Add terminal centerlines
    for segment in terminal_segments:
        if len(segment) > 1:
            line = pv.lines_from_points(np.array(segment))
            plotter.add_mesh(line, color="red", line_width=3)

    plotter.show()