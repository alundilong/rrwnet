import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import distance_transform_edt

import sys
import os

from post_utils import extract_segments, smooth_and_resample_segment
from post_utils import display_3d_surface,extract_isosurface,load_vti_or_image, get_largest_connected_region

# Get the parent directory of the current script (assuming project-root is the common parent)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add level set package to the Python path
sys.path.append(os.path.join(parent_dir, 'levelset\\Level-Set\\lv_set\\'))
sys.path.append(os.path.join(parent_dir, 'coronary'))

from CenterlineDialatorModule import CenterlineDialatorModule
from FastMarchingModule import FastMarchingFilter
from ImageForestingTransformModule import AlgorithmType
from lv_set.save_image import dump_image_to_vtk

# Load the image
image_path = "predictions/01.png"
image = cv2.imread(image_path)

# Set channel 1 (Green) and channel 2 (Red) to zero
# zerorize channel 1 and 2
image[:, :, 0] = 0  # Green Channel
image[:, :, 1] = 0  # Red Channel Artery
# image[:, :, 2] = 0  # Red Channel Vein

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

# Apply Otsu's thresholding to binarize vessels
_, thresh = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Convert thresholded image to binary format (needed for skeletonization)
binary_vessels = thresh > 0

# Apply skeletonization to extract vessel centerlines
skeleton = skeletonize(binary_vessels)

# Compute Euclidean Distance Transform (EDT)
distance_transform = distance_transform_edt(binary_vessels)

ordered_segments, radii, terminal = extract_segments(skeleton,distance_transform)

# Print number of extracted centerline segments
print(f"Extracted {len(ordered_segments)} vessel centerline segments.")

# Visualization
plt.figure(figsize=(6, 6))
plt.imshow(skeleton, cmap='gray')
for idx, segment in enumerate(ordered_segments):
    segment = np.array(segment)
    plt.plot(segment[:, 1], segment[:, 0], linewidth=0.5)  # Draw each segment
    # plt.scatter(segment[:, 1], segment[:, 0], s=1)  # Draw each segment
plt.title("Extracted Ordered Vessel Centerlines")
plt.axis("off")
plt.show()

# exit(1)


# Display results
fig, ax = plt.subplots(1, 5, figsize=(12, 6))
ax[0].imshow(thresh, cmap="gray")
ax[0].set_title("Thresholded Vessel Segmentation")
ax[0].axis("off")

ax[1].imshow(binary_vessels, cmap="gray")
ax[1].set_title("Binary Vessel Segmentation")
ax[1].axis("off")

ax[2].imshow(skeleton, cmap="gray")
ax[2].set_title("True Vessel Centerlines (Skeletonized)")
ax[2].axis("off")

ax[3].imshow(image_gray, cmap="gray")
ax[3].set_title("Gray Image")
ax[3].axis("off")

ax[4].imshow(distance_transform, cmap="gray")
ax[4].set_title("Distance Image")
ax[4].axis("off")

plt.show()

# Get image dimensions
height, width = skeleton.shape

# Compute the sphere's center and radius
cx, cy = width // 2, height // 2  # Center of the sphere
radius = max(width, height) // 2  # Sphere radius

sphere_points = []
sphere_points_radius = []
seg_sphere_points = []
seg_sphere_points_radius = []

for seg in ordered_segments:
    seg_points = []
    seg_radius = []

    for y, x in seg:

        # Normalize coordinates to the range [-1, 1] relative to the center
        nx = (x - cx) / radius
        ny = (y - cy) / radius

        # Compute the z-coordinate on the hemisphere using inverse spherical projection
        if nx**2 + ny**2 <= 1:  # Ensure points are within the unit circle
            nz = np.sqrt(1 - (nx**2 + ny**2))  # z is positive for upper hemisphere
            point = (nx * radius, ny * radius, nz * radius)
            sphere_points.append(point)
            seg_points.append(point)
            sphere_points_radius.append(distance_transform[y,x])
            seg_radius.append(distance_transform[y,x])

    seg_points, seg_radius = smooth_and_resample_segment(seg_points, seg_radius, spacing=2, smoothing=0.5)

    seg_sphere_points.append(seg_points)
    seg_sphere_points_radius.append(seg_radius)


# Convert to numpy array for easier processing
sphere_points = np.array(sphere_points)
sphere_points_radius = np.array(sphere_points_radius)
print(sphere_points_radius.min(), sphere_points_radius.max())

# 3D Plotting
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(sphere_points[:, 0], sphere_points[:, 1], sphere_points[:, 2], s=1, color='b')

# Set 3D plot labels and limits
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Mapped Vessel Centerline Points on Hemisphere")
ax.set_xlim([-radius, radius])
ax.set_ylim([-radius, radius])
ax.set_zlim([0, radius])  # Only positive z values

plt.show()
# exit(1)

extend = (np.max(sphere_points, axis=0) - np.min(sphere_points, axis=0))*0.1

# Determine the bounding box of the 3D points
min_bound = np.min(sphere_points, axis=0) - extend
max_bound = np.max(sphere_points, axis=0) + extend

# Define voxel grid resolution
voxel_size = radius / 200

# Compute grid dimensions based on the bounding box
grid_x = np.arange(min_bound[0], max_bound[0], voxel_size)
grid_y = np.arange(min_bound[1], max_bound[1], voxel_size)
grid_z = np.arange(min_bound[2], max_bound[2], voxel_size)

# Create an empty 3D volume
dimensions = (len(grid_x), len(grid_y), len(grid_z))
spacing = (1.0, 1.0, 1.0)

# Initialize the voxel ID array and radius array
voxel_id_array = [] 
radius_array = []

# Define a default radius for each point
max_radius = sphere_points_radius.max()

# Define the output directory in /tmp/
output_dir = "tmp/"
os.makedirs(output_dir, exist_ok=True)

# Assign each 3D point to the closest voxel and store its radius
for idx, seg in enumerate(seg_sphere_points):

    obj_filename = os.path.join(output_dir, f"segment_{idx}.obj")
    seg_3d_points = []
    with open(obj_filename, "w") as f:
        for point_idx, (px, py, pz) in enumerate(seg):
            voxel_radius = -seg_sphere_points_radius[idx][point_idx]  # Assign a default radius (can be customized per point)

            # Convert world coordinates to voxel indices
            idx_x = np.searchsorted(grid_x, px)
            idx_y = np.searchsorted(grid_y, py)
            idx_z = np.searchsorted(grid_z, pz)

            # Ensure indices are within bounds
            if 0 <= idx_x < dimensions[0] and 0 <= idx_y < dimensions[1] and 0 <= idx_z < dimensions[2]:
                point3d = (idx_x,idx_y,idx_z)
                voxel_id_array.append(point3d)
                radius_array.append(voxel_radius)  # Store the corresponding radius
                seg_3d_points.append(point3d)

        for point3d in seg_3d_points:
            f.write(f"v {point3d[0]} {point3d[1]} {point3d[2]}\n")

        for i in range(len(seg_3d_points) - 1):
            f.write(f"l {i + 1} {i + 2}\n")  # OBJ indices start at 1

    print(f"Saved segment {idx} to {obj_filename}")

radius_array = np.array(radius_array)

distance_transform = CenterlineDialatorModule(dimensions, spacing)
result = distance_transform.distance_transform_near(voxel_id_array, radius_array)
image_to_fm = distance_transform.output_image

radius_max = -np.min(radius_array)

fm_filter = FastMarchingFilter()
fm_filter.set_algorithm(AlgorithmType.SUBCLASS_DEFINED)
fm_filter.set_input_image_carries_initial_value(True)
fm_filter.set_thresholds(-np.finfo(np.float32).max, 1)
fm_filter.set_seed_threshold(-max_radius, -0.1)
fm_filter.stop_after_reach_value = radius_max
fm_filter.run(image_to_fm)

## set unvisited voxels to a distinguish value
fm_filter.output_image[fm_filter.unvisited_voxels == True] = radius_max
dump_image_to_vtk(fm_filter.output_image, "Dilator_distancemap.vti")

# Load the image from VTI file or the provided image data
# vtk_image = load_vti_or_image("Dilator_distancemap.vti", image_data=fm_filter.output_image)
vtk_image = load_vti_or_image("Dilator_distancemap.vti")

# Extract the isosurface with value 0
isosurface = extract_isosurface(vtk_image, iso_value=0)

# Extract only the largest connected region
largest_region = get_largest_connected_region(isosurface)

# Display the extracted surface
display_3d_surface(largest_region)