import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyproj import Transformer


class GeoCoordTransformer:
    def __init__(self):
        self.to_utm = Transformer.from_crs("epsg:4326", "epsg:32650", always_xy=True)
        self.to_wgs84 = Transformer.from_crs("epsg:32650", "epsg:4326", always_xy=True)

    def wgs84_to_utm(self, lon, lat):  # 注意顺序：先经度，后纬度
        try:
            easting, northing = self.to_utm.transform(lon, lat)
            if np.isinf(easting) or np.isinf(northing):
                raise ValueError('Invalid UTM coordinates')
            return easting, northing
        except Exception as e:
            raise

    def utm_to_wgs84(self, easting, northing):
        try:
            lon, lat = self.to_wgs84.transform(easting, northing)  # 注意顺序：先经度，后纬度
            if np.isinf(lat) or np.isinf(lon):
                raise ValueError('Invalid WGS84 coordinates')
            return lon, lat
        except Exception as e:
            raise


def compute_reprojection_error(pos3d, pixels, K, dist_coeffs, rvec, tvec):
    projected_points, _ = cv2.projectPoints(pos3d, rvec, tvec, K, dist_coeffs)
    projected_points = projected_points.squeeze()
    errors = np.linalg.norm(pixels - projected_points, axis=1)
    return errors


def estimate_camera_orientation(pos3d, pixels, focal_lengths, sensor_sizes, image_size, known_camera_origin):
    pos3d = np.asarray(pos3d, dtype=np.float64).reshape(-1, 3)
    pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    best_K = None
    best_reprojection_error = float('inf')
    best_initial_rotation_vector = None
    best_initial_translation_vector = None
    best_optimized_rotation_vector = None
    best_optimized_translation_vector = None
    best_inliers = None
    best_focal_length = None
    best_sensor_size = None

    all_results = []

    transformer = GeoCoordTransformer()

    for focal_length in focal_lengths:
        for (sensor_width, sensor_height) in sensor_sizes:
            pixel_size_width = sensor_width / image_size[0]
            pixel_size_height = sensor_height / image_size[1]

            fx = focal_length / pixel_size_width
            fy = focal_length / pixel_size_height

            K = np.array([
                [fx, 0, image_size[0] / 2],
                [0, fy, image_size[1] / 2],
                [0, 0, 1]
            ])

            success, initial_rotation_vector, initial_translation_vector, inliers = cv2.solvePnPRansac(
                pos3d, pixels, K, dist_coeffs, useExtrinsicGuess=False,
                iterationsCount=5000, reprojectionError=30.0, confidence=0.99
            )

            if not success or inliers is None or len(inliers) < 6:
                continue

            errors_initial = compute_reprojection_error(pos3d[inliers.flatten()], pixels[inliers.flatten()], K, dist_coeffs,
                                                        initial_rotation_vector, initial_translation_vector)
            mean_error_initial = np.mean(errors_initial)

            R_matrix, _ = cv2.Rodrigues(initial_rotation_vector)
            camera_origin = -R_matrix.T @ initial_translation_vector.flatten()
            distance_to_known_origin = np.linalg.norm(camera_origin - known_camera_origin)

            all_results.append((distance_to_known_origin, mean_error_initial, K, focal_length, sensor_width, sensor_height, initial_rotation_vector, initial_translation_vector, camera_origin))

            if mean_error_initial < best_reprojection_error:
                best_reprojection_error = mean_error_initial
                best_K = K
                best_focal_length = focal_length
                best_sensor_size = (sensor_width, sensor_height)
                best_initial_rotation_vector = initial_rotation_vector
                best_initial_translation_vector = initial_translation_vector
                best_inliers = inliers

    if best_K is None:
        print("PnP RANSAC failed for all K matrices.")
        return None, None

    all_results.sort(key=lambda x: x[0])  # Sort by distance to known camera origin

    print(f"Best K Matrix with minimum mean reprojection error {best_reprojection_error:.2f} pixels:")
    print(f"Focal Length: {best_focal_length} mm, Sensor Size: {best_sensor_size[0]}x{best_sensor_size[1]} mm")
    print(best_K)

    print("\nTop 5 camera origins closest to known camera origin (with WGS84 coordinates):\n")
    for i in range(min(5, len(all_results))):
        distance, reprojection_error, K, focal_length, sensor_width, sensor_height, rvec, tvec, camera_origin = all_results[i]
        lon, lat = transformer.utm_to_wgs84(camera_origin[0], camera_origin[1])
        print(f"Rank {i+1}:")
        print(f"Distance to known camera origin: {distance:.2f}")
        print("Inliers:\n", inliers)
        print(f"Reprojection error: {reprojection_error:.2f} pixels")
        print(f"Focal Length: {focal_length} mm, Sensor Size: {sensor_width}x{sensor_height} mm")
        print(f"K Matrix:\n{K}")
        print(f"Camera Origin (UTM): {camera_origin}")
        print(f"Camera Origin (WGS84): ({lon}, {lat}, {camera_origin[2]})\n")

    optimized_rotation_vector, optimized_translation_vector = cv2.solvePnPRefineLM(
        pos3d[best_inliers.flatten()], pixels[best_inliers.flatten()], best_K, dist_coeffs,
        best_initial_rotation_vector, best_initial_translation_vector
    )

    print("Optimized Rotation Vector (rvec):\n", optimized_rotation_vector)
    print("Optimized Translation Vector (tvec):\n", optimized_translation_vector)

    rotation_vector_diff = np.linalg.norm(optimized_rotation_vector - best_initial_rotation_vector)
    translation_vector_diff = np.linalg.norm(optimized_translation_vector - best_initial_translation_vector)
    print(f"Rotation vector difference: {rotation_vector_diff:.6f}")
    print(f"Translation vector difference: {translation_vector_diff:.6f}")

    T_known = np.array([[739424.6], [2888281.18], [770]])
    R_matrix, _ = cv2.Rodrigues(optimized_rotation_vector)
    known_translation_vector = -R_matrix @ T_known

    print("Known Translation Vector (T):\n", known_translation_vector)

    known_translation_diff = np.linalg.norm(known_translation_vector - optimized_translation_vector)
    print(f"Known Translation Vector difference: {known_translation_diff:.6f}")

    errors_initial = compute_reprojection_error(pos3d[best_inliers.flatten()], pixels[best_inliers.flatten()], best_K, dist_coeffs,
                                                best_initial_rotation_vector, best_initial_translation_vector)
    errors_optimized = compute_reprojection_error(pos3d[best_inliers.flatten()], pixels[best_inliers.flatten()], best_K, dist_coeffs,
                                                  optimized_rotation_vector, optimized_translation_vector)
    errors_initial_known = compute_reprojection_error(pos3d[best_inliers.flatten()], pixels[best_inliers.flatten()], best_K, dist_coeffs,
                                                      best_initial_rotation_vector, known_translation_vector)
    errors_optimized_known = compute_reprojection_error(pos3d[best_inliers.flatten()], pixels[best_inliers.flatten()], best_K, dist_coeffs,
                                                        optimized_rotation_vector, known_translation_vector)

    print(f"Initial Mean reprojection error: {np.mean(errors_initial):.2f} pixels")
    print(f"Initial Max reprojection error: {np.max(errors_initial):.2f} pixels")
    print(f"Optimized Mean reprojection error: {np.mean(errors_optimized):.2f} pixels")
    print(f"Optimized Max reprojection error: {np.max(errors_optimized):.2f} pixels")
    print(f"Initial with Known Translation Vector Mean reprojection error: {np.mean(errors_initial_known):.2f} pixels")
    print(f"Initial with Known Translation Vector Max reprojection error: {np.max(errors_initial_known):.2f} pixels")
    print(f"Optimized with Known Translation Vector Mean reprojection error: {np.mean(errors_optimized_known):.2f} pixels")
    print(f"Optimized with Known Translation Vector Max reprojection error: {np.max(errors_optimized_known):.2f} pixels")

    return optimized_rotation_vector, optimized_translation_vector


def visualize_camera_pose_and_points(R, T, pos3d):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    R_matrix, _ = cv2.Rodrigues(R.reshape(3, ))
    camera_origin = -R_matrix.T @ T.flatten()
    print("Corrected Camera Origin:", camera_origin)

    camera_x = camera_origin + np.dot(R_matrix.T, np.array([100000, 0, 0]))
    camera_y = camera_origin + np.dot(R_matrix.T, np.array([0, 100000, 0]))
    camera_z = camera_origin + np.dot(R_matrix.T, np.array([0, 0, 100000]))

    ax.scatter(pos3d[:, 0], pos3d[:, 1], pos3d[:, 2], c='r', marker='o')
    for i, (x, y, z) in enumerate(pos3d):
        ax.text(x, y, z, str(i), color='black')

    ax.quiver(camera_origin[0], camera_origin[1], camera_origin[2],
              camera_x[0] - camera_origin[0], camera_x[1] - camera_origin[1], camera_x[2] - camera_origin[2], color='r',
              length=100000, normalize=True)
    ax.quiver(camera_origin[0], camera_origin[1], camera_origin[2],
              camera_y[0] - camera_origin[0], camera_y[1] - camera_origin[1], camera_y[2] - camera_origin[2], color='g',
              length=100000, normalize=True)
    ax.quiver(camera_origin[0], camera_origin[1], camera_origin[2],
              camera_z[0] - camera_origin[0], camera_z[1] - camera_origin[1], camera_z[2] - camera_origin[2], color='b',
              length=100000, normalize=True)

    ax.view_init(elev=20, azim=45)
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_title('Camera Pose and 3D Points Visualization (UTM)')
    plt.show()

pos3d = np.array([
    [739031.2, 2888840.39, 726.0],
    [738995.929, 2888848.16, 724.0],
    [738963.052, 2888845.45, 721.0],
    [739173.616, 2888834.91, 697.0],
    [739077.689, 2888935.68, 726.0],
    [739033.253, 2888924.78, 726.0],
    [738973.016, 2888907.82, 723.0],
    [739136.184, 2889025.65, 705.0],
    [739179.948, 2888631.85, 702.0],
    [739140.769, 2888574.49, 702.0],
    [739312.871, 2888549.50, 720.0],
    [739249.159, 2888541.79, 707.0]
])
pixels = np.array([
    [582, 296],
    [402, 301],
    [272, 314],
    [1440, 467],
    [965, 296],
    [666, 265],
    [392, 283],
    [1583, 319],
    [729, 606],
    [169, 696],
    [1804, 672],
    [885, 824]
])

focal_lengths = [90, 100, 120, 150, 180, 210, 240, 300, 360]
sensor_sizes = [
    (102, 127),  # 4x5英寸
    (127, 178),  # 5x7英寸
    (203, 254)   # 8x10英寸
]
image_size = (2142, 1620)
known_camera_origin = np.array([739424.6, 2888281.18, 770])

R, T = estimate_camera_orientation(pos3d, pixels, focal_lengths, sensor_sizes, image_size, known_camera_origin)
if R is not None:
    visualize_camera_pose_and_points(R, T, pos3d)