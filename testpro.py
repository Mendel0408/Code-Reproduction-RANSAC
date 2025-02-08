# -*- coding: utf-8 -*-

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import font_manager
import csv
import glob
import math
import logging
import pyproj
from pyproj import Transformer
from scipy.spatial import distance
import seaborn as sns
from scipy.interpolate import RegularGridInterpolator
from osgeo import gdal

# 设置字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题

# 验证字体是否存在
font_path = font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
print("Available fonts:", font_path)

# 设置日志记录
logging.basicConfig(level=logging.DEBUG, filename='debug.log', filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')

# 创建一个 Transformer 对象，用于将 WGS84 坐标转换为 UTM 坐标
transformer = Transformer.from_crs("epsg:4326", "epsg:32650")

def wgs84_to_utm(lat, lon):
    try:
        logging.debug(f'Converting WGS84 to UTM: lat={lat}, lon={lon}')
        easting, northing = transformer.transform(lat, lon)
        if np.isinf(easting) or np.isinf(northing):
            logging.error(f'Invalid UTM coordinates: easting={easting}, northing={northing}')
            raise ValueError('Invalid UTM coordinates')
        return easting, northing
    except Exception as e:
        logging.error(f'Error converting WGS84 to UTM: {e}')
        raise

# 可视化
def plot_error_histogram(errors, title='误差频率图'):
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=30, alpha=0.75, color='blue', edgecolor='black')
    plt.title(title)
    plt.xlabel('误差大小')
    plt.ylabel('频率')
    plt.grid(True)
    plt.show()

def plot_camera_location_scores(scores):
    scores = np.array(scores)
    # 将X、Y坐标转换为经纬度坐标（WGS84）
    transformer_to_wgs84 = Transformer.from_crs("epsg:32650", "epsg:4326")
    latitudes, longitudes = transformer_to_wgs84.transform(scores[:, 4], scores[:, 5])
    plt.figure(figsize=(12, 8))
    # 绘制 min_score 的散点图，越小的误差颜色越深
    scatter = plt.scatter(longitudes, latitudes, c=scores[:, 1], cmap='viridis_r', marker='o')
    plt.colorbar(scatter, label='最小匹配误差 (min_score)')
    plt.title('潜在相机位置得分图')
    plt.xlabel('经度')
    plt.ylabel('纬度')
    plt.grid(True)
    plt.show()


def plot_camera_pose(camera_locations, best_location_idx):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    loc3ds = np.array([loc['pos3d'] for loc in camera_locations])
    # 将X、Y坐标转换为经纬度坐标（WGS84）
    transformer_to_wgs84 = Transformer.from_crs("epsg:32650", "epsg:4326")
    latitudes, longitudes = transformer_to_wgs84.transform(loc3ds[:, 0], loc3ds[:, 1])
    ax.scatter(longitudes, latitudes, loc3ds[:, 2], c='blue', marker='o')
    ax.scatter(longitudes[best_location_idx], latitudes[best_location_idx], loc3ds[best_location_idx, 2], c='red',
               marker='^')
    ax.set_title('相机位姿图')
    ax.set_xlabel('经度')
    ax.set_ylabel('纬度')
    ax.set_zlabel('高度')
    plt.show()

def plot_error_boxplot(errors):
    plt.figure(figsize=(10, 6))
    plt.boxplot(errors, vert=True, patch_artist=True)
    plt.title('误差分布箱线图')
    plt.ylabel('误差大小')
    plt.grid(True)
    plt.show()

def plot_distance_histogram(distances):
    plt.figure(figsize=(10, 6))
    plt.hist(distances, bins=30, alpha=0.75, color='green', edgecolor='black')
    plt.title('距离度量直方图')
    plt.xlabel('距离大小')
    plt.ylabel('频率')
    plt.grid(True)
    plt.show()


def plot_angle_rose(angles):
    plt.figure(figsize=(10, 6))
    plt.subplot(projection='polar')
    plt.hist(angles, bins=30, alpha=0.75, color='purple', edgecolor='black')
    plt.title('角度度量玫瑰图')
    plt.show()


def plot_nearest_neighbor_distances(nearest_neighbor_distances):
    plt.figure(figsize=(10, 6))
    plt.hist(nearest_neighbor_distances, bins=30, alpha=0.75, color='orange', edgecolor='black')
    plt.title('特征点最近邻距离图')
    plt.xlabel('距离大小')
    plt.ylabel('频率')
    plt.grid(True)
    plt.show()

def plot_homography_matrix_heatmap(H):
    plt.figure(figsize=(10, 6))
    sns.heatmap(H, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('单应性矩阵热图')
    plt.show()


def plot_ransac_scatter(inliers, outliers):
    plt.figure(figsize=(10, 6))
    if inliers.size > 0:
        plt.scatter(inliers[:, 0], inliers[:, 1], c='green', marker='o', label='内点')
    if outliers.size > 0:
        plt.scatter(outliers[:, 0], outliers[:, 1], c='red', marker='x', label='外点')
    plt.title('RANSAC算法散点图')
    plt.xlabel('X 坐标')
    plt.ylabel('Y 坐标')
    plt.legend()
    plt.grid(True)
    plt.show()


# **********
# Calculate true and pixel distances between features
# **********
def correlate_features(features, depth_val):
    result = ['id', 'sym_s', 'x_s', 'y_s', 'pixel_x_s', 'pixel_y_s', 'calc_pixel_x_s', 'calc_pixel_y_s',
              'sym_t', 'x_t', 'y_t', 'pixel_x_t', 'pixel_y_t', 'calc_pixel_x_t', 'calc_pixel_y_t',
              'dis_m_x', 'dis_m_y', 'dis_m', 'dis_pix_x', 'dis_pix_y', 'dis_pix', 'dis_c_pix_x', 'dis_c_pix_y',
              'dis_c_pix', 'bear_pix', 'dis_depth_pix', 'bear_c_pix', 'dis_depth_c_pix']

    results = []
    results.append(result)
    count = 1
    i = 0
    j = 0
    features.remove(features[0])  # remove the headers
    features.sort()  # sort alphabethically
    for f1 in features:
        i = j
        while i < len(features):
            if f1[1] != features[i][1]:
                dis_m_x = int(features[i][3]) - int(f1[3])
                dis_m_y = int(features[i][4]) - int(f1[4])
                dis_m = math.sqrt(math.pow(dis_m_x, 2) + math.pow(dis_m_y, 2))

                if f1[5] != 0 and features[i][5] != 0:
                    dis_pix_x = int(features[i][5]) - int(f1[5])
                    dis_pix_y = int(features[i][6]) - int(f1[6])
                else:
                    dis_pix_x = 0
                    dis_pix_y = 0
                dis_pix = math.sqrt(math.pow(dis_pix_x, 2) + math.pow(dis_pix_y, 2))

                if features[i][7] != 0 and f1[7] != 0:
                    dis_c_pix_x = int(features[i][7]) - int(f1[7])
                    dis_c_pix_y = int(features[i][8]) - int(f1[8])
                else:
                    dis_c_pix_x = 0
                    dis_c_pix_y = 0
                dis_c_pix = math.sqrt(math.pow(dis_c_pix_x, 2) + math.pow(dis_c_pix_y, 2))

                bear_pix = calc_bearing(f1[5], f1[6], features[i][5], features[i][6])
                if bear_pix != 0 and bear_pix <= 180:
                    dis_depth_pix = (abs(bear_pix - 90) / 90 + depth_val) * dis_pix
                elif bear_pix != 0 and bear_pix > 180:
                    dis_depth_pix = (abs(bear_pix - 270) / 90 + depth_val) * dis_pix
                else:
                    dis_depth_pix = 0

                bear_c_pix = calc_bearing(f1[7], f1[8], features[i][7], features[i][8])
                if bear_c_pix != 0 and bear_c_pix <= 180:
                    dis_depth_c_pix = (abs(bear_c_pix - 90) / 90 + depth_val) * dis_c_pix
                elif bear_c_pix != 0 and bear_c_pix > 180:
                    dis_depth_c_pix = (abs(bear_c_pix - 270) / 90 + depth_val) * dis_c_pix
                else:
                    dis_depth_c_pix = 0

                result = [str(count), f1[1], f1[3], f1[4], f1[5], f1[6], f1[7], f1[8], features[i][1], features[i][3],
                          features[i][4], features[i][5], features[i][6], features[i][7], features[i][8],
                          dis_m_x, dis_m_y, dis_m, dis_pix_x, dis_pix_y, dis_pix, dis_c_pix_x, dis_c_pix_y, dis_c_pix,
                          bear_pix, dis_depth_pix, bear_c_pix, dis_depth_c_pix]

                results.append(result)
                count += 1
            i += 1
        j += 1
    return results


# **********
# Calculation of the bearing from point 1 to point 2
# **********
def calc_bearing(x1, y1, x2, y2):
    if x1 == 0 or x2 == 0 or y1 == 0 or y2 == 0:
        degrees_final = 0
    else:
        deltaX = x2 - x1
        deltaY = y2 - y1

        degrees_temp = math.atan2(deltaX, deltaY) / math.pi * 180

        if degrees_temp < 0:
            degrees_final = 360 + degrees_temp
        else:
            degrees_final = degrees_temp

        if degrees_final < 180:
            degrees_final = 180 - degrees_final
        else:
            degrees_final = 360 + 180 - degrees_final

    return degrees_final


# **********
# Camera calibration process
# **********
def calibrate_camera(size):
    CHECKERBOARD = (6, 9)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, size, 0.001)  # was 30

    objpoints = []  # Creating vector to store vectors of 3D points for each checkerboard image
    imgpoints = []  # Creating vector to store vectors of 2D points for each checkerboard image

    # Defining the world coordinates for 3D points
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    prev_img_shape = None

    images = glob.glob(
        r'.\camera_calibration\images\*.jpg')  # TODO: change the path according to the path in your environmrnt
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

        if ret == True:
            objpoints.append(objp)
            # refining pixel coordinates for given 2d points.
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
            print(fname)

        cv2.waitKey(0)

    cv2.destroyAllWindows()
    h, w = img.shape[:2]
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return ret, mtx, dist, rvecs, tvecs


# **********
# Find homographies function
# **********
def find_homographies(recs, camera_locations, im, show, ransacbound, outputfile):
    pixels = []
    pos3ds = []
    symbols = []
    for r in recs:
        pixels.append(r['pixel'])
        pos3ds.append(r['pos3d'])
        symbols.append(r['symbol'])
    pixels = np.array(pixels)
    pos3ds = np.array(pos3ds)
    symbols = np.array(symbols)
    loc3ds = []
    grids = []
    for cl in camera_locations:
        grids.append(cl['grid_code'])
        loc3ds.append(cl['pos3d'])
    grids = np.array(grids)
    loc3ds = np.array(loc3ds)
    num_matches = np.zeros((loc3ds.shape[0], 2))
    scores = []
    for i in range(0, grids.shape[0], 1):  # 50
        if grids[i] >= grid_code_min:
            if show:
                print(i, grids[i], loc3ds[i])
            M, num_matches[i, 0], num_matches[i, 1] = find_homography(recs, pixels, pos3ds, symbols, loc3ds[i], im, show,
                                                                   ransacbound, outputfile)
        else:
            num_matches[i, :] = 0
        score = [i + 1, num_matches[i, 0], num_matches[i, 1], grids[i], loc3ds[i][0], loc3ds[i][1], loc3ds[i][2]]
        scores.append(score)

    if show is False:
        outputCsv = output.replace(".jpg", "_location.csv")
        csvFile = open(outputCsv, 'w', newline='', encoding='utf-8')
        csvWriter = csv.writer(csvFile)
        csvWriter.writerow(['location_id', 'min_score', 'max_score', 'grid_code', 'Z', 'X', 'Y'])
        for s in scores:
            csvWriter.writerow(s)

    plot_camera_location_scores(scores)
    plot_camera_pose(camera_locations, np.argmin(num_matches[:, 1]))

    return num_matches

# **********
# Find homography function
# **********
def find_homography(recs, pixels, pos3ds, symbols, camera_location, im, show, ransacbound, outputfile):
    pixels = np.array(pixels)
    pos2 = np.zeros((len(pixels), 2))
    good = np.zeros(len(pixels))
    for i in range(len(pixels)):
        good[i] = pixels[i][0] != 0 or pixels[i][1] != 0
        p = pos3ds[i, :] - camera_location
        p = np.array([p[2], p[1], p[0]])
        p = p / p[2]
        pos2[i, :] = p[0:2]
    M, mask = cv2.findHomography(pos2[good == 1], np.array(pixels)[good == 1], cv2.RANSAC, ransacbound)

    M = np.linalg.inv(M)
    logging.debug(f'Homography Matrix M: {M}')
    logging.debug(f'Mask: {mask}')
    if show:
        print('M', M, np.sum(mask))
    if show:
        plt.figure(figsize=(40, 20))
        plt.imshow(im)
        for rec in recs:
            symbol = rec['symbol']
            pixel = rec['pixel']
            if pixel[0] != 0 or pixel[1] != 0:
                plt.text(pixel[0], pixel[1], symbol, color='purple', fontsize=6, weight='bold')
    err1 = 0
    err2 = 0
    feature = ['id', 'symbol', 'name', 'x', 'y', 'pixel_x', 'pixel_y', 'calc_pixel_x', 'calc_pixel_y']
    features = []
    features.append(feature)
    for i in range(pos2[good == 1].shape[0]):
        p1 = np.array(pixels)[good == 1][i, :]
        pp = np.array([pos2[good == 1][i, 0], pos2[good == 1][i, 1], 1.0])
        pp2 = np.matmul(np.linalg.inv(M), pp)
        pp2 = pp2 / pp2[2]
        P1 = np.array([p1[0], p1[1], 1.0])
        PP2 = np.matmul(M, P1)
        PP2 = PP2 / PP2[2]
        P2 = pos2[good == 1][i, :]
        logging.debug(f'Feature {i}: mask={mask[i]}, p1={p1}, pp2={pp2[0:2]}, distance={np.linalg.norm(p1 - pp2[0:2])}')
        if show and good[i]:
            print(i)
            print(mask[i] == 1, p1, pp2[0:2], np.linalg.norm(p1 - pp2[0:2]))
            print(mask[i] == 1, P2, PP2[0:2], np.linalg.norm(P2 - PP2[0:2]))
        if mask[i] == 1:
            err1 += np.linalg.norm(p1 - pp2[0:2])
            err2 += np.linalg.norm(P2 - PP2[0:2])
        if show:
            color = 'green' if mask[i] == 1 else 'red'
            plt.plot([p1[0], pp2[0]], [p1[1], pp2[1]], color=color, linewidth=2)
            plt.plot(p1[0], p1[1], marker='X', color=color, markersize=3)
            plt.plot(pp2[0], pp2[1], marker='o', color=color, markersize=3)
            sym = ''
            name = ''
            for r in recs:
                px = r['pixel'].tolist()
                if px[0] == p1[0] and px[1] == p1[1]:
                    sym = r['symbol']
                    name = r['name']
                    x = r['pos3d'][0]
                    y = r['pos3d'][1]
                    break
            feature = [i, sym, name, x, y, p1[0], p1[1], pp2[0], pp2[1]]
            features.append(feature)

    i = -1
    for r in recs:  # Extracting features that were not noted on the image (pixel_x and pixel_y are 0)
        i += 1
        p1 = pixels[i, :]
        if p1[0] == 0 and p1[1] == 0:
            pp = np.array([pos2[i, 0], pos2[i, 1], 1.0])
            pp2 = np.matmul(np.linalg.inv(M), pp)
            pp2 = pp2 / pp2[2]
            logging.debug(f'Unnoted Feature {i}: symbol={r["symbol"]}, pp2={pp2[0:2]}')
            if show:
                plt.text(pp2[0], pp2[1], r['symbol'], color='black', fontsize=6, style='italic',
                         weight='bold')
                plt.plot(pp2[0], pp2[1], marker='s', markersize=3, color='black')
                x = r['pos3d'][0]
                y = r['pos3d'][1]
                feature = [i, recs[i]['symbol'], recs[i]['name'], x, y, 0, 0, pp2[0], pp2[1]]
                features.append(feature)
    if show:
        outputCsv = outputfile.replace(".jpg", "_accuracies.csv")
        with open(outputCsv, 'w', newline='', encoding='utf-8-sig') as csvFile:
            csvWriter = csv.writer(csvFile)
            for f in features:
                csvWriter.writerow(f)

        # 发送特征到相关函数
        results = correlate_features(features, 1)
        outputCsv = outputfile.replace(".jpg", "_correlations.csv")
        with open(outputCsv, 'w', newline='', encoding='utf-8') as csvFile:
            csvWriter = csv.writer(csvFile)
            for r in results:
                csvWriter.writerow(r)

        plot_distance_histogram(
            [math.sqrt(math.pow(int(f1[3]) - int(f2[3]), 2) + math.pow(int(f1[4]) - int(f2[4]), 2)) for f1 in features
             for f2 in features if f1 != f2])
        plot_angle_rose([calc_bearing(f1[5], f1[6], f2[5], f2[6]) for f1 in features for f2 in features if f1 != f2])
        points = np.array([[f[5], f[6]] for f in features])
        dist_matrix = distance.cdist(points, points, 'euclidean')
        np.fill_diagonal(dist_matrix, np.inf)
        plot_nearest_neighbor_distances(np.min(dist_matrix, axis=1))
        plot_error_histogram([err1], '误差频率图 (err1)')
        plot_error_histogram([err2], '误差频率图 (err2)')
        plot_error_boxplot([np.linalg.norm(p1 - pp2[0:2]) for i in range(pos2[good == 1].shape[0])])
        plot_homography_matrix_heatmap(M)
        inliers = np.array([p1 for i in range(pos2[good == 1].shape[0]) if mask[i] == 1])
        outliers = np.array([p1 for i in range(pos2[good == 1].shape[0]) if mask[i] == 0])
        plot_ransac_scatter(inliers, outliers)

        print('Output file: ', outputfile)
        plt.savefig(outputfile, dpi=300)
        plt.show()

    err2 += np.sum(1 - mask) * ransacbound
    if show:
        print('err', err1, err1 / np.sum(mask), err2, err2 / np.sum(mask))
    return M, err1, err2

# 加载DEM数据
def load_dem_data(dem_file):
    dataset = gdal.Open(dem_file)

    if dataset is None:
        raise RuntimeError(f"无法加载 DEM 文件: {dem_file}")

    dem_data = dataset.ReadAsArray()
    gt = dataset.GetGeoTransform()

    print(f"【DEBUG】DEM 形状: {dem_data.shape}")
    print(f"【DEBUG】DEM 仿射变换参数: {gt}")

    # 计算 DEM 坐标范围
    dem_x = np.arange(dem_data.shape[1]) * gt[1] + gt[0]  # 经度范围
    dem_y = np.arange(dem_data.shape[0]) * gt[5] + gt[3]  # 纬度范围

    print(f"【DEBUG】DEM 范围: 经度 [{dem_x.min()}, {dem_x.max()}], 纬度 [{dem_y.min()}, {dem_y.max()}]")

    dem_interpolator = RegularGridInterpolator((dem_y, dem_x), dem_data)  # 插值器

    return dem_interpolator, dem_x, dem_y  # ✅ 返回 dem_x, dem_y


# 分解单应性矩阵，得到内参矩阵和外参矩阵
def decompose_homography(M):
    logging.debug(f'Decomposing homography matrix M: {M}')

    if M.shape != (3, 3):
        raise ValueError("Input matrix M must be a 3x3 matrix")

    solutions = cv2.decomposeHomographyMat(M, np.eye(3))

    if solutions is None or len(solutions) < 3:
        raise RuntimeError("Homography decomposition failed, no valid solution found.")

    # 选择第一个解
    K = np.array(solutions[0], dtype=np.float64)
    R = solutions[1][0]  # 旋转矩阵，选择第一个解
    t = solutions[2][0]  # 平移向量，选择第一个解

    return K, R, t

# 使用PnP算法进行相机姿态估计
def estimate_camera_pose(pos3d, pixels, K):
    pos3d = np.asarray(pos3d, dtype=np.float64).reshape(-1, 3)  # 确保是 float64
    pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)  # 这里改为 (4,1) float64

    # 检查是否存在 NaN 或 Inf
    if np.isnan(pos3d).any() or np.isnan(pixels).any():
        raise ValueError("pos3d 或 pixels 含有 NaN 值")
    if np.isinf(pos3d).any() or np.isinf(pixels).any():
        raise ValueError("pos3d 或 pixels 含有 Inf 值")

    # 运行 solvePnP
    success, rotation_vector, translation_vector = cv2.solvePnP(pos3d, pixels, K, dist_coeffs)
    if not success:
        raise RuntimeError("PnP 计算失败，请检查输入数据。")

    return rotation_vector, translation_vector

# 检查并调整translation_vector的值
def check_translation_vector(translation_vector):
    max_value = 1e4  # 根据实际情况调整阈值
    if np.any(np.abs(translation_vector) > max_value):
        logging.warning(f"Translation vector values are too large: {translation_vector}")
        translation_vector = np.clip(translation_vector, -max_value, max_value)
    return translation_vector

# 将像素坐标转换为射线
def pixel_to_ray(pixel_coord, K, rotation_vector, translation_vector, ray_origin):
    pixel_coord_homogeneous = np.append(pixel_coord, 1).reshape(-1, 1)
    inv_K = np.linalg.inv(K)
    normalized_coord = np.dot(inv_K, pixel_coord_homogeneous)
    normalized_coord = normalized_coord / np.linalg.norm(normalized_coord)

    R, _ = cv2.Rodrigues(rotation_vector)
    T = translation_vector.reshape(-1, 1)
    T = check_translation_vector(T)  # 确保translation_vector的值在合理范围内

    # 检查 translation_vector 是否合理
    if np.any(np.abs(T) > 1e6):  # 假设合理的 translation_vector 值在 1e6 以内
        logging.warning(f"Translation vector is too large: {T}")
        T = np.clip(T, -1e6, 1e6)  # 限制 translation_vector 的值

    print(f"【DEBUG】最佳相机位置（ray_origin）: {ray_origin}")

    ray_direction = np.dot(R, normalized_coord).ravel()
    ray_direction = ray_direction / np.linalg.norm(ray_direction)
    print(f"【DEBUG】修正后的 ray_direction: {ray_direction}")

    print(f"【DEBUG】射线原点（UTM 或投影坐标）: {ray_origin}")
    print(f"【DEBUG】射线方向: {ray_direction}")

    # 转换 ray_origin 从 UTM 到 WGS84 经纬度
    transformer = Transformer.from_crs("epsg:32650", "epsg:4326")
    lat, lon = transformer.transform(ray_origin[0], ray_origin[1])
    ray_origin = np.array([lat, lon, ray_origin[2]])  # ✅ 确保格式是 (纬度, 经度, 高度)
    print(f"【DEBUG】修正后的 ray_origin (纬度, 经度, 高度): {ray_origin}")

    # 先将射线方向向量转换到 WGS84 坐标系
    utm_to_wgs_transformer = Transformer.from_crs("epsg:32650", "epsg:4326", always_xy=True)
    # 计算射线方向在 WGS84 坐标系下的变化
    dx, dy = ray_direction[0], ray_direction[1]
    lat_shift, lon_shift = utm_to_wgs_transformer.transform(ray_origin[0] + dx, ray_origin[1] + dy)
    # 计算转换后的方向向量
    ray_direction_wgs = np.array([lat_shift - ray_origin[0], lon_shift - ray_origin[1], ray_direction[2]])
    # 归一化射线方向
    ray_direction_wgs = ray_direction_wgs / np.linalg.norm(ray_direction_wgs)
    print(f"【DEBUG】修正后的 ray_direction (WGS84): {ray_direction_wgs}")

    return ray_origin, ray_direction

# 计算射线与DEM的交点
def ray_intersect_dem(ray_origin, ray_direction, dem_interpolator, dem_x, dem_y):
    print(f"【DEBUG】最终用于 DEM 计算的 ray_origin (WGS84): {ray_origin}")
    print(f"【DEBUG】最终用于 DEM 计算的 ray_direction (WGS84): {ray_direction_wgs}")

    t_values = np.linspace(0, 10000, 1000)
    intersection = ray_intersect_dem(ray_origin, ray_direction_wgs, dem_data)

    for t in t_values:
        point = ray_origin + t * ray_direction
        print(f"【DEBUG】射线步进 t={t}: point={point}")

        # **point 一开始就是 WGS84 坐标系，因此不需要转换**
        if (point[0] < dem_x.min() or point[0] > dem_x.max() or
                point[1] < dem_y.min() or point[1] > dem_y.max()):
            print(f"❌【错误】点超出 DEM 范围: {point}")
            continue

        dem_height = dem_interpolator([point[1], point[0]])[0]
        print(f"【DEBUG】DEM 高度: {dem_height}")

        if point[2] <= dem_height:
            intersection = point
            break

    if intersection is None:
        print("❌【错误】射线未与 DEM 相交")

    return intersection


# 输入像素坐标，输出地理坐标
def pixel_to_geo(pixel_coord, K, rotation_vector, translation_vector, ray_origin, dem_interpolator, dem_x, dem_y):
    ray_origin, ray_direction = pixel_to_ray(pixel_coord, K, rotation_vector, translation_vector, ray_origin)
    geo_coord = ray_intersect_dem(ray_origin, ray_direction, dem_interpolator, dem_x, dem_y)  # ✅ 传入 dem_x, dem_y
    return geo_coord

# **********
# read data from the features file
# **********
def read_points_data(filename, pixel_x, pixel_y, scale):
    with open(filename, encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        recs = []
        pixels = []
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
                names = row
                indx = names.index(pixel_x)
                indy = names.index(pixel_y)
            else:
                line_count += 1
                symbol = row[1]
                name = row[2]
                pixel = np.array([int(row[indx]), int(row[indy])]) / scale
                longitude = float(row[4])
                latitude = float(row[5])
                elevation = float(row[6])
                height = float(row[3]) + float(elevation)
                # 跳过像素坐标为0,0的点
                if pixel[0] == 0 and pixel[1] == 0:
                    continue
                pixels.append(pixel)
                # 添加坐标转换
                try:
                    logging.debug(f'Processing row {line_count}: lat={latitude}, lon={longitude}')
                    easting, northing = wgs84_to_utm(latitude, longitude)
                    pos3d = np.array([easting, northing, height])
                except ValueError as e:
                    logging.error(f'Error processing row {line_count}: {e}')
                    continue

                rec = {'symbol': symbol,
                       'pixel': pixel,
                       'pos3d': pos3d,
                       'name': name}
                recs.append(rec)
        logging.debug(f'Processed {line_count} lines.')
        return recs


# **********
# read data from the potential camera locations file
# **********
def read_camera_locations():
    with open(camera_locations, encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        recs = []
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
                names = row
            else:
                line_count += 1
                grid_code = int(row[1])
                longitude = float(row[2])
                latitude = float(row[3])
                height = float(row[4]) + 2.0  # addition of 2 meters as the observer height
                # 添加坐标转换
                try:
                    logging.debug(f'Processing row {line_count}: lat={latitude}, lon={longitude}')
                    easting, northing = wgs84_to_utm(latitude, longitude)
                    pos3d = np.array([easting, northing, height])
                except ValueError as e:
                    logging.error(f'Error processing row {line_count}: {e}')
                    continue

                rec = {'grid_code': grid_code,
                       'pos3d': pos3d}
                recs.append(rec)
        logging.debug(f'Processed {line_count} lines.')
        return recs


# **********
# Main function
# **********
def do_it(image_name, features, pixel_x, pixel_y, output, scale, dem_file):
    im = cv2.imread(image_name)
    im2 = np.copy(im)
    im[:, :, 0] = im2[:, :, 2]
    im[:, :, 1] = im2[:, :, 1]
    im[:, :, 2] = im2[:, :, 0]

    plt.figure(figsize=(11.69, 8.27))  # 40,20
    plt.imshow(im)

    recs = read_points_data(features, pixel_x, pixel_y, scale)
    locations = read_camera_locations()
    pixels = []
    for rec in recs:
        symbol = rec['symbol']
        pixel = rec['pixel']
        if pixel[0] != 0 or pixel[1] != 0:
            plt.text(pixel[0], pixel[1], symbol, color='red', fontsize=6)
        pixels.append(pixel)
    pixels = np.array(pixels)

    num_matches12 = find_homographies(recs, locations, im, False, 75.0, output)
    num_matches2 = num_matches12[:, 1]
    # print(np.min(num_matches2[num_matches2 > 0]))
    # print(np.max(num_matches2[num_matches2 > 0]))

    num_matches2[num_matches2 == 0] = 1000000
    print(np.min(num_matches2))

    theloci = np.argmin(num_matches2)  # theloci contains the best location for the camera
    # 读取最佳相机位置（地理坐标）
    ray_origin = locations[theloci]['pos3d']  # 这里 theloci 是已筛选出的最佳相机位置（地理坐标）
    print(f"【DEBUG】最佳相机位置（ray_origin）: {ray_origin}")

    best_location = locations[theloci]['pos3d']
    print('location id: ' + str(theloci) + ' - ' + str(locations[theloci]))

    find_homographies(recs, [locations[theloci]], im, True, 75.0, output)  # Orig = 120.0

    # 使用现有代码计算的单应性矩阵
    best_homography_matrix, err1, err2 = find_homography(recs, pixels, np.array([rec['pos3d'] for rec in recs]),
                                                         np.array([rec['symbol'] for rec in recs]), best_location, im,
                                                         False, 75.0, output)
    logging.debug(f'Best homography matrix: {best_homography_matrix}, err1: {err1}, err2: {err2}')

    # 分解单应性矩阵
    K, R, t = decompose_homography(best_homography_matrix)
    if K.shape != (3, 3):
        print(f"🚨【错误】K 计算失败，使用默认相机矩阵！")
        # 根据图像尺寸计算 K 矩阵
        width, height = im.shape[1], im.shape[0]  # 读取图像的宽度和高度
        cx, cy = width / 2, height / 2  # 主点位置假设为图像中心

        # 焦距可以暂时使用默认值，或根据图像尺寸来调整
        fx = fy = 1000  # 假设焦距为 1000，实际可以根据相机内参来调整
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

        print(f"【DEBUG】计算出的 K 矩阵: \n{K}")
    logging.debug(f'Chosen K: {K}')
    print(f"【DEBUG】K 形状: {K.shape}, dtype: {K.dtype}")

    M = best_homography_matrix

    # 使用第一个非零像素坐标
    pixel_x, pixel_y = pixels[0]

    # 打印 pixel_x 和 pixel_y 的类型
    print(f"【DEBUG】pixel_x 类型: {type(pixel_x)}, pixel_y 类型: {type(pixel_y)}")

    # 计算 `ray_direction`，使用相机旋转矩阵 `R`
    ray_direction = np.dot(R, np.linalg.inv(K) @ np.array([pixel_x, pixel_y, 1]))  # ✅ 先归一化像素坐标，再转换
    ray_direction = ray_direction / np.linalg.norm(ray_direction)  # 归一化射线方向

    print(f"【DEBUG】修正后的 ray_direction (UTM): {ray_direction}")
    # 先转换 `ray_direction` 到 WGS84
    utm_to_wgs_transformer = Transformer.from_crs("epsg:32650", "epsg:4326", always_xy=True)
    dx, dy = ray_direction[0], ray_direction[1]
    lat_shift, lon_shift = utm_to_wgs_transformer.transform(ray_origin[0] + dx, ray_origin[1] + dy)

    # 计算转换后的方向向量
    ray_direction_wgs = np.array([lat_shift - ray_origin[0], lon_shift - ray_origin[1], ray_direction[2]])
    ray_direction_wgs = ray_direction_wgs / np.linalg.norm(ray_direction_wgs)  # 归一化

    print(f"【DEBUG】修正后的 ray_direction (WGS84): {ray_direction_wgs}")

    # ✅ 添加测试代码，检查 `pos3d` 和 `pixels`
    # ✅ 先提取 `pos3d` 和 `pixels`
    pos3d = np.array([rec['pos3d'] for rec in recs])
    pixels = np.array([rec['pixel'] for rec in recs])
    # 过滤掉 `pixels` 为 `(0,0)` 的数据
    valid_indices = np.where(~np.all(pixels == [0, 0], axis=1))  # 只选取 `pixels` 不等于 (0,0) 的行

    pos3d_valid = pos3d[valid_indices]  # 仅保留有效的 3D 点
    pixels_valid = pixels[valid_indices]  # 仅保留有效的 2D 像素点

    print(f"【DEBUG】过滤后 pos3d 形状: {pos3d_valid.shape}")
    print(f"【DEBUG】过滤后 pixels 形状: {pixels_valid.shape}")

    print(f"【DEBUG】有效 3D 点（UTM 50N）：\n{pos3d_valid}")
    print(f"【DEBUG】有效 2D 像素点（像素坐标）：\n{pixels_valid}")

    # 归一化3D点和2D点
    def normalize_points(pos3d, pixels):
        pos3d_mean = np.mean(pos3d, axis=0)
        pixels_mean = np.mean(pixels, axis=0)

        pos3d_normalized = pos3d - pos3d_mean
        pixels_normalized = pixels - pixels_mean

        return pos3d_normalized, pixels_normalized

    # 在运行PnP算法前，归一化3D点和2D点
    pos3d_normalized, pixels_normalized = normalize_points(pos3d_valid, pixels_valid)
    print(f"【DEBUG】pos3d（3D 世界坐标）: \n{pos3d[:5]}")  # 打印前 5 个点
    print(f"【DEBUG】pixels（2D 像素坐标）: \n{pixels[:5]}")  # 打印前 5 个像素点

    # 确保 pos3d 和 pixels 数量一致
    if len(pos3d) != len(pixels):
        raise ValueError("🚨 `pos3d` 和 `pixels` 数量不一致！")

    # 运行PnP算法
    success, rotation_vector, translation_vector = cv2.solvePnP(
        pos3d_normalized, pixels_normalized, K, np.zeros(4, dtype=np.float32)
    )

    # 确保translation_vector的值在合理范围内
    translation_vector = check_translation_vector(translation_vector)

    if not success:
        print("🚨 `solvePnP` 计算失败，请检查输入数据！")
    print(f"【DEBUG】solvePnP 计算的 rotation_vector:\n{rotation_vector}")
    print(f"【DEBUG】solvePnP 计算的 translation_vector:\n{translation_vector}")

    if np.linalg.norm(translation_vector) > 10000:
        print(f"🚨【错误】translation_vector 数值过大，可能计算错误: {translation_vector}")

    # 继续运行相机位姿估计
    rotation_vector, translation_vector = estimate_camera_pose(np.array([rec['pos3d'] for rec in recs]), pixels, K)

    # 读取DEM数据
    dem_interpolator, dem_x, dem_y = load_dem_data(dem_file)

    # 交互式输入像素坐标，输出地理坐标
    while True:
        try:
            input_pixel_x, input_pixel_y = None, None  # 确保变量已初始化
            input_pixel = input("请输入像素坐标 (x, y) 或输入 'exit' 退出: ").strip()

            if input_pixel.lower() == 'exit':
                break

            print(f"【DEBUG】输入原始内容: {repr(input_pixel)}")  # 调试信息，查看输入内容

            # 处理中文逗号，去除空格
            pixel_values = input_pixel.replace(" ", "").replace("，", ",").split(",")
            print(f"【DEBUG】解析后: {pixel_values}")  # 查看解析后的数据

            if len(pixel_values) != 2:
                print("输入格式错误，请使用 (x, y) 形式，例如：755,975")
                continue  # 避免变量未赋值时继续执行

            input_pixel_x, input_pixel_y = map(float, pixel_values)
            print(f"【DEBUG】转换为浮点数: x={input_pixel_x}, y={input_pixel_y}")  # 检查是否成功解析

            input_pixel = np.array([input_pixel_x, input_pixel_y])

            # 计算地理坐标
            geo_coord = pixel_to_geo(input_pixel, K, rotation_vector, translation_vector, ray_origin, dem_interpolator,
                                     dem_x, dem_y)

            if geo_coord is not None:
                print(
                    f"像素坐标 ({input_pixel_x}, {input_pixel_y}) 对应的地理坐标: 经度 {geo_coord[0]:.6f}, 纬度 {geo_coord[1]:.6f}")
            else:
                print(f"无法找到 ({input_pixel_x}, {input_pixel_y}) 对应的地理坐标，请检查输入或 DEM 数据。")

        except ValueError as e:
            print(f"输入格式错误，请使用 (x, y) 形式，例如：755,975，错误详情: {e}")
        except Exception as e:
            print(f"发生未知错误: {e}")


img = '1898'
# img = '1900-1910'
# img = '1910'
# img = '1912s'
# img = '1915 (2)'
# img = '1915'
# img = '1920-1930'
# img = '1925-1930'
# img = '1930'
# img = 'center of the settlement kuliang'
# img = 'kuliang hills'
# img = 'kuliang panorama central segment'
# img = 'kuliang Pine Crag road'
# img = 'Siems Siemssen'
# img = 'View Kuliang includes tennis courts'
# img = 'Worley Family-20'

camera_locations = ''
grid_code_min = 0

if img == '1898':
    ret, mtx, dist, rvecs, tvecs = calibrate_camera(23)
    if ret is None:
        logging.error("Camera calibration failed.")
    else:
        img = cv2.imread('1898.jpg')
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)  # un-distort
        cv2.imwrite('dst1898.jpg', dst)

        image_name = 'dst1898.jpg'
        features = 'feature_points_with_annotations.csv'
        camera_locations = 'potential_camera_locations.csv'
        pixel_x = 'Pixel_x_1898.jpg'
        pixel_y = 'Pixel_y_1898.jpg'
        output = 'zOutput_1898.png'
        scale = 1.0
        dem_file = 'dem_data.tif'


elif img == '1900-1910':
    ret, mtx, dist, rvecs, tvecs = calibrate_camera(23)
    img = cv2.imread('1900-1910.jpg')
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)  # un-distort
    cv2.imwrite('dst1900-1910.png', dst)

    image_name = 'dst1900-1910.png'
    features = 'feature_points_with_annotations.csv'
    camera_locations = 'potential_camera_locations.csv'
    pixel_x = 'Pixel_x_1900-1910.jpg'
    pixel_y = 'Pixel_y_1900-1910.jpg'
    output = 'zOutput_1900-1910.png'
    scale = 1.0

elif img == '1910':
    ret, mtx, dist, rvecs, tvecs = calibrate_camera(23)
    img = cv2.imread('1910.jpg')
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)  # un-distort
    cv2.imwrite('dst1910.png', dst)

    image_name = 'dst1910.png'
    features = 'feature_points_with_annotations.csv'
    camera_locations = 'potential_camera_locations.csv'
    pixel_x = 'Pixel_x_1910.jpg'
    pixel_y = 'Pixel_y_1910.jpg'
    output = 'zOutput_1910.png'
    scale = 1.0


elif img == '1912s':
    image_name = '1912s.jpg'
    features = 'feature_points_with_annotations.csv'
    camera_locations = 'potential_camera_locations.csv'
    pixel_x = 'Pixel_x_1912s.jpg'
    pixel_y = 'Pixel_y_1912s.jpg'
    output = 'zOutput_1912s.png'
    scale = 1.0


elif img == '1915 (2)':
    image_name = '1915 (2).jpg'
    features = 'feature_points_with_annotations.csv'
    camera_locations = 'potential_camera_locations.csv'
    pixel_x = 'Pixel_x_1915 (2).jpg'
    pixel_y = 'Pixel_y_1915 (2).jpg'
    output = 'zOutput_1915 (2).png'
    scale = 1.0


elif img == '1915':
    image_name = '1915.jpg'
    features = 'feature_points_with_annotations.csv'
    camera_locations = 'potential_camera_locations.csv'
    pixel_x = 'Pixel_x_1915.jpg'
    pixel_y = 'Pixel_y_1915.jpg'
    output = 'zOutput_1915.png'
    scale = 1.0


elif img == '1920-1930':
    image_name = '1920-1930.jpg'
    features = 'feature_points_with_annotations.csv'
    camera_locations = 'potential_camera_locations.csv'
    pixel_x = 'Pixel_x_1920-1930.jpg'
    pixel_y = 'Pixel_y_1920-1930.jpg'
    output = 'zOutput_1920-1930.png'
    scale = 1.0


elif img == '1925-1930':
    image_name = '1925-1930.jpg'
    features = 'feature_points_with_annotations.csv'
    camera_locations = 'potential_camera_locations.csv'
    pixel_x = 'Pixel_x_1925-1930.jpg'
    pixel_y = 'Pixel_y_1925-1930.jpg'
    output = 'zOutput_1925-1930.png'
    scale = 1.0


elif img == '1930':
    image_name = '1930.jpg'
    features = 'feature_points_with_annotations.csv'
    camera_locations = 'potential_camera_locations.csv'
    pixel_x = 'Pixel_x_1930.jpg'
    pixel_y = 'Pixel_y_1930.jpg'
    output = 'zOutput_1930.png'
    scale = 1.0


elif img == 'center of the settlement kuliang':
    image_name = 'center of the settlement kuliang.jpg'
    features = 'feature_points_with_annotations.csv'
    camera_locations = 'potential_camera_locations.csv'
    pixel_x = 'Pixel_x_center of the settlement kuliang.jpg'
    pixel_y = 'Pixel_y_center of the settlement kuliang.jpg'
    output = 'zOutput_center of the settlement kuliang.png'
    scale = 1.0


elif img == 'kuliang hills':
    image_name = 'kuliang hills.jpg'
    features = 'feature_points_with_annotations.csv'
    camera_locations = 'potential_camera_locations.csv'
    pixel_x = 'Pixel_x_kuliang hills.jpg'
    pixel_y = 'Pixel_y_kuliang hills.jpg'
    output = 'zOutput_kuliang hills.png'
    scale = 1.0


elif img == 'kuliang panorama central segment':
    image_name = 'kuliang panorama central segment.jpg'
    features = 'feature_points_with_annotations.csv'
    camera_locations = 'potential_camera_locations.csv'
    pixel_x = 'Pixel_x_kuliang panorama central segment.jpg'
    pixel_y = 'Pixel_y_kuliang panorama central segment.jpg'
    output = 'zOutput_kuliang panorama central segment.png'
    scale = 1.0


elif img == 'kuliang Pine Crag road':
    image_name = 'kuliang Pine Crag road.jpg'
    features = 'feature_points_with_annotations.csv'
    camera_locations = 'potential_camera_locations.csv'
    pixel_x = 'Pixel_x_kuliang Pine Crag road.jpg'
    pixel_y = 'Pixel_y_kuliang Pine Crag road.jpg'
    output = 'zOutput_kuliang Pine Crag road.png'
    scale = 1.0


elif img == 'Siems Siemssen':
    image_name = 'Siems Siemssen.jpg'
    features = 'feature_points_with_annotations.csv'
    camera_locations = 'potential_camera_locations.csv'
    pixel_x = 'Pixel_x_Siems Siemssen.jpg'
    pixel_y = 'Pixel_y_Siems Siemssen.jpg'
    output = 'zOutput_Siems Siemssen.png'
    scale = 1.0


elif img == 'View Kuliang includes tennis courts':
    image_name = 'View Kuliang includes tennis courts.jpg'
    features = 'feature_points_with_annotations.csv'
    camera_locations = 'potential_camera_locations.csv'
    pixel_x = 'Pixel_x_View Kuliang includes tennis courts.jpg'
    pixel_y = 'Pixel_y_View Kuliang includes tennis courts.jpg'
    output = 'zOutput_View Kuliang includes tennis courts.png'
    scale = 1.0


elif img == 'Worley Family-20':
    image_name = 'Worley Family-20.jpg'
    features = 'feature_points_with_annotations.csv'
    camera_locations = 'potential_camera_locations.csv'
    pixel_x = 'Pixel_x_Worley Family-20.jpg'
    pixel_y = 'Pixel_y_Worley Family-20.jpg'
    output = 'zOutput_Worley Family-20.png'
    scale = 1.0


else:
    print('No file was selected')

do_it(image_name, features, pixel_x, pixel_y, output, scale, dem_file)

print('**********************')
# print ('ret: ')
# print (ret)
# print ('mtx: ')
# print (mtx)
# print ('dist: ')
# print (dist)
# print('rvecs: ')
# print(rvecs)
# print ('tvecs: ')
# print(tvecs)

print('Done!')
