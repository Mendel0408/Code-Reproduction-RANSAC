# -*- coding: utf-8 -*-

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import font_manager
from mpl_toolkits.mplot3d import Axes3D
import csv
import glob
import math
import logging
import pyproj
from pyproj import Transformer
from scipy.spatial import distance
import seaborn as sns
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import least_squares
from osgeo import gdal

# 设置字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题

# 验证字体是否存在
font_path = font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
print("Available fonts:", font_path)

# 设置日志记录
logging.basicConfig(level=logging.DEBUG, filename='debug.log', filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')

# 坐标系统统一化
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

geo_transformer = GeoCoordTransformer()

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
    dem_dataset = gdal.Open(dem_file)
    if dem_dataset is None:
        raise RuntimeError(f"无法加载 DEM 文件: {dem_file}")

    dem_array = dem_dataset.ReadAsArray()
    gt = dem_dataset.GetGeoTransform()
    dem_x = np.arange(dem_array.shape[1]) * gt[1] + gt[0]
    dem_y = np.arange(dem_array.shape[0]) * gt[5] + gt[3]

    dem_interpolator = RegularGridInterpolator((dem_y, dem_x), dem_array)
    dem_data = {
        'interpolator': dem_interpolator,
        'x_range': (dem_x.min(), dem_x.max()),
        'y_range': (dem_y.min(), dem_y.max()),
        'data': dem_array
    }
    logging.debug(f'DEM 范围: 经度 {dem_data["x_range"]}, 纬度 {dem_data["y_range"]}')
    return dem_data

# 使用PnP算法进行相机姿态估计
def estimate_camera_pose(pos3d, pixels, K):
    pos3d = np.asarray(pos3d, dtype=np.float64).reshape(-1, 3)
    pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    print("3D points:\n", pos3d)
    print("2D points:\n", pixels)
    print("Camera matrix K:\n", K)

    # 可视化3D点
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(pos3d[:, 0], pos3d[:, 1], pos3d[:, 2], c='r', marker='o')
    ax.set_title('3D Points')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 可视化2D点
    ax2 = fig.add_subplot(122)
    ax2.scatter(pixels[:, 0], pixels[:, 1], c='b', marker='x')
    ax2.set_title('2D Points')
    ax2.set_xlabel('Pixel X')
    ax2.set_ylabel('Pixel Y')
    ax2.invert_yaxis()  # 图像坐标系的Y轴是向下的

    plt.show()

    # 使用PnP算法估计旋转向量和平移向量，并返回内点
    success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(
        pos3d, pixels, K, dist_coeffs,
        iterationsCount=5000,
        reprojectionError=30.0,
        confidence=0.99
    )
    print("Inliers:\n", inliers)
    if not success or inliers is None or len(inliers) < 6:
        print("PnP RANSAC failed or insufficient inliers.")
        return None, None, None

    rotation_vector, translation_vector = cv2.solvePnPRefineLM(pos3d[inliers], pixels[inliers], K, dist_coeffs,
                                                               rotation_vector, translation_vector)
    print(f"Rotation Vector (R):\n{rotation_vector}")
    print(f"Translation Vector (T):\n{translation_vector}")
    return rotation_vector, translation_vector, inliers

# 检查并调整translation_vector的值
def check_translation_vector(translation_vector):
    max_value = 1e4  # 根据实际情况调整阈值
    if np.any(np.abs(translation_vector) > max_value):
        logging.warning(f"Translation vector values are too large: {translation_vector}")
        translation_vector = np.clip(translation_vector, -max_value, max_value)
    return translation_vector

# 获取DEM数据中的海拔值
def get_dem_elevation(dem_data, easting, northing):
    """
    将 UTM 坐标转换为 WGS84，经纬度，并从 DEM 插值器中查询海拔。
    """
    lon, lat = geo_transformer.utm_to_wgs84(easting, northing)
    # 插值器构造时使用的坐标顺序为 (lat, lon)
    dem_elev = dem_data['interpolator']((lat, lon))
    return dem_elev

def recalculate_translation_vector(corrected_camera_position_wgs84, R):
    """
    根据修正后的相机位置重新计算平移向量 T
    :param corrected_camera_position_wgs84: 修正后的相机位置 (lon, lat, height)
    :param R: 相机旋转矩阵
    :return: 修正后的平移向量 T
    """
    # 将修正后的相机位置转换为 UTM 坐标
    corrected_position_utm = np.array(geo_transformer.wgs84_to_utm(corrected_camera_position_wgs84[0], corrected_camera_position_wgs84[1]) + (corrected_camera_position_wgs84[2],))

    # 计算平移向量 T
    T = -R @ corrected_position_utm
    return T

# 将像素坐标转换为射线
def pixel_to_ray(pixel_x, pixel_y, K, R, ray_origin):
    """
    计算从相机位置到像素坐标 (pixel_x, pixel_y) 的射线方向，并转换到 WGS84 坐标系

    参数:
        pixel_x, pixel_y: 图像中的像素坐标
        K: 相机内参矩阵 (3x3)
        R: 相机旋转矩阵 (3x3)
        ray_origin: 相机在WGS84坐标系中的位置 (lon, lat, height)

    返回:
        ray_origin: WGS84 下的射线起点 (lon, lat, height)
        ray_direction: WGS84 下的射线方向 (X, Y 以度计算，Z 以米计算)
    """
    # 构建齐次像素坐标
    pixel_homogeneous = np.array([pixel_x, pixel_y, 1.0], dtype=np.float64)

    # 计算相机坐标系中的方向向量
    camera_ray = np.linalg.inv(K) @ pixel_homogeneous
    camera_ray[2] = 1.0  # 确保Z分量不过小
    camera_ray /= np.linalg.norm(camera_ray)  # 归一化

    # 旋转到世界坐标系
    ray_direction = R.T @ camera_ray
    ray_direction /= np.linalg.norm(ray_direction)  # 归一化

    # 调试输出
    print(f"【DEBUG】pixel_homogeneous : {pixel_homogeneous}")
    print(f"【DEBUG】camera_ray (UTM): {camera_ray}")
    print(f"【DEBUG】ray_origin (UTM): {ray_origin}")
    print(f"【DEBUG】Computed UTM ray_direction (单位向量，各分量单位为米): {ray_direction}")
    return ray_origin, ray_direction

# 新增函数：根据像素坐标计算射线方向（不输出调试信息），用于控制点部分
def compute_ray_direction_from_pixel(pixel, K, R, ray_origin):
    pixel_homogeneous = np.array([pixel[0], pixel[1], 1.0], dtype=np.float64)
    camera_ray = np.linalg.inv(K) @ pixel_homogeneous
    camera_ray[2] = 1.0
    camera_ray = camera_ray / np.linalg.norm(camera_ray)
    ray_direction = R.T @ camera_ray
    ray_direction = ray_direction / np.linalg.norm(ray_direction)
    return ray_direction

# 计算射线与DEM的交点
def ray_intersect_dem_with_scales(ray_origin, ray_direction, scales, dem_data, max_search_dist=10000, step=0.001):
    """
    ray_origin: WGS84坐标 (lon, lat, height)
    dem_data: 存储DEM数据的字典，包含x_range(经度范围)和y_range(纬度范围)
    ray_direction: 射线方向，单位向量（X, Y以度，Z以米）
    scales: 比例因子数组 [s_x, s_y, s_z]
    dem_data: DEM数据字典，包含 'interpolator', 'x_range', 'y_range'
    max_search_dist: 最大搜索距离（单位与 X, Y 坐标相同）
    step: 步进距离
    """
    current_pos = np.array(ray_origin, dtype=np.float64)
    for _ in range(10000):
        print(f"【DEBUG】当前UTM坐标: {current_pos}, 当前射线方向: {ray_direction}")
        current_easting = current_pos[0]
        current_northing = current_pos[1]
        if (dem_data['x_range'][0] <= current_easting <= dem_data['x_range'][1] and
                dem_data['y_range'][0] <= current_northing <= dem_data['y_range'][1]):
            # 插值时的顺序为 (northing, easting)
            dem_elev = dem_data['interpolator']((current_northing, current_easting))
            print(f"【DEBUG】DEM海拔: {dem_elev}, 当前高度: {current_pos[2]}")
        else:
            print(f"【错误】UTM坐标 ({current_easting:.2f}, {current_northing:.2f}) 超出DEM范围")
            return None

        if current_pos[2] <= dem_elev:
            return np.array([current_northing, current_easting, current_pos[2]])

        # 更新当前坐标
        current_pos[0] += step * ray_direction[0] * scales[0]
        current_pos[1] += step * ray_direction[1] * scales[1]
        current_pos[2] += step * ray_direction[2] * scales[2]
        if np.linalg.norm(current_pos[:2] - np.array(ray_origin[:2])) > max_search_dist:
            break

    return None

# --- 新增函数：利用内点构造残差函数进行比例因子优化 ---
# 使用相机位置(ray_origin)与控制点真实3D坐标之差构成理想射线方向，
# 并与 pixel_to_ray 得到的射线方向做元素乘法校正，计算各控制点的残差（向量差），
# 最后拼接所有残差向量返回给最小二乘优化模块。
def residual_scales_control_points(scales, control_points, K, R, ray_origin):
    """
    对每个控制点（包含图像像素 'pixel' 和UTM下的3D点 'pos3d'）执行下列操作：
      1. 计算理想射线方向：ideal_direction = normalize(control_point_3d - ray_origin)
      2. 通过像素计算射线方向（在UTM下计算），并应用比例因子校正；
      3. 返回校正后的射线方向与理想射线方向之间的向量差作为残差，用于最小二乘优化。
    """
    errors = []
    for cp in control_points:
        true_geo = np.array(cp['pos3d'], dtype=np.float64)
        ideal_direction = true_geo - ray_origin
        norm_ideal = np.linalg.norm(ideal_direction)
        if norm_ideal == 0:
            continue
        ideal_direction /= norm_ideal
        print(f"【DEBUG】控制点 {cp.get('symbol', '未知')} 的理想UTM射线方向: {ideal_direction}")
        _, computed_ray = pixel_to_ray(cp['pixel'][0], cp['pixel'][1], K, R, ray_origin)
        corrected_ray = np.array([
            scales[0] * computed_ray[0],
            scales[1] * computed_ray[1],
            scales[2] * computed_ray[2]
        ])
        norm_corr = np.linalg.norm(corrected_ray)
        if norm_corr == 0:
            continue
        corrected_ray /= norm_corr

        errors.append(corrected_ray - ideal_direction)
    if errors:
        return np.concatenate(errors)
    else:
        return np.array([])

# 输入像素坐标，输出地理坐标
def pixel_to_geo(pixel_coord, K, rotation_vector, translation_vector, ray_origin, dem_interpolator, dem_x, dem_y):
    ray_origin, ray_direction = pixel_to_ray(pixel_coord[0], pixel_coord[1], K, rotation_vector, ray_origin)

    # 调试信息
    print(f"【DEBUG】ray_origin 形状: {ray_origin.shape}, ray_origin 值: {ray_origin}")
    print(f"【DEBUG】ray_direction 形状: {ray_direction.shape}, ray_direction 值: {ray_direction}")

    geo_coord = ray_intersect_dem(ray_origin, ray_direction, dem_interpolator, dem_x, dem_y)
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
                    easting, northing = geo_transformer.wgs84_to_utm(longitude, latitude)  # 注意顺序
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
                    easting, northing = geo_transformer.wgs84_to_utm(longitude, latitude)  # 注意顺序
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

    plt.figure(figsize=(11.69, 8.27))
    plt.imshow(im)

    recs = read_points_data(features, pixel_x, pixel_y, scale)
    locations = read_camera_locations()
    for rec in recs:
        symbol = rec['symbol']
        pixel = rec['pixel']
        if pixel[0] != 0 or pixel[1] != 0:
            plt.text(pixel[0], pixel[1], symbol, color='red', fontsize=6)
    num_matches12 = find_homographies(recs, locations, im, False, 75.0, output)
    num_matches2 = num_matches12[:, 1]
    num_matches2[num_matches2 == 0] = 1000000
    print(np.min(num_matches2))
    theloci = np.argmin(num_matches2)
    print(f"【DEBUG】最佳相机位置 (UTM): {locations[theloci]['pos3d']}")

    # 设置 K 矩阵
    width, height = im.shape[1], im.shape[0]
    cx, cy = width / 2, height / 2
    fx = 2529
    fy = 1365
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    print(f"【DEBUG】K 矩阵: \n{K}")

    # 加载DEM数据（DEM范围为UTM）
    dem_data = load_dem_data(dem_file)

    # 使用PnP算法估计 R 和 T，得到的相机位置camera_origin为UTM数据
    pos3d = np.array([rec['pos3d'] for rec in recs])
    pixels = np.array([rec['pixel'] for rec in recs])
    R, t, inliers = estimate_camera_pose(pos3d, pixels, K)
    if R is None or t is None:
        print("Failed to estimate camera pose using PnP.")
        return
    R, _ = cv2.Rodrigues(R)
    inlier_indices = inliers.flatten().tolist()
    control_points = [
        {
            'pixel': recs[idx]['pixel'],
            'pos3d': recs[idx]['pos3d'],
            'dem_data': dem_data,
            'symbol': recs[idx]['symbol']
        }
        for idx in inlier_indices
    ]
    print("【DEBUG】控制点数据 (UTM):")
    for cp in control_points:
        print(cp)

    # 使用PnP求解后的相机位置（camera_origin为UTM坐标）
    camera_origin = -R.T @ t.flatten()
    print(f"【DEBUG】PnP求解的相机位置 (UTM): {camera_origin}")

    # --- DEM-based elevation correction ---
    # The PnP solution typically gives an altitude that is too high.
    # Correct it by obtaining the DEM elevation at the camera's (E, N) position,
    # then update the camera origin (adding an observer offset; here we use 1.5 m).
    dem_elev = get_dem_elevation(dem_data, camera_origin[0], camera_origin[1])
    corrected_camera_origin = np.array([camera_origin[0], camera_origin[1], dem_elev + 1.5])
    print(f"【DEBUG】修正后的相机位置 (UTM): {corrected_camera_origin}")

    # 直接使用修正后的camera_origin作为ray_origin
    ray_origin = corrected_camera_origin
    print(f"【DEBUG】用于射线计算的 ray_origin (UTM): {ray_origin}")

    # 将 DEM 范围（WGS84）转换为 UTM 坐标
    # dem_data['x_range'] 为经度范围，dem_data['y_range'] 为纬度范围
    lon_min, lon_max = dem_data['x_range']
    lat_min, lat_max = dem_data['y_range']
    utm_x1, utm_y1 = geo_transformer.to_utm.transform(lon_min, lat_min)
    utm_x2, utm_y2 = geo_transformer.to_utm.transform(lon_max, lat_max)
    utm_x_range = (min(utm_x1, utm_x2), max(utm_x1, utm_x2))
    utm_y_range = (min(utm_y1, utm_y2), max(utm_y1, utm_y2))

    tol = 1e-5
    # 验证相机位置的 X,Y 坐标（UTM 平面坐标）是否在 DEM 的 UTM 覆盖范围内
    if not (utm_x_range[0] - tol <= camera_origin[0] <= utm_x_range[1] + tol and
            utm_y_range[0] - tol <= camera_origin[1] <= utm_y_range[1] + tol):
        print(f"【错误】ray_origin {camera_origin} 超出 DEM 范围")
        print(f"【DEBUG】DEM 范围 (UTM): 经度 {utm_x_range}, 纬度 {utm_y_range}")
        return

    while True:
        try:
            input_pixel = input("请输入像素坐标 (x, y) 或输入 'exit' 退出: ").strip()
            if input_pixel.lower() == 'exit':
                break

            pixel_values = input_pixel.replace(" ", "").replace("，", ",").split(",")
            if len(pixel_values) != 2:
                print("输入格式错误，例：755,975")
                continue

            input_pixel_x, input_pixel_y = map(float, pixel_values)
            print(f"【DEBUG】转换为浮点数: x={input_pixel_x}, y={input_pixel_y}")

            ray_origin_utm, ray_direction = pixel_to_ray(input_pixel_x, input_pixel_y, K, R, ray_origin)
            print(f"【DEBUG】ray_origin (UTM) from pixel_to_ray: {ray_origin_utm}")
            print(f"【DEBUG】初始射线方向 (UTM): {ray_direction}")

            initial_scales = np.array([1.0, 1.0, 1.0])
            result = least_squares(
                lambda s: residual_scales_control_points(s, control_points, K, R, ray_origin_utm),
                x0=initial_scales)
            optimized_scales = result.x
            print(f"【DEBUG】优化后的比例因子 (基于控制点内点): {optimized_scales}")

            geo_coord = ray_intersect_dem_with_scales(ray_origin_utm, ray_direction, optimized_scales, dem_data)
            if geo_coord is not None:
                print(f"像素坐标 ({input_pixel_x}, {input_pixel_y}) 对应的UTM地理坐标:")
                print(f"Easting: {geo_coord[0]:.2f}, Northing: {geo_coord[1]:.2f}, 高度: {geo_coord[2]:.2f}")
            else:
                print(f"无法找到 ({input_pixel_x}, {input_pixel_y}) 对应的地理坐标，请检查输入或 DEM 数据。")

        except ValueError as e:
            print(f"输入格式错误: {e}")
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
