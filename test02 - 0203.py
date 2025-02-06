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
transformer = Transformer.from_crs("epsg:4326", "epsg:32633")  # 这里使用 UTM zone 33N，具体 zone 需要根据实际情况调整

def wgs84_to_utm(lat, lon):
    easting, northing = transformer.transform(lat, lon)
    return easting, northing

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
    transformer_to_wgs84 = Transformer.from_crs("epsg:32633", "epsg:4326")
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
    transformer_to_wgs84 = Transformer.from_crs("epsg:32633", "epsg:4326")
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
            num_matches[i, 0], num_matches[i, 1] = find_homography(recs, pixels, pos3ds, symbols, loc3ds[i], im, show,
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
    pos2 = np.zeros((pixels.shape[0], 2))
    good = np.zeros(pixels.shape[0])
    for i in range(pixels.shape[0]):
        good[i] = pixels[i, 0] != 0 or pixels[i, 1] != 0
        p = pos3ds[i, :] - camera_location
        p = np.array([p[2], p[1], p[0]])
        p = p / p[2]
        pos2[i, :] = p[0:2]
    M, mask = cv2.findHomography(pos2[good == 1], pixels[good == 1], cv2.RANSAC, ransacbound)
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
                # plt.text(pixel[0],pixel[1],symbol, style='italic',fontsize=30, weight ='bold', bbox=dict(boxstyle="round", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8),))
    err1 = 0
    err2 = 0
    feature = ['id', 'symbol', 'name', 'x', 'y', 'pixel_x', 'pixel_y', 'calc_pixel_x', 'calc_pixel_y']
    features = []
    features.append(feature)
    for i in range(pos2[good == 1].shape[0]):
        p1 = pixels[good == 1][i, :]
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
    return err1, err2


# 读取DEM数据
def load_dem_data(dem_file):
    dataset = gdal.Open(dem_file)
    dem_data = dataset.ReadAsArray()
    gt = dataset.GetGeoTransform()
    dem_x = np.arange(dem_data.shape[1]) * gt[1] + gt[0]
    dem_y = np.arange(dem_data.shape[0]) * gt[5] + gt[3]
    dem_interpolator = RegularGridInterpolator((dem_y, dem_x), dem_data)
    return dem_interpolator


# 分解单应性矩阵，得到内参矩阵和外参矩阵
def decompose_homography(H):
    K, R, t, _ = cv2.decomposeHomographyMat(H, np.eye(3))
    return K, R[0], t[0]


# 使用PnP算法进行相机姿态估计
def estimate_camera_pose(world_coords, pixels, K):
    success, rotation_vector, translation_vector = cv2.solvePnP(world_coords, pixels, K, np.zeros(4))
    if not success:
        raise RuntimeError("PnP算法计算失败")
    return rotation_vector, translation_vector


# 将像素坐标转换为射线
def pixel_to_ray(pixel_coord, K, rotation_vector, translation_vector):
    pixel_coord_homogeneous = np.append(pixel_coord, 1).reshape(-1, 1)
    inv_K = np.linalg.inv(K)
    normalized_coord = np.dot(inv_K, pixel_coord_homogeneous)
    normalized_coord = normalized_coord / np.linalg.norm(normalized_coord)

    R, _ = cv2.Rodrigues(rotation_vector)
    T = translation_vector.reshape(-1, 1)

    ray_origin = T.ravel()
    ray_direction = np.dot(R.T, normalized_coord).ravel()
    return ray_origin, ray_direction


# 计算射线与DEM的交点
def ray_intersect_dem(ray_origin, ray_direction, dem_interpolator):
    t_values = np.linspace(0, 10000, 1000)
    intersection = None
    for t in t_values:
        point = ray_origin + t * ray_direction
        dem_height = dem_interpolator([point[1], point[0]])[0]  # 注意坐标顺序
        if point[2] <= dem_height:
            intersection = point
            break
    return intersection


# 输入像素坐标，输出地理坐标
def pixel_to_geo(pixel_coord, K, rotation_vector, translation_vector, dem_interpolator):
    ray_origin, ray_direction = pixel_to_ray(pixel_coord, K, rotation_vector, translation_vector)
    geo_coord = ray_intersect_dem(ray_origin, ray_direction, dem_interpolator)
    return geo_coord

# **********
# read data from the features file
# **********
def read_points_data(filename, pixel_x, pixel_y, scale):
    with open(filename, encoding= 'utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        recs = []
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
                # 添加坐标转换
                easting, northing = wgs84_to_utm(latitude, longitude)
                pos3d = np.array([easting, northing, height])

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
                height = float(row[4]) + 2.0  # addition of 2 meters  as the observer height
                # 添加坐标转换
                easting, northing = wgs84_to_utm(latitude, longitude)
                pos3d = np.array([easting, northing, height])

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

    num_matches12 = find_homographies(recs, locations, im, False, 75.0, output)
    num_matches2 = num_matches12[:, 1]
    # print(np.min(num_matches2[num_matches2 > 0]))
    # print(np.max(num_matches2[num_matches2 > 0]))

    num_matches2[num_matches2 == 0] = 1000000
    print(np.min(num_matches2))

    theloci = np.argmin(num_matches2)  # theloci contains the best location for the camera
    best_location = locations[theloci]['pos3d']
    print('location id: ' + str(theloci) + ' - ' + str(locations[theloci]))

    find_homographies(recs, [locations[theloci]], im, True, 75.0, output)  # Orig = 120.0

    # 使用现有代码计算的单应性矩阵
    best_homography_matrix = find_homography(recs, pixels, np.array([rec['pos3d'] for rec in recs]),
                                             np.array([rec['symbol'] for rec in recs]), best_location, im, False, 75.0, output)

    # 分解单应性矩阵
    K, R, t = decompose_homography(best_homography_matrix)
    rotation_vector, translation_vector = estimate_camera_pose(np.array([rec['pos3d'] for rec in recs]), pixels, K)

    # 读取DEM数据
    dem_interpolator = load_dem_data(dem_file)

    # 交互式输入像素坐标
    while True:
        try:
            input_pixel_x = input("Enter pixel X coordinate (or type 'exit' to quit): ")
            if input_pixel_x.lower() == 'exit':
                break
            input_pixel_x = float(input_pixel_x)
            input_pixel_y = float(input("Enter pixel Y coordinate: "))
            input_pixel = np.array([input_pixel_x, input_pixel_y])
            geo_coord = pixel_to_geo(input_pixel, K, rotation_vector, translation_vector, dem_interpolator)
            print(f"Geographic Coordinate: {geo_coord}")
        except ValueError:
            print("Invalid input. Please enter numeric values.")

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
