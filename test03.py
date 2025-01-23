import numpy as np
import cv2
import matplotlib.pyplot as plt
import csv
import glob
import math
import pyproj

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


def solve_pnp_ransac(recs, camera_matrix, dist_coeffs, ransac_threshold=3.0):
    """
    使用PnP-RANSAC求解相机位姿
    """
    object_points = []
    image_points = []
    for rec in recs:
        if rec['pixel'][0] != 0 or rec['pixel'][1] != 0:
            object_points.append(rec['pos3d'])
            image_points.append(rec['pixel'])
    
    object_points = np.array(object_points, dtype=np.float32)
    image_points = np.array(image_points, dtype=np.float32)
    
    # 使用RANSAC求解PnP
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=object_points,
        imagePoints=image_points,
        cameraMatrix=camera_matrix,
        distCoeffs=dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
        reprojectionError=ransac_threshold,
        iterationsCount=1000
    )
    
    if success:
        # 将旋转向量转换为旋转矩阵
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        return {
            'rvec': rvec,
            'tvec': tvec,
            'rotation_matrix': rotation_matrix,
            'inliers': inliers
        }
    else:
        return None


# **********
# read data from the features file
# **********
def read_points_data(filename, pixel_x, pixel_y, scale):

    wgs84 = pyproj.CRS('EPSG:4326')
    utm_zone = None
    utm_epsg = None

    with open(filename, encoding= 'utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        recs = []
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
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

                if utm_zone is None:
                    utm_zone = int((longitude + 180) // 6 + 1)
                    hemisphere = 'north' if latitude >= 0 else 'south'
                    utm_epsg = 32600 + utm_zone if hemisphere == 'north' else 32700 + utm_zone
                    utm_crs = pyproj.CRS(f'EPSG:{utm_epsg}')
                    transformer = pyproj.Transformer.from_crs(wgs84, utm_crs)

                easting, northing = transformer.transform(longitude, latitude)
                pos3d = np.array([easting, northing, elevation])


                rec = {'symbol': symbol,
                       'pixel': pixel,
                       'pos3d': pos3d,
                       'name': name
                }
                recs.append(rec)


        print(f'Processed {line_count} lines.')
        return recs, utm_epsg


# **********
# read data from the potential camera locations file
# **********
def read_camera_locations(camera_locations_file, utm_epsg):

    wgs84 = pyproj.CRS('EPSG:4326')
    utm_crs = pyproj.CRS(f'EPSG:326{utm_epsg}')
    transformer = pyproj.Transformer.from_crs(wgs84, utm_crs)

    with open(camera_locations, encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        recs = []
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                line_count += 1
                grid_code = int(row[1])
                longitude = float(row[2])
                latitude = float(row[3])
                height = float(row[4]) + 2.0  # addition of 2 meters  as the observer height

                easting, northing = transformer.transform(longitude, latitude)
                pos3d = np.array([easting, northing, height])

                rec = {'grid_code': grid_code,
                       'pos3d': pos3d}
                recs.append(rec)
        print(f'Processed {line_count} lines.')
        return recs


# **********
# Main function
# **********
def do_it(image_name, features, pixel_x, pixel_y, output, scale, camera_matrix, dist_coeffs):
    # 读取图像
    im = cv2.imread(image_name)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    
    # 读取特征点和相机内参
    recs, utm_epsg = read_points_data(features, pixel_x, pixel_y, scale)
    
    #
    camera_locations = read_camera_locations('potential_camera_locations.csv', utm_epsg)

    camera_matrix = camera_matrix
    dist_coeffs = dist_coeffs
    
    # 使用PnP-RANSAC求解相机位姿
    pnp_result = solve_pnp_ransac(recs, camera_matrix, dist_coeffs, ransac_threshold=5.0)
    
    if pnp_result is not None:
        rvec = pnp_result['rvec']
        tvec = pnp_result['tvec']
        inliers = pnp_result['inliers']
        
        # 可视化重投影结果
        plt.figure(figsize=(40, 20))
        plt.imshow(im)
        for i, rec in enumerate(recs):
            pixel = rec['pixel']
            if pixel[0] != 0 or pixel[1] != 0:
                # 计算重投影坐标
                projected_point, _ = cv2.projectPoints(
                    rec['pos3d'].reshape(1, 3),
                    rvec,
                    tvec,
                    camera_matrix,
                    dist_coeffs
                )
                proj_x, proj_y = projected_point[0][0]
                
                # 绘制实际点和重投影点
                color = 'green' if i in inliers else 'red'
                plt.plot([pixel[0], proj_x], [pixel[1], proj_y], color=color, linewidth=5)
                plt.scatter(pixel[0], pixel[1], color=color, s=100, marker='x')
                plt.scatter(proj_x, proj_y, color=color, s=100, marker='o')
        
        plt.savefig(output, dpi=300)
        plt.close()
        
        # 输出相机位姿
        print("相机位姿 (旋转矩阵):\n", pnp_result['rotation_matrix'])
        print("相机位姿 (平移向量):\n", tvec)
        print("内点数量:", len(inliers))
    else:
        print("PnP求解失败！")


img = '1898'
# img = '1900-1910'
# img = '1910'
# img = '1912s'
# img = '1915(2)'

camera_locations = ''
grid_code_min = 7

if img == '1898':
    ret, mtx, dist, rvecs, tvecs = calibrate_camera(30)
    if not ret:
        print("标定失败，无法继续执行！")
        exit()

    print("相机内参矩阵（mtx）：\n", mtx)

    img = cv2.imread('1898.jpg')
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)  # un-distort
    cv2.imwrite('undistorted_1898.jpg', dst)

    image_name = 'tmp1898.jpg'
    features = 'feature_points_with_annotations.csv'
    camera_locations = 'potential_camera_locations.csv'
    pixel_x = 'Pixel_x_1898.jpg'
    pixel_y = 'Pixel_y_1898.jpg'
    output = 'zOutput_1898.jpg'
    scale = 1.0

else:
    print('No file was selected')

do_it(image_name, features, pixel_x, pixel_y, output, scale, mtx, dist)

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