import numpy as np
import cv2 
import matplotlib.pyplot as plt
import csv
import glob
import math


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
                dis_m = math.sqrt(math.pow(dis_m_x,2) + math.pow(dis_m_y,2))

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
                dis_c_pix = math.sqrt(math.pow(dis_c_pix_x,2) + math.pow(dis_c_pix_y,2))

                bear_pix = calc_bearing(f1[5], f1[6], features[i][5], features[i][6])
                if bear_pix != 0 and bear_pix <= 180:
                    dis_depth_pix = (abs(bear_pix-90)/90 + depth_val) * dis_pix
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

                result = [str(count), f1[1], f1[3], f1[4], f1[5], f1[6], f1[7], f1[8],features[i][1], features[i][3],
                          features[i][4], features[i][5], features[i][6], features[i][7], features[i][8],
                          dis_m_x, dis_m_y, dis_m, dis_pix_x, dis_pix_y ,dis_pix, dis_c_pix_x, dis_c_pix_y, dis_c_pix,
                          bear_pix, dis_depth_pix, bear_c_pix, dis_depth_c_pix]

                results.append(result)
                count += 1
            i += 1
        j += 1
    return results


# **********
# Calculation of the bearing from point 1 to point 2
# **********
def calc_bearing (x1, y1, x2, y2):
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

    objpoints = []   # Creating vector to store vectors of 3D points for each checkerboard image
    imgpoints = []   # Creating vector to store vectors of 2D points for each checkerboard image

    # Defining the world coordinates for 3D points
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    prev_img_shape = None

    images = glob.glob('.\camera_calibration\images\*.jpg')   # TODO: change the path according to the path in your environmrnt
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
    num_matches = np.zeros((loc3ds.shape[0],2))
    scores = []
    for i in range(0, grids.shape[0], 1):  # 50
        if grids[i] >= grid_code_min:
            if show:
                print(i,grids[i],loc3ds[i])
            num_matches[i, 0], num_matches[i, 1] = find_homography(recs, pixels, pos3ds, symbols, loc3ds[i], im, show, ransacbound, outputfile)
        else:
            num_matches[i, :] = 0
        score = [i+1, num_matches[i, 0], num_matches[i, 1], grids[i], loc3ds[i][0], loc3ds[i][1], loc3ds[i][2]]
        scores.append(score)

    if show is False:
        outputCsv = output.replace(".png","_location.csv")
        csvFile = open(outputCsv, 'w', newline='', encoding='utf-8')
        csvWriter = csv.writer(csvFile)
        csvWriter.writerow(['location_id', 'min_score', 'max_score', 'grid_code', 'Z', 'X', 'Y'])
        for s in scores:
            csvWriter.writerow(s)

    return num_matches


# **********
# Find homography function
# **********
def find_homography(recs, pixels, pos3ds, symbols, camera_location, im, show, ransacbound, outputfile):
    pos2 = np.zeros((pixels.shape[0],2))
    good = np.zeros(pixels.shape[0])
    for i in range(pixels.shape[0]):
        good[i] = pixels[i,0]!=0 or pixels[i,1]!=0
        p = pos3ds[i,:] - camera_location
        p = np.array([p[2],p[1],p[0]])
        p = p/p[2]
        pos2[i,:]=p[0:2]
    M, mask = cv2.findHomography(pos2[good==1],pixels[good==1], cv2.RANSAC,ransacbound)
    M = np.linalg.inv(M)
    if show:
        print('M',M,np.sum(mask))
    if show:
        plt.figure(figsize=(40, 20))
        plt.imshow(im)
        for rec in recs:
            symbol = rec['symbol']
            pixel = rec['pixel']
            if pixel[0]!= 0 or pixel[1]!=0:
                plt.text(pixel[0],pixel[1],symbol, color='purple',fontsize=42, weight ='bold')
                #plt.text(pixel[0],pixel[1],symbol, style='italic',fontsize=30, weight ='bold', bbox=dict(boxstyle="round", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8),))
    err1 = 0
    err2 = 0
    feature = ['id', 'symbol', 'name', 'x', 'y', 'pixel_x', 'pixel_y', 'calc_pixel_x', 'calc_pixel_y']
    features = []
    features.append(feature)
    for i in range(pos2[good == 1].shape[0]):
        p1 = pixels[good == 1][i, :]
        pp = np.array([pos2[good == 1][i,0],pos2[good == 1][i, 1], 1.0])
        pp2 = np.matmul(np.linalg.inv(M),pp)
        pp2 = pp2/pp2[2]
        P1 = np.array([p1[0],p1[1],1.0])
        PP2 = np.matmul(M,P1)
        PP2 = PP2/PP2[2]
        P2 = pos2[good==1][i,:]
        if show and good[i]:
            print(i)
            print(mask[i]==1,p1,pp2[0:2],np.linalg.norm(p1-pp2[0:2]))
            print(mask[i]==1,P2,PP2[0:2],np.linalg.norm(P2-PP2[0:2]))
        if mask[i] == 1:
            err1 += np.linalg.norm(p1-pp2[0:2])
            err2 += np.linalg.norm(P2-PP2[0:2])
        if show:
            color = 'green' if mask[i] == 1 else 'red'
            plt.plot([p1[0],pp2[0]],[p1[1],pp2[1]],color = color, linewidth=5)
            plt.plot(p1[0], p1[1], marker = 'X', color=color, markersize=10)
            plt.plot(pp2[0], pp2[1], marker='o', color=color, markersize=10)
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
    for r in recs:    # Extracting features that were not noted on the image (pixel_x and pixel_y are 0)
        i += 1
        p1 = pixels[i,:]
        if p1[0] == 0 and p1[1] == 0:
            pp = np.array([pos2[i,0],pos2[i,1],1.0])
            pp2 = np.matmul(np.linalg.inv(M),pp)
            pp2 = pp2/pp2[2]
            if show:
                plt.text(pp2[0],pp2[1],r['symbol'],color='black',fontsize=42, style='italic',
                         weight='bold')
                plt.plot(pp2[0],pp2[1],marker='s', markersize=10, color='black')
                x = r['pos3d'][0]
                y = r['pos3d'][1]
                feature = [i, recs[i]['symbol'], recs[i]['name'], x, y, 0, 0, pp2[0], pp2[1]]
                features.append(feature)
    if show:
        outputCsv = output.replace(".png", "_accuracies.csv")
        csvFile = open(outputCsv, 'w', newline='', encoding='utf-8')
        csvWriter = csv.writer(csvFile)
        for f in features:
            csvWriter.writerow(f)

        # send features to the function that correlates between the feature themsrlves
        results = correlate_features(features, 1)
        # get the results and write to a nother CSV file
        outputCsv = output.replace(".png", "_correlations.csv")
        csvFile = open(outputCsv, 'w', newline='', encoding='utf-8')
        csvWriter = csv.writer(csvFile)
        for r in results:
            csvWriter.writerow(r)

        print('Output file: ',outputfile)
        plt.savefig(outputfile, dpi=300)
        plt.show()

    err2 += np.sum(1-mask)*ransacbound
    if show:
        print ('err',err1,err1/np.sum(mask),err2,err2/np.sum(mask))
    return err1,err2


# **********
# read data from the features file
# **********
def read_points_data(filename,pixel_x,pixel_y,scale):
    with open(filename) as csv_file:
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
                symbol = row[6]
                pixel = np.array([int(row[indx]),int(row[indy])])/scale
                height = float(row[5]) + float(row[2])
                pos3d = np.array([float(row[3]),float(row[4]),height])
                name = row[1]
             
                rec = {'symbol' : symbol,
                        'pixel' : pixel,
                        'pos3d' : pos3d,
                        'name' : name}
                recs.append(rec)
        print(f'Processed {line_count} lines.')
        return recs


# **********
# read data from the potential camera locations file
# **********
def read_camera_locations():
    with open(camera_locations) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        recs = []
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
                names = row
            else:
                line_count += 1
                grid_code = int(row[2])
                height = float(row[5]) + 2.0    # addition of 2 meters  as the observer height
                pos3d = np.array([height,float(row[3]),float(row[4])])
                rec = {'grid_code' : grid_code,
                       'pos3d' : pos3d}
                recs.append(rec)
        print(f'Processed {line_count} lines.')
        return recs
    

# **********
# Main function
# **********
def do_it(image_name, features, pixel_x, pixel_y, output, scale):
    im = cv2.imread(image_name)
    im2 = np.copy(im)
    im[:,:,0] = im2[:,:,2]
    im[:,:,1] = im2[:,:,1]
    im[:,:,2] = im2[:,:,0]

    plt.figure(figsize=(11.69, 8.27))  # 40,20
    plt.imshow(im)

    recs = read_points_data(features,pixel_x,pixel_y,scale)
    locations = read_camera_locations()
    pixels = []
    for rec in recs:
        symbol = rec['symbol']
        pixel = rec['pixel']
        if pixel[0] != 0 or pixel[1] != 0:
            plt.text(pixel[0],pixel[1],symbol,color='red',fontsize=42)
        pixels.append(pixel)

    num_matches12 = find_homographies(recs, locations, im, False, 120.0, output)
    num_matches2 = num_matches12[:, 1]
    #print(np.min(num_matches2[num_matches2 > 0]))
    #print(np.max(num_matches2[num_matches2 > 0]))

    num_matches2[num_matches2 == 0] = 1000000
    print(np.min(num_matches2))

    theloci = np.argmin(num_matches2)      # theloci contains the best location for the camera
    print('location id: ' + str(theloci) + ' - ' + str(locations[theloci]))

    find_homographies(recs, [locations[theloci]], im, True, 120.0, output)  # Orig = 120.0



#img = '0539'
#img = '0518'
#img = 'Henn'
#img = 'Broyn'
#img = 'Tirion'
img = 'Laboard_b'


camera_locations = ''
grid_code_min = 7

if img == '0539':
    ret, mtx, dist, rvecs, tvecs = calibrate_camera(23)
    img = cv2.imread('DSC_0539.tif')
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)   # un-distort
    cv2.imwrite('tmpDSC_0539.png', dst)

    image_name = 'tmpDSC_0539.png'
    features = 'features.csv'
    camera_locations = 'potential_camera_locations_3D.csv'
    pixel_x = 'Pixel_x_DSC_0539'
    pixel_y = 'Pixel_y_DSC_0539'
    output = 'zOutput_DSC_0539.png'
    scale = 1.0
    
elif img == '0518':
    ret, mtx, dist, rvecs, tvecs = calibrate_camera(23)
    img = cv2.imread('DSC_0518.tif')
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)   # un-distort
    cv2.imwrite('tmpDSC_0518.png', dst)

    image_name = 'tmpDSC_0518.png'
    features = 'features.csv'
    camera_locations = 'potential_camera_locations_3D.csv'
    pixel_x = 'Pixel_x_DSC_0518'
    pixel_y = 'Pixel_y_DSC_0518'
    output = 'zOutput_DSC_0518.png'
    scale = 1.0
elif img == 'Henn':
    image_name = 'NNL_Henniker.jpg'
    features = 'features.csv'
    camera_locations = 'potential_camera_locations_3D.csv'
    pixel_x = 'Pixel_x_Henniker'
    pixel_y = 'Pixel_y_Henniker'
    output = 'zOutput_Henniker.png'
    scale = 1.0

elif img == 'Broyn':
    image_name = 'de-broyn-1698.tif'
    features = 'features.csv'
    camera_locations = 'potential_camera_locations_3D.csv'
    pixel_x = 'Pixel_x_Broyin'
    pixel_y = 'Pixel_y_Broyin'
    output = 'zOutput_Broyin.png'
    scale = 1.0
elif img == 'Tirion':
    image_name = 'Tirion-1732.tif'
    features = 'features.csv'
    camera_locations = 'potential_camera_locations_3D.csv'
    pixel_x = 'Pixel_x_Tirion'
    pixel_y = 'Pixel_y_Tirion'
    output = 'zOutput_Tirion.png'
    scale = 1.0
elif img == 'Laboard_b':
    image_name = 'laboard_before.tif'
    features = 'features_tiberias.csv'
    camera_locations = 'potential_camera_locations_tiberias_3D.csv'
    pixel_x = 'Pixel_x_Laboard_b'
    pixel_y = 'Pixel_y_Laboard_b'
    output = 'zOutput_Laboard_b.png'
    scale = 1.0
else:
    print('No file was selected')

do_it(image_name,features,pixel_x,pixel_y,output,scale)


print ('**********************')
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

print ('Done!')





