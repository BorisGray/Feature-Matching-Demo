# coding:utf-8
import os
import uuid
import cv2
import numpy as np
import urllib2
import json
import traceback
import devicepassai.common.LogUtils as LogUtils

try:
    from PIL import Image
    PIL_installed = True
except ImportError:
    PIL_installed = False

log = LogUtils.MyLog('/tmp/feature_matching.log')
TMP_PATH = '/tmp/'

def downloadFile(url_remote, file_name_local):
    try:
        url = url_remote.strip()

        r = urllib2.Request(url)
        req = urllib2.urlopen(r)

        saveFile = open(file_name_local, 'wb')
        saveFile.write(req.read())

        saveFile.close()
        req.close()
        return True
    except:
        log.error(traceback.format_exc())
        return False

def jsonParser(js_data) :
    log.info("jsonParser.............")

    js_dict = json.loads(js_data)
    # for key, value in js_dict.items():
    #     print key, value

    tl_x = round(js_dict['left-top-x'])
    tl_y = round(js_dict['left-top-y'])
    rd_x = round(js_dict['right-down-x'])
    rd_y = round(js_dict['right-down-y'])

    area = (0, 0, 0, 0)
    if tl_x >= rd_x or tl_y >= rd_y:
        log.info("[FeatureMatching]: Object coordinates is INVALID!!!")
        return False, area
    else:
        area = (tl_x, tl_y, rd_x, rd_y)
        return True, area

def verifyRectObjInScene(obj_rect, scene_mat):
        scn_h = scene_mat.shape[0]
        scn_w = scene_mat.shape[1]

        obj_left  = obj_rect[0]
        obj_upper = obj_rect[1]
        obj_right = obj_rect[2]
        obj_down  = obj_rect[3]

        if (obj_left > scn_w
            or obj_right > scn_w
            or obj_upper > scn_h
            or obj_down > scn_h) :
            return False
        else:
            return True

class DetectorExractorPolicy :
    SURF_POLICY = 0
    SIFT_POLICY = 1
    AKAZE_POLICY = 2

class DescriptorMatcherPolicy :
    FLANN_BASED_POLICY = 0
    BRUTE_FORCE_HAMMING_POLICY = 1

class CalibEstimatePolicy :
    RANSAC_POLICY = 8
    LMEDS_POLICY = 4

class TemplateMatchingMethod :
    CV_TM_SQDIFF = 0
    CV_TM_SQDIFF_NORMED = 1
    CV_TM_CCORR = 2
    CV_TM_CCORR_NORMED = 3
    CV_TM_CCOEFF = 4
    CV_TM_CCOEFF_NORMED = 5

class FeatureMatchingTiming :
    def __init__(self):
        self.detector_tm   = 0.0
        self.descriptor_tm = 0.0
        self.matcher_tm    = 0.0
        self.homography_tm = 0.0
        self.transform_tm  = 0.0

class FeatureMatchingOptions :
    def __init__(self):
        self.detectorExractorPolicy    = DetectorExractorPolicy.SURF_POLICY
        self.descriptorMatcherPolicy   = DescriptorMatcherPolicy.FLANN_BASED_POLICY
        self.calibEstimatePolicy       = CalibEstimatePolicy.RANSAC_POLICY
        self.templateMatchingMethod    = TemplateMatchingMethod.CV_TM_CCORR_NORMED
        self.min_hessian  = 400
        self.max_distance = 0.0
        self.min_distance = 100.0

TEMPLAT_MATCH_THRESHOLD = 0.8
class FeatureMatching:
    def __init__(self, object_image_info, scene_image_url, feat_match_opts, replay_flag = False):
        self.init_succ_token = True
        self.replay_flag     = replay_flag
        self.feat_match_opts = feat_match_opts

        # scene_file_name = "scene_image.png"
        self.scene_file_name = str(uuid.uuid4()) + ".png"

        # replay
        if self.replay_flag:
            log.info("[FeatureMatching]: REPLAY >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

            self.object_file_name = object_image_info
            log.info("[FeatureMatching]: Object file name: " + self.object_file_name)

            log.info("[FeatureMatching]: Scene image url: " + scene_image_url)
            if "http" in scene_image_url:
                download_success = downloadFile(scene_image_url, self.scene_file_name)
                assert download_success, "Download image file ERROR!!!"
            else:
                self.scene_file_name = scene_image_url
            log.info("scene_file_name = " + self.scene_file_name)

            # Read image file using color space mode
            self.object_image_color = cv2.imread(TMP_PATH + self.object_file_name, cv2.IMREAD_COLOR)
            if not self.object_image_color.data:
                log.info("[FeatureMatching]:Read object image file FAILURE!!!")
                self.init_succ_token = False
                return
            # cv2.imshow("object color", self.object_image_color)
            # cv2.waitKey(0)

            self.object_image_grey = cv2.cvtColor(self.object_image_color, cv2.COLOR_BGR2GRAY)
            if not getattr(self.object_image_grey, 'data'):
                log.info("[FeatureMatching]:Object image convert FAILURE!!!")
                self.init_succ_token = False
                return

            self.object_width = self.object_image_color.shape[1]
            self.object_height = self.object_image_color.shape[0]

            self.scene_image_color = cv2.imread(self.scene_file_name, cv2.IMREAD_COLOR)
            log.info(self.scene_image_color.shape)
            # if self.scene_image_color == None:
            #     print "[FeatureMatching]:Read scene image file FAILURE!!!"
            #     return

            self.scene_image_grey = cv2.cvtColor(self.scene_image_color, cv2.COLOR_BGR2GRAY)
            log.info(self.scene_image_grey.shape)
            if not getattr(self.scene_image_grey, 'data'):
                log.info("[FeatureMatching]:Scene image convert FAILURE!!!")
                self.init_succ_token = False
                return

            # self.init(DetectorExractorPolicy.SIFT_POLICY)
        # record
        else:
            log.info("[FeatureMatching]: RECORD >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            json_obj_coord = object_image_info
            if "http" in scene_image_url:
                download_success = downloadFile(scene_image_url, self.scene_file_name)
                assert download_success, "Download image file ERROR!!!"
            else:
                self.scene_file_name = scene_image_url
            log.info("scene_file_name= " + self.scene_file_name)

            self.scene_image_color = cv2.imread(self.scene_file_name, cv2.IMREAD_COLOR)
            # if not self.scene_image_color.data:
            #     print "[FeatureMatching]:Read scene image file FAILURE!!!"
            #     return
            # cv2.imshow("scene_image_color", scene_image_color)
            # cv2.waitKey(0)

            success, self.obj_area = jsonParser(json_obj_coord)
            if not success:
                self.init_succ_token = False
                return

            if not verifyRectObjInScene(self.obj_area, self.scene_image_color):
                log.info("[FeatureMatching]: Object is NOT in scene!!!")
                self.init_succ_token = False
                return

            self.object_image_color = self.imgCrop(self.scene_image_color, self.obj_area)
            # cv2.imwrite("obj_img_color.png", self.object_image_color)

            self.scene_image_grey = cv2.cvtColor(self.scene_image_color, cv2.COLOR_BGR2GRAY)

    # def __del__(self):
    #     # if os.path.exists(self.scene_file_name):
    #     #     os.remove(self.scene_file_name)
    #
    #     if os.path.exists(self.object_file_name):
    #         os.remove(self.object_file_name)

    def setObjectFileName(self, obj_file_name):
        self.object_file_name = obj_file_name

    def getSceneFileName(self):
        return self.scene_file_name

    def imgCrop(self, img, obj_area):
        left  = int(obj_area[0])
        upper = int(obj_area[1])
        right = int(obj_area[2])
        down  = int(obj_area[3])
        return img[upper:down, left:right, :]

    def imgCrop2(self, img, obj_area):
        left  = int(obj_area[0])
        upper = int(obj_area[1])
        right = int(obj_area[2])
        down  = int(obj_area[3])
        return img[upper:down, left:right]

    def init(self, detector_exractor_policy):
        if DetectorExractorPolicy.SURF_POLICY == detector_exractor_policy:
            surf = cv2.xfeatures2d.SURF_create(self.feat_match_opts.min_hessian)

        elif DetectorExractorPolicy.SIFT_POLICY == detector_exractor_policy:
            sift = cv2.xfeatures2d.SIFT_create()
        # elif (DetectorExractorPolicy.AKAZE_POLICY == detector_exractor_policy)
        #     detector = cv2.xfeatures2d.AKAZE_create();

    # def perHash(self, mat1, mat2):
    #     cv::Mat
    #     mat_dst1, mat_dst2;
    #
    #     cv::resize(mat1, mat_dst1, cv::Size(8, 8), 0, 0, cv::INTER_CUBIC);
    #     cv::resize(mat2, mat_dst2, cv::Size(8, 8), 0, 0, cv::INTER_CUBIC);
    #
    #     cv::cvtColor(mat_dst1, mat_dst1, CV_BGR2GRAY);
    #     cv::cvtColor(mat_dst2, mat_dst2, CV_BGR2GRAY);
    #
    #     int
    #     iavg1 = 0, iavg2 = 0;
    #     int
    #     arr1[64], arr2[64];
    #
    #     for (int i = 0; i < 8; i++) {
    #     uchar * data1 = mat_dst1.ptr < uchar > (i);
    #     uchar * data2 = mat_dst2.ptr < uchar > (i);
    #
    #     int tmp = i * 8;
    #
    #     for (int j = 0; j < 8; j++) {
    #     int tmp1 = tmp + j;
    #
    #     arr1[tmp1] = data1[j] / 4 * 4;
    #     arr2[tmp1] = data2[j] / 4 * 4;
    #
    #     iavg1 += arr1[tmp1];
    #     iavg2 += arr2[tmp1];
    #     }
    #     }
    #
    #     iavg1 /= 64;
    #     iavg2 /= 64;
    #
    #     for (int i = 0; i < 64; i++) {
    #     arr1[i] = (arr1[i] >= iavg1) ? 1: 0;
    #     arr2[i] = (arr2[i] >= iavg2) ? 1: 0;
    #     }
    #
    #     int
    #     idiffNum = 0;
    #
    #     for (int i = 0; i < 64; i++)
    #     if (arr1[i] != arr2[i])
    #         ++idiffNum;
    #
    # return idiffNum;


    def handleException(self, tl_pt, br_pt):
        rect = (tl_pt[0], tl_pt[1], br_pt[0], br_pt[1])
        mat_grey = self.imgCrop2(self.scene_image_grey, rect)
        # cv2.imwrite("grey.png", mat_grey)
        # cv2.meanStdDev(mat_grey, mat_mv, mat_sd)

        surf = cv2.xfeatures2d.SURF_create(self.feat_match_opts.min_hessian)
        (kps, descs) = surf.detectAndCompute(mat_grey, None)

        log.info("[FeatureMatching]: Key points number of mathched object: " + str(len(kps)))
        if (len(kps) < 4):
            return False
        return True

    # def handleException(self, tl_pt, br_pt):
    #     rect = (tl_pt[0], tl_pt[1], br_pt[0], br_pt[1])
    #     obj_grey = self.imgCrop2(self.scene_image_grey, rect)
    #     # cv2.imwrite("grey.png", mat_grey)
    #     # cv2.meanStdDev(mat_grey, mat_mv, mat_sd)
    #
    #     # sift = cv2.xfeatures2d.SIFT_create()
    #     # (kps, descs) = sift.detectAndCompute(mat_grey, None)
    #     #
    #     # log.info("[FeatureMatching]: Key points number of mathched object: " + str(len(kps)))
    #     # if (len(kps) < 4):
    #     #     return False
    #
    #     MIN_MATCH_COUNT = 10
    #     # Initiate SIFT detector
    #     sift = cv2.xfeatures2d.SIFT_create()
    #     # find the keypoints and descriptors with SIFT
    #     kp1, des1 = sift.detectAndCompute(obj_grey, None)
    #     kp2, des2 = sift.detectAndCompute(self.scene_image_grey, None)
    #     FLANN_INDEX_KDTREE = 0
    #     index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    #     search_params = dict(checks=50)
    #     flann = cv2.FlannBasedMatcher(index_params, search_params)
    #     matches = flann.knnMatch(des1, des2, k=2)
    #     # store all the good matches as per Lowe's ratio test.
    #     good = []
    #     for m, n in matches:
    #         if m.distance < 0.7 * n.distance:
    #             good.append(m)
    #     log.info(str(len(good)) + ',' + str(len(matches)))
    #
    #     succss_token = True
    #     if len(good) < MIN_MATCH_COUNT:
    #         log.info("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
    #         return False
    #     return True

    def templateMatchingForRecord(self, obj, scn):
        json_matched_data = ""
        if self.replay_flag:
            return json_matched_data

        img_display = scn.copy()
        
        result_cols = scn.shape[1] - obj.shape[1] + 1
        result_rows = scn.shape[0] - obj.shape[0] + 1

        self.feat_match_opts.templateMatchingMethod = TemplateMatchingMethod.CV_TM_CCOEFF_NORMED
        result = cv2.matchTemplate(scn, obj, self.feat_match_opts.templateMatchingMethod)
        cv2.normalize(result, result, 0, 1, cv2.NORM_MINMAX)

        # cv2.imwrite("record_match_result.png", result)

        threshold = TEMPLAT_MATCH_THRESHOLD
        max_match_score = 0.0
        after_filter = []
        temp_x = temp_y = max_match_x = max_match_y = 0
        matched_num = 0

        obj_rows = obj.shape[0]
        obj_cols = obj.shape[1]
     
        result_rows = result.shape[0]
        result_cols = result.shape[1]
        if result_rows != 0 and result_cols != 0 and result.data:
            max_match_score = float(result[0][0])

        mapScore2Point = {}
        for i in range(0, result_rows):
            for j in range(0, result_cols):
                match_value = float(result[i][j])
                if match_value >= threshold:
                    cv2.rectangle(img_display, (j ,i), (j + obj_cols, i + obj_rows), (0, 255, 0), 1, 8)
                    cv2.putText(img_display, str(match_value), (j, i+100), cv2.FONT_ITALIC, 1, (0, 0, 255), 1)

                    if match_value * 1000000 > max_match_score * 1000000:
                        max_match_score = match_value
                        max_match_x = j
                        max_match_y = i
                    mapScore2Point[match_value] = (j, i)
                    matched_num += 1
        log.info("total matched num:\t" + str(matched_num))

        if matched_num == 0:
            json_matched_data = self.exportMatchedNum(matched_num)
            return json_matched_data

        listScore2Point2_sortedByVal = sorted(mapScore2Point.iteritems(), key=lambda d:d[1][1])
        # print listScore2Point2_sortedByVal

        i = x = y = 0
        s = temp_s = d = 0.0
        l = len(listScore2Point2_sortedByVal)

        mapScore2Point_filtered = {}
        if l != 0:
            temp_s = listScore2Point2_sortedByVal[0][0]
            temp_x = listScore2Point2_sortedByVal[0][1][0]
            temp_y = listScore2Point2_sortedByVal[0][1][1]

            if (l == 1):
                mapScore2Point_filtered[temp_s] = (temp_x, temp_y)

        mean_obj_w_h = (obj_cols + obj_rows) / 4
        for i in range(1, l):
            x = listScore2Point2_sortedByVal[i][1][0]
            y = listScore2Point2_sortedByVal[i][1][1]
            s = listScore2Point2_sortedByVal[i][0]
            d = (x - temp_x) * (x - temp_x) + (y - temp_y) * (y - temp_y)

            if (s > temp_s):
                if d > mean_obj_w_h * mean_obj_w_h:
                    mapScore2Point_filtered[temp_s] = (temp_x, temp_y)
            else:
                if d > mean_obj_w_h * mean_obj_w_h:
                    mapScore2Point_filtered[temp_s] = (temp_x, temp_y)
                else:
                    continue

            temp_s = listScore2Point2_sortedByVal[i][0]
            temp_x = listScore2Point2_sortedByVal[i][1][0]
            temp_y = listScore2Point2_sortedByVal[i][1][1]

        if len(mapScore2Point_filtered) == 0 and len(listScore2Point2_sortedByVal) != 0:
            log.info("vecfiltered.size() ==0 && vecScore2Point.size() != 0")
            k = listScore2Point2_sortedByVal[0][0]
            x = listScore2Point2_sortedByVal[0][1][0]
            y = listScore2Point2_sortedByVal[0][1][1]
            mapScore2Point_filtered[k] = (x, y)

        for i in range(len(listScore2Point2_sortedByVal)):
            k = listScore2Point2_sortedByVal[i][0]
            x = listScore2Point2_sortedByVal[i][1][0]
            y = listScore2Point2_sortedByVal[i][1][1]
            # print str(x) + " => " + str(x) + ", " + str(y)

        filtered_matched_num = len(mapScore2Point_filtered)
        log.info("filter matched num:\t" + str(filtered_matched_num))

        # cv2.imwrite("templ_match_record.png", img_display);
        tl_pt = (max_match_x, max_match_y)
        rd_pt = (max_match_x + obj_cols, max_match_y + obj_rows)
        log.info("[FeatureMatching]: Matched number is: " + str(filtered_matched_num))

        if not self.handleException(tl_pt, rd_pt):
            log.info("[FeatureMatching]: Key points too little, template matching NOTHING! ")
            filtered_matched_num = 0
        else:
            if filtered_matched_num == 0:
                log.info("[FeatureMatching]: Matched numer = 0, template matching NOTHING! ")

        json_matched_data = self.exportMatchingInfoForRecord(filtered_matched_num, tl_pt, rd_pt, max_match_score)
        # print "[FeatureMatching]: Template matching for record OK! "
        return json_matched_data

    def exportMatchingInfoForReplay(self, tp_point, rd_point, scn_width, scn_height):
        json_result = ""
        result = {}
        result["left_top_x"] = tp_point[0]
        result["left_top_y"] = tp_point[1]
        result["right_down_x"] = rd_point[0]
        result["right_down_y"] = rd_point[1]
        result["scn_width"] = scn_width
        result["scn_height"] = scn_height
        json_result = json.dumps(result, indent=4)
        return json_result

    def exportMatchedNum(self, matched_num):
        data = {}
        data["matched_num"] = matched_num
        json_result = json.dumps(data, indent=4)
        log.info(json_result)
        return json_result

    def exportMatchingInfoForRecord(self, matched_num, tp_point, rd_point, max_match_score):
        data = {}
        data["matched_num"] = matched_num
        if (matched_num != 0):
            matched_rect_list = {}
            matched_rect_list["left_top_x"] = tp_point[0]
            matched_rect_list["left_top_y"] = tp_point[1]
            matched_rect_list["right_down_x"] = rd_point[0]
            matched_rect_list["right_down_y"] = rd_point[1]
            list = [matched_rect_list]
            data["max_match_score"] = max_match_score
            data["matched_rect"] = list

        json_result = json.dumps(data, indent=4)
        log.info(json_result)
        return json_result

    def verifyCoordinateResult(self, dst, h, w):
        # print type(dst)
        p1 = dst[0][0]; p2 = dst[1][0]
        p3 = dst[2][0]; p4 = dst[3][0]

        l1 = p1.tolist(); l2 = p2.tolist()
        l3 = p3.tolist(); l4 = p4.tolist()

        x_list = []
        x_list.append(l1[0]); x_list.append(l2[0])
        x_list.append(l3[0]); x_list.append(l4[0])

        for x in x_list:
            if x < 0 or x > w:
                return False

        y_list = []
        y_list.append(l1[1]); y_list.append(l2[1])
        y_list.append(l3[1]); y_list.append(l4[1])

        for y in y_list:
            if y < 0 or y > h:
                return False

        max_x = max(x_list)
        max_y = max(y_list)

        if round(max_x) > w or round(max_y) > h:
            return False
        else:
            return True

    def siftFeatureMatchingForReplay(self, obj, scn):
        json_matched_data = ""
        MIN_MATCH_COUNT = 10
        scn_width, scn_height = scn.shape[::-1]

        # Initiate SIFT detector
        sift = cv2.xfeatures2d.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(obj, None)
        kp2, des2 = sift.detectAndCompute(scn, None)
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)
        log.info(str(len(good)) + ',' +str(len(matches)))

        succss_token = True
        dst = None
        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()
            h, w = obj.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            # print type(dst)
            scn = cv2.polylines(scn, [np.int32(dst)], True, 0, 3, cv2.LINE_AA)
            succss_token = True
        else:
            log.info("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
            matchesMask = None
            succss_token = False

        if not succss_token:
            return self.templateMatchingForReplay(obj, scn)
        else:
            # plt.imshow(scn, 'gray'), plt.show()
            if not self.verifyCoordinateResult(dst, self.scene_image_color.shape[0], self.scene_image_color.shape[1]):
                return json_matched_data

            x, y, w, h = cv2.boundingRect(dst)
            json_matched_data = self.exportMatchingInfoForReplay((x,y), (x+w,y+h), scn_width, scn_height)
        return json_matched_data

    def templateMatchingForReplay(self, obj, scn):
        json_matched_data = ""
    #     exportMatchingInfoForReplay(matched_num, top_left_point, right_down_point, minVal, maxVal, json_matched_data)
    #   print "[FeatureMatching]: Template matching for replay success! "
        scn_width, scn_height = scn.shape[::-1]
        matched_num = 0
        top_left = ()
        bottom_right = ()
        w, h = obj.shape[::-1]
        self.feat_match_opts.templateMatchingMethod = TemplateMatchingMethod.CV_TM_CCOEFF_NORMED
        res = cv2.matchTemplate(scn, obj, self.feat_match_opts.templateMatchingMethod)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if (self.feat_match_opts.templateMatchingMethod == TemplateMatchingMethod.CV_TM_CCOEFF_NORMED):
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(scn, top_left, bottom_right, 255, 2)
        matched_num = matched_num + 1
        log.info("[FeatureMatching]: Template matching for replay success! ")
        return self.exportMatchingInfoForReplay(top_left, bottom_right, scn_width, scn_height)


    # def templateMatchingForReplay(self, obj, scn):
    #     '''回放接口的模板匹配，模板匹配的特点是只能用于二维平面，匹配精度高，但不具备尺存和旋转不变性特征'''
    #     json_matched_data = ""
    #
    #     if not self.replay_flag:
    #         return json_matched_data
    #
    #     matched_num = 0
    #     img_display = scn.copy()
    #
    #     result_cols = scn.cols - obj.cols + 1
    #     result_rows = scn.rows - obj.rows + 1
    #
    #     # Mat result;
    #     # result.create(result_cols, result_rows, CV_32FC1);
    #     self.feat_match_opts.templateMatchingMethod = TemplateMatchingMethod.CV_TM_CCOEFF_NORMED
    #     result = cv2.matchTemplate(scn, obj, self.feat_match_opts.templateMatchingMethod)
    #     cv2.normalize(result, result, 0, 1, cv2.NORM_MINMAX)
    #
    #     minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
    #
    #     matchLoc = maxLoc
    #
    #     matched_num + 1
    #
    #     cv2.rectangle(img_display, matchLoc, (matchLoc.x + obj.cols, matchLoc.y + obj.rows), 255, 2)
    #     # cv2.imwrite("template_match_replay.jpg", img_display)
    #
    #     top_left_point = matchLoc
    #     right_down_point = cv2.Point(matchLoc.x + obj.cols, matchLoc.y + obj.rows)
    #
    #     cv2.Mat match_rect_color;
    #     cv2.Rect rect(top_left_point, right_down_point);
    #     match_rect_color = self.img_scene_color(rect).clone();
    #
    #     cv2.imwrite("template_rectangle.jpg", match_rect_color);
    #
    #     idiff_num = perHash(match_rect_color, img_object_color)
    #
    #     print "[FeatureMatching]: idiff_num = " + idiff_num
    #
    #     if idiff_num <= 5:
    #         print "[FeatureMatching]: Two images are very similar!"
    #     elif idiff_num > 10:
    #         print "[FeatureMatching]: They are two different images!"
    #         matched_num = 0
    #         json_matched_data = ""
    #         return False
    #     else:
    #         print "[FeatureMatching]: Two image are somewhat similar!"
    #
    #     self.exportMatchingInfoForReplay(matched_num, top_left_point, right_down_point, minVal, maxVal, json_matched_data)
    #     print "[FeatureMatching]: Template matching for replay OK! "
    #     return True

    def processRecordExporJson(self):
        json_matched_data = ""
        json_matched_data = self.templateMatchingForRecord(self.object_image_color, self.scene_image_color)
        return json_matched_data

    def processRecordExporImgName(self):
        # obj_file_name = "object_image.png"
        self.object_file_name = str(uuid.uuid4()) + ".png"

        cv2.imwrite(self.object_file_name, self.object_image_color)

        return self.object_file_name

    def processReplay(self):
        json_matched_data = ""

        surf = cv2.xfeatures2d.SURF_create(self.feat_match_opts.min_hessian)
        (kps, descs) = surf.detectAndCompute(self.object_image_grey, None)

        log.info("[FeatureMatching]: Key points number of mathched object: " + str(len(kps)))
        if (len(kps) < 4):
            return json_matched_data

        json_matched_data = self.siftFeatureMatchingForReplay(self.object_image_grey, self.scene_image_grey)
        return json_matched_data

    # def verifyRectObjInScene(obj_rect, scene_mat):
    #
    #     scn_h = scene_mat.shape[0]
    #     scn_w = scene_mat.shape[1]
    #
    #     obj_left  = obj_rect[0]
    #     obj_upper = obj_rect[1]
    #     obj_right = obj_rect[2]
    #     obj_down  = obj_rect[3]
    #
    #     if (obj_left > scn_w
    #         or obj_right > scn_w
    #         or obj_upper > scn_h
    #         or obj_down > scn_h) :
    #         return False
    #     else:
    #         return True
