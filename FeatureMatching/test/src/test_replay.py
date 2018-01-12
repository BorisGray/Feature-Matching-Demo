
import os
import urllib
import urllib2
import time
import datetime
import json
import base64
import cv2
import logging

def trace_logger():
    # logging.basicConfig(level=logging.DEBUG,
    #                     format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
    #                     datefmt='%a, %d %b %Y %H:%M:%S',
    #                     filename='logger.log',
    #                     filemode='w')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)




INPUT_IMAGE_PATH = "/root/workspace/dp-ai/featurematchv1/test/test-images/"
OUTPUT_IMAGE_PATH = "/root/workspace/dp-ai/featurematchv1/test/test-result/"

def encodeImage(img_file_name):
    in_file = open(img_file_name, 'rb')
    encoded = base64.b64encode(in_file.read())
    in_file.close()
    ext_name = os.path.splitext(img_file_name)[1]
    print ext_name
    if ext_name == ".jpg":
        img_header = "data:image/jpeg;base64,"
    elif ext_name == ".png":
        img_header = "data:image/png;base64,"
    else:
        img_header = "data:image/jpeg;base64,"
    return img_header + encoded

def decodedImage(encode_str):
    print encode_str
    decoded = base64.b64decode(encode_str)
    return decoded

image_file_list = ['obj1.jpg',
'obj2.jpg',
'obj3.jpg',
'obj4.jpg',
'obj5.jpg',
'obj6.jpg',
'obj7.jpg',
'obj8.jpg',
'obj9.jpg',
'obj10.jpg',
'obj11.jpg',
'obj12.jpg',
'obj12.jpg',
'obj13.jpg',
'obj14.jpg',
'obj13.jpg',
'obj14.jpg',
'obj15.jpg',
'obj16.jpg',
'obj16.jpg',
'obj17.jpg',
'obj18.jpg',
'obj19.jpg',
# 'obj20.jp,
'obj21.jpg',
'obj21.jpg',
'obj22.jpg',
'obj23.jpg',
'obj24.jpg',
'obj25.jpg',

'obj26.jpg',
'obj27.jpg',

'obj28.jpg',
'obj29.jpg',
'obj30.jpg',
'obj31.jpg'
]

# build k-v map
filename_map = {
# 'obj1_0.9.jpg':'scn1.jpg',
# 'obj1_0.8.jpg':'scn1.jpg',
# 'obj1_0.7.jpg':'scn1.jpg',
# 'obj1_0.6.jpg':'scn1.jpg',
# 'obj1_1.1.jpg':'scn1.jpg',
# 'obj1_1.2.jpg':'scn1.jpg',
# 'obj1_1.3.jpg':'scn1.jpg',
# 'obj1_1.4.jpg':'scn1.jpg',

# 'obj2.jpg':'scn5.jpg',
# 'obj2_0.9.jpg':'scn5.jpg',
# 'obj2_0.8.jpg':'scn5.jpg',
# 'obj2_0.7.jpg':'scn5.jpg',
# 'obj2_0.6.jpg':'scn5.jpg',
# 'obj2_1.1.jpg':'scn5.jpg',
# 'obj2_1.2.jpg':'scn5.jpg',
# 'obj2_1.3.jpg':'scn5.jpg',
# 'obj2_1.4.jpg':'scn5.jpg'

# 'obj3.jpg':'scn1.jpg',
# 'obj3_0.9.jpg':'scn1.jpg',
# 'obj3_0.8.jpg':'scn1.jpg',
# 'obj3_0.7.jpg':'scn1.jpg',
# 'obj3_0.6.jpg':'scn1.jpg',
# 'obj3_1.1.jpg':'scn1.jpg',
# 'obj3_1.2.jpg':'scn1.jpg',
# 'obj3_1.3.jpg':'scn1.jpg',
# 'obj3_1.4.jpg':'scn1.jpg',

# 'obj4.jpg':'scn1.jpg',
# 'obj4_0.9.jpg':'scn1.jpg',
# 'obj4_0.8.jpg':'scn1.jpg',
# 'obj4_0.7.jpg':'scn1.jpg',
# 'obj4_0.6.jpg':'scn1.jpg',
# 'obj4_1.1.jpg':'scn1.jpg',
# 'obj4_1.2.jpg':'scn1.jpg',
# 'obj4_1.3.jpg':'scn1.jpg',
# 'obj4_1.4.jpg':'scn1.jpg',


# 'obj5.jpg':'scn2.jpg',
# 'obj5_0.9.jpg':'scn2.jpg',
# 'obj5_0.8.jpg':'scn2.jpg',
# 'obj5_0.7.jpg':'scn2.jpg',
# 'obj5_0.6.jpg':'scn2.jpg',
# 'obj5_1.1.jpg':'scn2.jpg',
# 'obj5_1.2.jpg':'scn2.jpg',
# 'obj5_1.3.jpg':'scn2.jpg',
# 'obj5_1.4.jpg':'scn2.jpg',

# 'obj6.jpg':'scn2.jpg',
# 'obj6_0.9.jpg':'scn2.jpg',
# 'obj6_0.8.jpg':'scn2.jpg',
# 'obj6_0.7.jpg':'scn2.jpg',
# 'obj6_0.6.jpg':'scn2.jpg',
# 'obj6_1.1.jpg':'scn2.jpg',
# 'obj6_1.2.jpg':'scn2.jpg',
# 'obj6_1.3.jpg':'scn2.jpg',
# 'obj6_1.4.jpg':'scn2.jpg',
#
# 'obj7.jpg':'scn2.jpg',
# 'obj7_0.9.jpg':'scn2.jpg',
# 'obj7_0.8.jpg':'scn2.jpg',
# 'obj7_0.7.jpg':'scn2.jpg',
# 'obj7_0.6.jpg':'scn2.jpg',
# 'obj7_1.1.jpg':'scn2.jpg',
# 'obj7_1.2.jpg':'scn2.jpg',
# 'obj7_1.3.jpg':'scn2.jpg',
# 'obj7_1.4.jpg':'scn2.jpg',


'obj17.jpg':'scn6.jpg',
'obj17_0.9.jpg':'scn6.jpg',
'obj17_0.8.jpg':'scn6.jpg',
'obj17_0.7.jpg':'scn6.jpg',
'obj17_0.6.jpg':'scn6.jpg',
'obj17_1.1.jpg':'scn6.jpg',
'obj17_1.2.jpg':'scn6.jpg',
'obj17_1.3.jpg':'scn6.jpg',
'obj17_1.4.jpg':'scn6.jpg',

# # 'obj1.jpg':'scn1.jpg',
# # 'obj2.jpg':'scn5.jpg',
# # 'obj3.jpg':'scn1.jpg',
# # 'obj4.jpg':'scn1.jpg',
# # 'obj5.jpg':'scn2.jpg',
# #
# # 'obj6.jpg':'scn2.jpg',
# # 'obj7.jpg':'scn2.jpg',
# # 'obj8.jpg':'scn3.jpg',
# # 'obj9.jpg':'scn3.jpg',
# # 'obj10.jpg':'scn3.jpg',
# #
# # 'obj11.jpg':'scn3.jpg',
# # 'obj12.jpg':'scn3.jpg',
# # 'obj12.jpg':'scn4.jpg',
# # 'obj13.jpg':'scn3.jpg',
# # 'obj14.jpg':'scn3.jpg',
# # 'obj13.jpg':'scn4.jpg',
# # 'obj14.jpg':'scn4.jpg',
# # 'obj15.jpg':'scn5.jpg',
# #
# # 'obj16.jpg':'scn5.jpg',
# # 'obj17.jpg':'scn6.jpg',
# # 'obj18.jpg':'scn6.jpg',
# # 'obj19.jpg':'scn6.jpg',
# # # 'obj20.jpg':'scn8.jpg',
# #
# # 'obj21.jpg':'scn6.jpg',
# # 'obj21.jpg':'scn8.jpg',
# # 'obj22.jpg':'scn9.jpg',
# # 'obj23.jpg':'scn9.jpg',
# # 'obj24.jpg':'scn9.jpg',
# # 'obj25.jpg':'scn9.jpg',
#
# 'obj26.jpg':'scn9.jpg',
# 'obj27.jpg':'scn8.jpg',
#
# 'obj28.jpg':'scn10.jpg',
# 'obj29.jpg':'scn10.jpg',
# 'obj30.jpg':'scn11.jpg',
# 'obj31.jpg':'scn11.jpg',
}

def scaleDownUp(img_file_name, factor):
    image_color = cv2.imread(INPUT_IMAGE_PATH + img_file_name, cv2.IMREAD_COLOR)
    height = image_color.shape[0]
    width = image_color.shape[1]
    scale_img_color = cv2.resize(image_color, None, fx = factor, fy = factor, interpolation = cv2.INTER_CUBIC)
    file_name = os.path.splitext(img_file_name)[0] + "_" + str(factor)
    cv2.imwrite(file_name + ".jpg", scale_img_color)
    return file_name

def createScaleDownUpImage():
    for i in range(len(image_file_list)):
        print scaleDownUp(image_file_list[i], 0.9)
        print scaleDownUp(image_file_list[i], 0.8)
        print scaleDownUp(image_file_list[i], 0.7)
        print scaleDownUp(image_file_list[i], 0.6)
        print scaleDownUp(image_file_list[i], 1.1)
        print scaleDownUp(image_file_list[i], 1.2)
        print scaleDownUp(image_file_list[i], 1.3)
        print scaleDownUp(image_file_list[i], 1.4)

# createScaleDownUpImage()

start = time.clock()
url = "http://localhost:8001/check"
for k, v in filename_map.items():
    print k, v
    encoded = encodeImage(INPUT_IMAGE_PATH + k)
    print encoded

    values = {"object-image-base64-str": encoded, "scene-image-url": INPUT_IMAGE_PATH + v, "replay-flag": True}
    print values

    # JSON encoding
    jdata = json.dumps(values)
    req = urllib2.Request(url, jdata)
    print req

    # send request
    response = urllib2.urlopen(req)
    res = response.read()
    print type(res),   res

    if res != '[""]':
        json_list = json.loads(res)
        print type(json_list[0]), json_list[0]
        json_dict = json.loads(json_list[0])
        print type(json_dict)
        for key, value in json_dict.items():
            print key, value

        tl_x = json_dict["left_top_x"]
        tl_y = json_dict["left_top_y"]
        rd_x = json_dict["right_down_x"]
        rd_y = json_dict["right_down_y"]

        scene_image_color = cv2.imread(INPUT_IMAGE_PATH + v, cv2.IMREAD_COLOR)
        cv2.rectangle(scene_image_color, (tl_x, tl_y), (rd_x, rd_y), (0, 0, 255), 2, 8)
        obj_name = os.path.splitext(k)[0]
        scn_name = os.path.splitext(v)[0]
        filename = OUTPUT_IMAGE_PATH + obj_name+"_"+scn_name+".jpg"
        cv2.imwrite(filename, scene_image_color)
    time.sleep(1)

end = time.clock()
print "Elapse: %f s" % (end - start)
