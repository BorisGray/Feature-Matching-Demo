import os
import os.path
import json
import base64
import uuid

# from libfeatMatch import *
import feature_matching as FM

obj_file_name = ""
TMP_PATH = '/tmp/'

def encodeImage(img_file_name):
    in_file = open(img_file_name, 'rb')
    encoded = base64.b64encode(in_file.read())
    in_file.close()
    ext_name = os.path.splitext(img_file_name)[1]
        # print ext_name
    if ext_name == ".jpg":
        img_header = "data:image/jpeg;base64,"
    elif ext_name == ".png":
        img_header = "data:image/png;base64,"
    else:
        img_header = "data:image/jpeg;base64,"
    return img_header + encoded


def decodedImage(encode_str):
        # print encode_str
    decoded = base64.b64decode(encode_str)
    return decoded

def invoke_feature_matching(data):
    global obj_file_name
    # print (">>>>>>>>MainHandler::invoke_feature_matching()")
    # for key, value in data.items():
    #     print key, value

    replay_flag = data['replay-flag']
    replay_flag = eval(replay_flag)
    # print replay_flag
    # replay
    if replay_flag == True:
        obj = data['object-image-name']
    # record
    elif replay_flag == False:
        obj = data['object-image-points']
    # print obj

    utf_obj = obj.encode("utf-8")
    utf_img_scn_url = data['scene-image-url'].encode("utf-8")

    export_list = []
    opts = FM.FeatureMatchingOptions()
    featureMatching = FM.FeatureMatching(utf_obj, utf_img_scn_url, opts, replay_flag)
    if not featureMatching.init_succ_token:
        return export_list

    # replay
    if replay_flag == True:
        export_list.append(featureMatching.processReplay())
        featureMatching.setObjectFileName(obj_file_name)

        return export_list

    # record
    elif replay_flag == False:
        json_matched_data = featureMatching.processRecordExporJson()
        # print json_matched_data
        img_file_name = featureMatching.processRecordExporImgName()
        encoded = encodeImage(img_file_name)
        # print encoded

        export_list.append(json_matched_data)
        export_list.append(encoded)
        return export_list

def main_entry(json_str):
    global obj_file_name
    param = json_str.decode('utf-8')
    # print(param)
    json_dict = json.loads(param)
    # for key, value in json_dict.items():
    #     print key, value

    scn_img_url = json.dumps(json_dict['scene-image-url'])
    json_dict['scene-image-url'] = scn_img_url.strip('"')

    # replay
    if eval(json_dict['replay-flag']) == True:
        obj_base64_str = json_dict['object-image-base64-str']
        str_img_data = json.dumps(obj_base64_str).strip('"')
        l1 = len((str_img_data))
        if "data:image/jpeg;base64," in str_img_data:
            l2 = len("data:image/jpeg;base64,")
            str_img_data2 = str_img_data[l2: l1]
            # print str_img_data2
        elif "data:image/png;base64," in str_img_data:
            l2 = len("data:image/png;base64,")
            str_img_data2 = str_img_data[l2: l1]
            # print str_img_data2

        decoded = decodedImage(str_img_data2)
        # print decoded
        # obj_file_name = "object_image.png"
        obj_file_name = str(uuid.uuid4()) + ".png"
        # print obj_file_name
        out_file = open(TMP_PATH + obj_file_name, 'wb')
        out_file.write(decoded)
        out_file.close()
        json_dict['object-image-name'] = obj_file_name
        del json_dict['object-image-base64-str']

    # record
    elif eval(json_dict['replay-flag']) == False:
        obj_coord_json = json.dumps(json_dict['object-image-points'])
        json_dict['object-image-points'] = obj_coord_json

    result = invoke_feature_matching(json_dict)
    return result
