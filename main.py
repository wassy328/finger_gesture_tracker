import cv2 as cv
import numpy as np
import mediapipe as mp
import csv
import copy
import itertools
import datetime
import pprint
from model import KeyPointClassifier
from utils import CvFpsCalc
from utils import draw
#from utils import hand
from collections import Counter

def main():
    mode = 0
    out_no = 87
    var = out_no
    #var = hand.Variables(out_no)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands = 2,
        min_detection_confidence = 0.7,
        min_tracking_confidence = 0.5,
    )
    keypoint_classifier = KeyPointClassifier(
        'model/keypoint_classifier/keypoint_classifier.tflite'
        )

    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding = 'utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]

    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 540)

    cvFpsCalc = CvFpsCalc(buffer_len = 10)

    while True:
        fps = cvFpsCalc.get()

        key = cv.waitKey(0)
        if key == 27:
            break
        number, mode = select_mode(key, mode)

        ret, image = cap.read()
        if not ret:
            break

        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks is not None and mode != 9:
            left_fg, right_fg = 0, 0
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                logging_csv(number, mode, pre_processed_landmark_list,
                            pre_processed_point_history_list)

                if handedness.classification[0]/label[0:] == "Left":
                    left_fg = 1
                    var.hand_sign_id = out_no
                else:
                    right_fg = 1
                    var.hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                    landmark_tips = get_landmark_tips(image, hand_landmarks)

                    var.get_hand_sign_id(landmark_tips, hand_landmarks)

                    landmarks_ave = calc_landmarks_ave(hand_landmarks)
                    var.get_mode_id(hand_landmarks, landmarks_ave)

                    var.fix_hand_sign_id()

                    if left_fg == 0:
                        var.get_input_letters(keypoint_classifier_labels)
                        
                brect_center = calc_brect_center(brect)
                var.get_input_letters_position_x = round(brect_center[0])
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    point_history_classifier_labels[most_common_fg_id[0][0]],
                )

            if left_fg == 1 and right_fg == 1:
                var.hand_sign_id = out_no - 1
                if get_tm_int() - var.del_lock_tm > 100 and 2 <= len(var.input_letters):
                    var.input_letters = var.input_letters[:-2] + var.input_letters[-1]
                    var.del_lock_tm = get_tm_int()
        else:
            if get_tm_int() - var.del_lock_tm > 300:
                var.input_letters = ""
                var.del_lock_tm = get_tm_int()

        debug_image = draw_info(debug_image, fps, mode, number)
        debug_image = draw_info_text(debug_image, var.input_letters)

        cv.imshow("hand gesture recog", debug_image)

    cap.release()
    hands.close()
    cv.destroyAllWindows()

def get_tm_int():
    return int(datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')[:16])

def get_landmark_tips(image, landmarks):
    up_num, down_num, left_num, right_num = 0, 0, 0, 0
    up_base, down_base, left_base, right_base = image.shape[0], 0, image.shape[1], 0
    for i, landmark in enumerate(landmarks.landmark):
        if up_base >= landmark.y:
            up_num, up_base = i, landmark.y
        if down_base <= landmark.y:
            down_num, down_base = i, landmark.y
        if left_base >= landmark.x:
            left_num, left_base = i, landmark.x
        if right_base <= landmark.x:
            right_num, right_base = i, landmark.x

    return [up_num, down_num, left_num, right_num]

def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:
        number = key - 48
    if key == 110:
        mode = 0
    if key == 107:
        mode = 1
    if key == ord("s"):
        mode = 9

    return number, mode

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point = [np.array((landmark_x, landmark_y))]
        landmark_array = np.append(landmark_array, landmark_point, axis = 0)
        x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]

def calc_brect_center(brect):
    center_x = (brect[0] + brect[2]) / 2
    center_y = (brect[1] + brect[3]) / 2
    
    return [center_x , center_y]

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_z = landmark.z
        landmark_point.append([landmark_x, landmark_y, landmark_z])

    return landmark_point

def calc_landmarks_ave(landmarks):
    sum_x, sum_y, sum_z = 0, 0, 0

    for _, landmark in enumerate(landmarks.landmark):
        sum_x = sum_x + landmark.x
        sum_y = sum_y + landmark.y
        sum_z = sum_z + landmark.z
    ave_x, ave_y, ave_z = sum_x / 21, sum_y / 21, sum_z / 21

    return [ave_x, ave_y, ave_z]

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y, base_z = 0, 0, 0

    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y, base_z = landmark_point[0], landmark_point[1], landmark_point[2]
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
        temp_landmark_list[index][2] = (temp_landmark_list[index][2] - base_z) * 200
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    max_value = max(list(map(abs, temp_landmark_list)))
    def normalize_(n): return n / max_value
    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

def logging_csv(number, mode, landmark_list):

    if mode == 0: pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline = "") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])

if __name__ == "__main__":
    main()