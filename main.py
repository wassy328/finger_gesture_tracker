import cv2 as cv
import numpy as np
import mediapipe as mp
import csv, copy, itertools, datetime, pprint
from model import KeyPointClassifier
from utils import CvFpsCalc, draw, hand
from collections import Counter

def main():
    mode = 0
    out_no = 50
    var = hand.Variables(out_no)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands = 2,
        min_detection_confidence = 0.7,
        min_tracking_confidence = 0.5,
    )
    keypoint_classifier = KeyPointClassifier(
        'model/keypoint_classifier/hand_keypoint_classifier.tflite'
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

if __name__ == "__main__":
    main()