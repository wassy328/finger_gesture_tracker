import cv2 as cv
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from math import floor

Font = '/System/Library/Fonts/ヒラギノ丸ゴ ProN W4.ttc'

def get_point_x_y(landmark_point):
    return landmark_point[0], landmark_point[1]

def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        cv.line(image, get_point_x_y(landmark_point[2]), get_point_x_y(landmark_point[3]),
                (0, 0, 0), 6)
        cv.line(image, get_point_x_y(landmark_point[2]), get_point_x_y(landmark_point[3]),
                (255, 255, 255), 2)
        cv.line(image, get_point_x_y(landmark_point[3]), get_point_x_y(landmark_point[4]),
                (0, 0, 0), 6)
        cv.line(image, get_point_x_y(landmark_point[3]), get_point_x_y(landmark_point[4]),
                (255, 255, 255), 2)
        cv.line(image, get_point_x_y(landmark_point[5]), get_point_x_y(landmark_point[6]),
                (0, 0, 0), 6)
        cv.line(image, get_point_x_y(landmark_point[5]), get_point_x_y(landmark_point[6]),
                (255, 255, 255), 2)
        cv.line(image, get_point_x_y(landmark_point[6]), get_point_x_y(landmark_point[7]),
                (0, 0, 0), 6)
        cv.line(image, get_point_x_y(landmark_point[6]), get_point_x_y(landmark_point[7]),
                (255, 255, 255), 2)
        cv.line(image, get_point_x_y(landmark_point[7]), get_point_x_y(landmark_point[8]),
                (0, 0, 0), 6)
        cv.line(image, get_point_x_y(landmark_point[7]), get_point_x_y(landmark_point[8]),
                (255, 255, 255), 2)

        cv.line(image, get_point_x_y(landmark_point[9]), get_point_x_y(landmark_point[10]),
                (0, 0, 0), 6)
        cv.line(image, get_point_x_y(landmark_point[9]), get_point_x_y(landmark_point[10]),
                (255, 255, 255), 2)
        cv.line(image, get_point_x_y(landmark_point[10]), get_point_x_y(landmark_point[11]),
                (0, 0, 0), 6)
        cv.line(image, get_point_x_y(landmark_point[10]), get_point_x_y(landmark_point[11]),
                (255, 255, 255), 2)
        cv.line(image, get_point_x_y(landmark_point[11]), get_point_x_y(landmark_point[12]),
                (0, 0, 0), 6)
        cv.line(image, get_point_x_y(landmark_point[11]), get_point_x_y(landmark_point[12]),
                (255, 255, 255), 2)

        cv.line(image, get_point_x_y(landmark_point[13]), get_point_x_y(landmark_point[14]),
                (0, 0, 0), 6)
        cv.line(image, get_point_x_y(landmark_point[13]), get_point_x_y(landmark_point[14]),
                (255, 255, 255), 2)
        cv.line(image, get_point_x_y(landmark_point[14]), get_point_x_y(landmark_point[15]),
                (0, 0, 0), 6)
        cv.line(image, get_point_x_y(landmark_point[14]), get_point_x_y(landmark_point[15]),
                (255, 255, 255), 2)
        cv.line(image, get_point_x_y(landmark_point[15]), get_point_x_y(landmark_point[16]),
                (0, 0, 0), 6)
        cv.line(image, get_point_x_y(landmark_point[15]), get_point_x_y(landmark_point[16]),
                (255, 255, 255), 2)

        cv.line(image, get_point_x_y(landmark_point[17]), get_point_x_y(landmark_point[18]),
                (0, 0, 0), 6)
        cv.line(image, get_point_x_y(landmark_point[17]), get_point_x_y(landmark_point[18]),
                (255, 255, 255), 2)
        cv.line(image, get_point_x_y(landmark_point[18]), get_point_x_y(landmark_point[19]),
                (0, 0, 0), 6)
        cv.line(image, get_point_x_y(landmark_point[18]), get_point_x_y(landmark_point[19]),
                (255, 255, 255), 2)
        cv.line(image, get_point_x_y(landmark_point[19]), get_point_x_y(landmark_point[20]),
                (0, 0, 0), 6)
        cv.line(image, get_point_x_y(landmark_point[19]), get_point_x_y(landmark_point[20]),
                (255, 255, 255), 2)

        cv.line(image, get_point_x_y(landmark_point[0]), get_point_x_y(landmark_point[1]),
                (0, 0, 0), 6)
        cv.line(image, get_point_x_y(landmark_point[0]), get_point_x_y(landmark_point[1]),
                (255, 255, 255), 2)
        cv.line(image, get_point_x_y(landmark_point[1]), get_point_x_y(landmark_point[2]),
                (0, 0, 0), 6)
        cv.line(image, get_point_x_y(landmark_point[1]), get_point_x_y(landmark_point[2]),
                (255, 255, 255), 2)
        cv.line(image, get_point_x_y(landmark_point[2]), get_point_x_y(landmark_point[5]),
                (0, 0, 0), 6)
        cv.line(image, get_point_x_y(landmark_point[2]), get_point_x_y(landmark_point[5]),
                (255, 255, 255), 2)
        cv.line(image, get_point_x_y(landmark_point[5]), get_point_x_y(landmark_point[9]),
                (0, 0, 0), 6)
        cv.line(image, get_point_x_y(landmark_point[5]), get_point_x_y(landmark_point[9]),
                (255, 255, 255), 2)
        cv.line(image, get_point_x_y(landmark_point[9]), get_point_x_y(landmark_point[13]),
                (0, 0, 0), 6)
        cv.line(image, get_point_x_y(landmark_point[9]), get_point_x_y(landmark_point[13]),
                (255, 255, 255), 2)
        cv.line(image, get_point_x_y(landmark_point[13]), get_point_x_y(landmark_point[17]),
                (0, 0, 0), 6)
        cv.line(image, get_point_x_y(landmark_point[13]), get_point_x_y(landmark_point[17]),
                (255, 255, 255), 2)
        cv.line(image, get_point_x_y(landmark_point[17]), get_point_x_y(landmark_point[0]),
                (0, 0, 0), 6)
        cv.line(image, get_point_x_y(landmark_point[17]), get_point_x_y(landmark_point[0]),
                (255, 255, 255), 2)

    for index, landmark in enumerate(landmark_point):
        if index == 0:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image

def draw_bounding_rect(image, brect):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)

    return image

def draw_info_brect(image, brect):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)

    return image

def draw_jp_input_letters(image, str):
    if str != "":
        margin = 3
        font_size = 30
        image_width, image_height = image.shape[1], image.shape[0]
        #cv.rectangle(
        #    image,
        #    (floor(image_width / 2 - len(str) * font_size / 2 + margin - 4), floor(image_height / 10 - margin)),
        #    (floor(image_width / 2 + len(str) * font_size / 2 + margin - 2), floor(image_height / 10 + font_size + margin)),
        #    (255, 255, 255), -1)

        tmp_Image = Image.fromarray(image)
        d = ImageDraw.Draw(tmp_Image)
        d.font = ImageFont.truetype(Font, 30)
        bw = 1
        pos = [image_width / 2 - len(str) * 15, image_height / 10]
        color_f = (255, 255, 255)
        color_b = (0, 0, 0)

        d.text((pos[0] - bw, pos[1] - bw), str, color_b)
        d.text((pos[0] - bw, pos[1] + bw), str, color_b)
        d.text((pos[0] + bw, pos[1] - bw), str, color_b)
        d.text((pos[0] + bw, pos[1] + bw), str, color_b)
        d.text((pos[0], pos[1]), str, color_f)

        image = np.array(tmp_Image)

    return image

def draw_jp_brect(image, brect, hand_sign_text):
    if hand_sign_text != "":
        info_text = hand_sign_text

        tmp_Image = Image.fromarray(image)
        d = ImageDraw.Draw(tmp_Image)
        d.font = ImageFont.truetype(Font, 30)
        bw = 1
        pos = [round((brect[0] + brect[2])/2), brect[1] - 50]
        color_f = (255, 255, 255)
        color_b = (0, 0, 0)

        d.text((pos[0] - bw, pos[1] - bw), info_text, color_b)
        d.text((pos[0] - bw, pos[1] + bw), info_text, color_b)
        d.text((pos[0] + bw, pos[1] - bw), info_text, color_b)
        d.text((pos[0] + bw, pos[1] + bw), info_text, color_b)
        d.text((pos[0], pos[1]), info_text, color_f)

        image = np.array(tmp_Image)

    return image

def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    if 0 <= mode <= 9:
        cv.putText(image, "MODE:" + str(mode), (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)

    return image

def overlay_Image(image, overlay, brect):
    bw = 10
    overlay = cv.resize(overlay, (brect[2] - brect[0] - bw * 2, brect[3] - brect[1] - bw))

    pil_src = Image.fromarray(image)
    pil_src = pil_src.convert('RGBA')

    pil_overlay = Image.fromarray(overlay)
    pil_overlay = pil_overlay.convert('RGBA')

    pil_tmp = Image.new('RGBA', pil_src.size, (0, 0, 0, 0))
    pil_tmp.paste(pil_overlay, (brect[0] + bw, brect[1] - bw * 2), pil_overlay)
    result_image = Image.alpha_composite(pil_src, pil_tmp)

    image = np.array(result_image)

    return image