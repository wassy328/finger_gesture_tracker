import numpy as np
import datetime
from collections import deque

class Variables:
    def __init__(self, out_number):
        self.__hand_sign_id = out_number
        self.__pre_hand_number = out_number
        self.__point_history = deque(maxlen = 12)
        self.__move_id = 0
        self.__move_lock_tm = self.get_tm_int()
        self.__input_letters = ""
        self.__input_letters_position_x = 0
        self.__letter_lock_id = out_number
        self.__letter_lock_tm = self.get_tm_int()
        self.__del_lock_tm = self.get_tm_int()

    @property
    def hand_sign_id(self): return self.__hand_sign_id
    @hand_sign_id.setter
    def hand_sign_id(self, val): self.__hand_sign_id = val

    @property
    def pre_hand_number(self): return self.__pre_hand_number
    @pre_hand_number.setter
    def pre_hand_number(self, val): self.__pre_hand_number = val

    @property
    def point_history(self): return self.__point_history
    @point_history.setter
    def point_history(self, val): self.__point_history = val

    @property
    def move_id(self): return self.__move_id
    @move_id.setter
    def move_id(self, val): self.__move_id = val

    @property
    def move_lock_tm(self): return self.__move_lock_tm
    @move_lock_tm.setter
    def move_lock_tm(self, val): self.__move_lock_tm = val

    @property
    def input_letters(self): return self.__input_letters
    @input_letters.setter
    def input_letters(self, val): self.__input_letters = val

    @property
    def input_letters_position_x(self): return self.__input_letters_position_x
    @input_letters_position_x.setter
    def input_letters_position_x(self, val): self.__input_letters_position_x = val

    @property
    def letter_lock_id(self): return self.__letter_lock_id
    @letter_lock_id.setter
    def letter_lock_id(self, val): self.__letter_lock_id = val

    @property
    def letter_lock_tm(self): return self.__letter_lock_tm
    @letter_lock_tm.setter
    def letter_lock_tm(self, val): self.__letter_lock_tm = val

    @property
    def del_lock_tm(self): return self.__del_lock_tm
    @del_lock_tm.setter
    def del_lock_tm(self, val): self.__del_lock_tm = val

    def get_move_id(self, hand_landmarks, landmarks_ave):
        if self.pre_hand_number != self.hand_sign_id:
            if self.get_tm_int() - self.move_lock_tm > 100:
                for i in range(12): self.point_history.append(landmarks_ave)
                self.pre_hand_number = self.hand_sign_id
                self.move_lock_tm = self.get_tm_int()
                self.move_id = 0

        elif self.move_id == 0:
            t_v1, t_v2 = 1.03, 1.05
            self.point_history.append(landmarks_ave)

            sum_p1 = np.mean(np.array(list(self.point_history)[0:3]), axis = 0)
            sum_p2 = np.mean(np.array(list(self.point_history)[4:7]), xis = 0)
            sum_p3 = np.mean(np.array(list(self.point_history)[8:11]), axis = 0)

            if sum_p1[0] < sum_p2[0] and sum_p2[0]*t_v1 < sum_p3[0] and sum_p1[0]*t_v2 < sum_p3[0]: self.move_id = 2
            elif sum_p1[1] < sum_p2[1] and sum_p2[1]*t_v1 < sum_p3[1] and sum_p1[1]*t_v2 < sum_p3[1]: self.move_id = 1
            elif sum_p3[1] < sum_p2[1] and sum_p2[1]*t_v1 < sum_p1[1] and sum_p3[1]*t_v2 < sum_p1[1]: self.move_id = 3

            print("move_id: " + str(self.move_id))
            self.move_unlock_tm = self.get_tm_int()
            
        elif self.get_tm_int() - self.move_unlock_tm > 100:
            self.move_id = 0
            for i in range(12): self.point_history.append(landmarks_ave)

    def get_hand_sign_id(self, landmark_tips, hand_landmarks):
        if self.hand_sign_id == 0:
            if landmark_tips[0] in [8]: self.hand_sign_id = 16 
            elif hand_landmarks.landmark[4].y <= hand_landmarks.landmark[15].y: self.hand_sign_id = 11 
            elif hand_landmarks.landmark[12].y <= hand_landmarks.landmark[19].y: self.hand_sign_id = 43 
            elif hand_landmarks.landmark[16].y <= hand_landmarks.landmark[19].y: self.hand_sign_id = 27 
            else: self.hand_sign_id = 26 
        elif self.hand_sign_id == 1:
            if landmark_tips[2] in [8]: self.hand_sign_id = 34 
            elif landmark_tips[0] in [5,9]: self.hand_sign_id = 24 
            elif landmark_tips[0] in [6,7]: self.hand_sign_id = 32 
            elif landmark_tips[0] in [8]: 
                if hand_landmarks.landmark[12].x <= hand_landmarks.landmark[4].x and not hand_landmarks.landmark[2].y <= hand_landmarks.landmark[12].y: self.hand_sign_id = 15 
                elif hand_landmarks.landmark[8].x <= hand_landmarks.landmark[10].x: self.hand_sign_id = 36 
                else: self.hand_sign_id = 55 
        elif self.hand_sign_id == 2:
            if landmark_tips[3] in [1,2,5]: self.hand_sign_id = 29 
            elif landmark_tips[3] in [8,12]: self.hand_sign_id = 35 
            elif landmark_tips[0] in [12]: 
                if hand_landmarks.landmark[8].x <= hand_landmarks.landmark[12].x: self.hand_sign_id = 12 
                else: self.hand_sign_id = 48 
            elif landmark_tips[0] in [0,1,9]: self.hand_sign_id = 30 
            elif landmark_tips[0] in [5]: self.hand_sign_id = 49 
            elif landmark_tips[2] in [12]: self.hand_sign_id = 31 
        elif self.hand_sign_id == 3:
            if landmark_tips[0] in [0,1]: self.hand_sign_id = 40 
            elif landmark_tips[2] in [12]: self.hand_sign_id = 41 
            elif landmark_tips[3] in [1,2,5,6,7]: self.hand_sign_id = 46 
            elif landmark_tips[2] in [2,8]: self.hand_sign_id = 53 
        elif self.hand_sign_id == 5: 
            if landmark_tips[2] in [4]: self.hand_sign_id = 10 
            elif landmark_tips[0] in [4]: self.hand_sign_id = 25 
        elif self.hand_sign_id == 6: 
            if landmark_tips[1] in [8]: self.hand_sign_id = 37 
            elif landmark_tips[2] in [8]: self.hand_sign_id = 42 
            elif landmark_tips[0] in [8]: self.hand_sign_id = 51 
        elif self.hand_sign_id == 7: 
            if landmark_tips[2] in [12]: self.hand_sign_id = 21 
            elif landmark_tips[1] in [12,17]: self.hand_sign_id = 22 
            elif landmark_tips[0] in [8,12]: self.hand_sign_id = 50 
        elif self.hand_sign_id == 8: 
            if landmark_tips[0] in [0,4,5,9,13]: self.hand_sign_id = 38 
            elif landmark_tips[1] in [0]: self.hand_sign_id = 45 
        elif self.hand_sign_id == 9: 
            if landmark_tips[0] in [4]: self.hand_sign_id = 17 
            elif landmark_tips[2] in [12]: self.hand_sign_id = 47 
            elif landmark_tips[0] in [10,11]: self.hand_sign_id = 13 
            elif landmark_tips[2] in [17,18,19,20]: self.hand_sign_id = 39 
            elif landmark_tips[2] not in [3,4]: self.hand_sign_id = 18 
            elif landmark_tips[0] in [12]: self.hand_sign_id = 28 
            elif landmark_tips[0] in [0]: self.hand_sign_id = 33 

    def fix_hand_sign_id(self):
        if self.move_id != 0:
            if self.hand_sign_id == 14 and self.move_id == 1: self.hand_sign_id = 54 
            elif self.hand_sign_id == 15 and self.move_id == 2: self.hand_sign_id = 56 
            elif self.hand_sign_id == 16 and self.move_id == 2: self.hand_sign_id = 57 
            elif self.hand_sign_id == 17 and self.move_id == 2: self.hand_sign_id = 58 
            elif self.hand_sign_id == 18 and self.move_id == 2: self.hand_sign_id = 59 
            elif self.hand_sign_id == 19 and self.move_id == 2: self.hand_sign_id = 60 
            elif self.hand_sign_id == 20 and self.move_id == 2: self.hand_sign_id = 61 
            elif self.hand_sign_id == 21 and self.move_id == 2: self.hand_sign_id = 62 
            elif self.hand_sign_id == 22 and self.move_id == 2: self.hand_sign_id = 63 
            elif self.hand_sign_id == 23 and self.move_id == 2: self.hand_sign_id = 64 
            elif self.hand_sign_id == 24 and self.move_id == 2: self.hand_sign_id = 65 
            elif self.hand_sign_id == 25 and self.move_id == 2: self.hand_sign_id = 66 
            elif self.hand_sign_id == 26 and self.move_id == 2: self.hand_sign_id = 67 
            elif self.hand_sign_id == 27 and self.move_id == 2: self.hand_sign_id = 68 
            elif self.hand_sign_id == 28 and self.move_id == 2: self.hand_sign_id = 69 
            elif self.hand_sign_id == 29 and self.move_id == 2: self.hand_sign_id = 70 
            elif self.hand_sign_id == 35 and self.move_id == 2: self.hand_sign_id = 71 
            elif self.hand_sign_id == 36 and self.move_id == 2: self.hand_sign_id = 72 
            elif self.hand_sign_id == 37 and self.move_id == 2: self.hand_sign_id = 73 
            elif self.hand_sign_id == 38 and self.move_id == 2: self.hand_sign_id = 74 
            elif self.hand_sign_id == 39 and self.move_id == 2: self.hand_sign_id = 75 
            elif self.hand_sign_id == 35 and self.move_id == 3: self.hand_sign_id = 76 
            elif self.hand_sign_id == 36 and self.move_id == 3: self.hand_sign_id = 77 
            elif self.hand_sign_id == 37 and self.move_id == 2: self.hand_sign_id = 73 
            elif self.hand_sign_id == 38 and self.move_id == 2: self.hand_sign_id = 74 
            elif self.hand_sign_id == 39 and self.move_id == 2: self.hand_sign_id = 75 
            elif self.hand_sign_id == 35 and self.move_id == 3: self.hand_sign_id = 76 
            elif self.hand_sign_id == 36 and self.move_id == 3: self.hand_sign_id = 77 
            elif self.hand_sign_id == 37 and self.move_id == 3: self.hand_sign_id = 78 
            elif self.hand_sign_id == 38 and self.move_id == 3: self.hand_sign_id = 79 
            elif self.hand_sign_id == 39 and self.move_id == 3: self.hand_sign_id = 80 
            elif self.hand_sign_id == 27 and self.move_id == 1: self.hand_sign_id = 81 
            elif self.hand_sign_id == 45 and self.move_id == 1: self.hand_sign_id = 82 
            elif self.hand_sign_id == 46 and self.move_id == 1: self.hand_sign_id = 83 
            elif self.hand_sign_id == 47 and self.move_id == 1: self.hand_sign_id = 84 
            elif self.hand_sign_id == 24 and self.move_id == 1: self.hand_sign_id = 85 

    def get_input_letters(self, keypoint_classifier_labels):
        if self.letter_lock_id == self.hand_sign_id:
            if self.get_tm_int() - self.letter_lock_tm > 400:
                self.input_letters = self.input_letters + keypoint_classifier_labels[self.hand_sign_id]
                self.letter_lock_tm = self.get_tm_int()
            elif self.get_tm_int() - self.letter_lock_tm > 20:
                if len(self.input_letters) == 0:
                    self.input_letters = self.input_letters + keypoint_classifier_labels[self.hand_sign_id]
                elif self.input_letters[-1] == "お" and self.hand_sign_id == 54: self.input_letters = self.input_letters[0:-1] + "を"
                elif self.input_letters[-1] == "か" and self.hand_sign_id == 56: self.input_letters = self.input_letters[0:-1] + "が"
                elif self.input_letters[-1] == "き" and self.hand_sign_id == 57: self.input_letters = self.input_letters[0:-1] + "ぎ"
                elif self.input_letters[-1] == "く" and self.hand_sign_id == 58: self.input_letters = self.input_letters[0:-1] + "ぐ"
                elif self.input_letters[-1] == "け" and self.hand_sign_id == 59: self.input_letters = self.input_letters[0:-1] + "げ"
                elif self.input_letters[-1] == "こ" and self.hand_sign_id == 60: self.input_letters = self.input_letters[0:-1] + "ご"
                elif self.input_letters[-1] == "さ" and self.hand_sign_id == 61: self.input_letters = self.input_letters[0:-1] + "ざ"
                elif self.input_letters[-1] == "し" and self.hand_sign_id == 62: self.input_letters = self.input_letters[0:-1] + "じ"
                elif self.input_letters[-1] == "す" and self.hand_sign_id == 63: self.input_letters = self.input_letters[0:-1] + "ず"
                elif self.input_letters[-1] == "せ" and self.hand_sign_id == 64: self.input_letters = self.input_letters[0:-1] + "ぜ"
                elif self.input_letters[-1] == "そ" and self.hand_sign_id == 65: self.input_letters = self.input_letters[0:-1] + "ぞ"
                elif self.input_letters[-1] == "た" and self.hand_sign_id == 66: self.input_letters = self.input_letters[0:-1] + "だ"
                elif self.input_letters[-1] == "ち" and self.hand_sign_id == 67: self.input_letters = self.input_letters[0:-1] + "ぢ"
                elif self.input_letters[-1] == "つ" and self.hand_sign_id == 68: self.input_letters = self.input_letters[0:-1] + "づ"
                elif self.input_letters[-1] == "て" and self.hand_sign_id == 69: self.input_letters = self.input_letters[0:-1] + "で"
                elif self.input_letters[-1] == "と" and self.hand_sign_id == 70: self.input_letters = self.input_letters[0:-1] + "ど"
                elif self.input_letters[-1] == "は" and self.hand_sign_id == 71: self.input_letters = self.input_letters[0:-1] + "ば"
                elif self.input_letters[-1] == "ひ" and self.hand_sign_id == 72: self.input_letters = self.input_letters[0:-1] + "び"
                elif self.input_letters[-1] == "ふ" and self.hand_sign_id == 73: self.input_letters = self.input_letters[0:-1] + "ぶ"
                elif self.input_letters[-1] == "へ" and self.hand_sign_id == 74: self.input_letters = self.input_letters[0:-1] + "べ"
                elif self.input_letters[-1] == "ほ" and self.hand_sign_id == 75: self.input_letters = self.input_letters[0:-1] + "ぼ"
                elif self.input_letters[-1] == "は" and self.hand_sign_id == 76: self.input_letters = self.input_letters[0:-1] + "ぱ"
                elif self.input_letters[-1] == "ひ" and self.hand_sign_id == 77: self.input_letters = self.input_letters[0:-1] + "ぴ"
                elif self.input_letters[-1] == "ふ" and self.hand_sign_id == 78: self.input_letters = self.input_letters[0:-1] + "ぷ"
                elif self.input_letters[-1] == "へ" and self.hand_sign_id == 79: self.input_letters = self.input_letters[0:-1] + "ぺ"
                elif self.input_letters[-1] == "ほ" and self.hand_sign_id == 80: self.input_letters = self.input_letters[0:-1] + "ぽ"
                elif self.input_letters[-1] == "つ" and self.hand_sign_id == 81: self.input_letters = self.input_letters[0:-1] + "っ"
                elif self.input_letters[-1] == "や" and self.hand_sign_id == 82: self.input_letters = self.input_letters[0:-1] + "ゃ"
                elif self.input_letters[-1] == "ゆ" and self.hand_sign_id == 83: self.input_letters = self.input_letters[0:-1] + "ゅ"
                elif self.input_letters[-1] == "よ" and self.hand_sign_id == 84: self.input_letters = self.input_letters[0:-1] + "ょ"
                elif self.input_letters[-1] == "そ" and self.hand_sign_id == 85: self.input_letters = self.input_letters[0:-1] + "ー"

                elif self.input_letters[-1] != keypoint_classifier_labels[self.hand_sign_id][-1]:
                    self.input_letters = self.input_letters + keypoint_classifier_labels[self.hand_sign_id]
        else:
            self.letter_lock_id = self.hand_sign_id
            self.letter_lock_tm = self.get_tm_int()

    def get_tm_int(self):
        return int(datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')[:16])