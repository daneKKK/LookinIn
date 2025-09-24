import time
import cv2
import numpy as np
from pyvista import set_default_active_vectors
import yaml
from utils import get_landmarks, get_screen_size
from tracker import FaceLandmarker, options
import os
import os.path as osp
import mediapipe as mp


class Calibrator:
    def __init__(self,
                 save_folder: str | None = None,
                 user_name: str | None = None,
                 ) -> None:
        self.landmarker = FaceLandmarker.create_from_options(options)
        self.save_folder = save_folder
        self.user_name = user_name
        self.frame_counter = 0
        self.w, self.h = get_screen_size()
        
        self.background_color = np.array((0, 0, 0), dtype=np.uint8)

        self.circle_color = (0, 0, 255)
        self.circle_size = int(self.w / 40) # arbitrary

        self.x_steps = 5
        self.y_steps = 5 # total of 5 x 5 steps

        x_start = 2 * self.circle_size
        x_end = self.w - 2 * self.circle_size

        y_start = 2 * self.circle_size
        y_end = self.h - 2 * self.circle_size

        x_positions = np.linspace(x_start, x_end, self.x_steps)
        y_positions = np.linspace(y_start, y_end, self.y_steps)

        self.circle_centers = [(int(x), int(y)) for y in y_positions for x in x_positions ]
        self.cur_step = 0

        self.is_warning = True

        self.time_per_circle_sec = 0.5

        self.saved_landmarks = []

        self.save_path = osp.join(save_folder, user_name)
        if not osp.exists(osp.join(save_folder, user_name)):
            os.makedirs(self.save_path)

    def create_frame(self,
                    ) -> np.ndarray:
        blank_frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        blank_frame[:, :] = self.background_color

        if not self.is_warning:
            frame = cv2.circle(
                blank_frame,
                self.circle_centers[self.cur_step],
                self.circle_size,
                self.circle_color,
                thickness=-1
            )
            frame = cv2.circle(
                blank_frame,
                self.circle_centers[self.cur_step],
                int(self.circle_size / 3),
                (255, 255, 255),
                thickness=-1
            )
            self.cur_step += 1
            self.is_warning = True
        else:
            frame = cv2.circle(
                blank_frame,
                self.circle_centers[self.cur_step],
                self.circle_size,
                (122, 122, 122),
                thickness=-1
            )
            self.is_warning = False
        return frame

    def show_frame(self):
        frame = self.create_frame()
        cv2.imshow('current_frame', frame)
        key = cv2.waitKey(int(self.time_per_circle_sec * 1000))
        if key == ord('q'):
            raise KeyboardInterrupt()

    def get_webcam_frame(self, cap: cv2.VideoCapture):
        success, frame = cap.read()
        if success:
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            frame_timestamp_ms = int(time.time() * 1000)
            face_landmarker_result = self.landmarker.detect_for_video(
                mp_image,
                frame_timestamp_ms
                )
            landmark_dict = get_landmarks(face_landmarker_result,
                                          self.w,
                                          self.h,
                                          )
            self.saved_landmarks.append(landmark_dict)
            with open(f"{self.save_path}/{self.cur_step}.yaml", 'w') as f:
                yaml.dump(landmark_dict, f)
                
        else:
            print("WEBCAM DID NOT WORK")
            self.saved_landmarks.append(None)
    
    def show_starting_screen(self):
        blank_frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        blank_frame[:, :] = self.background_color
        frame = cv2.putText(blank_frame,
                            "PRESS S WHEN READY",
                            (self.h // 2, self.w // 10), 
                            cv2.FONT_HERSHEY_COMPLEX,
                            1.0,
                            (255, 255, 255),
                            1,
                            2)
        cv2.imshow('CALIBRATION', frame)
        key = cv2.waitKey(int(1e6))
        if key == ord('s'):
            cv2.destroyAllWindows()
            return
    
    def run(self,
            ):
        cap = cv2.VideoCapture(0)
        self.success = True
        self.show_starting_screen()
        while True:
            self.show_frame()
            if not self.is_warning:
                self.get_webcam_frame(cap)
        

if __name__ == "__main__":
    calib = Calibrator(
        save_folder="calibration",
        user_name="stas",
        )
    calib.run()

