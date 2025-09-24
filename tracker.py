# imports 
from attr import dataclass
import cv2
import mediapipe as mp
import numpy as np
import time
import pyvista as pv
import yaml

from utils import get_screen_size, get_landmarks, process_landmarks
from manifold_fitter import ManifoldFitter

# Constants
ALERT_FRAMES = 50

LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

LEFT_EYE_CORNERS = [33, 133]
RIGHT_EYE_CORNERS = [362, 263]

LEFT_EYE_LIDS = [159, 145]
RIGHT_EYE_LIDS = [386, 374]

LEFT_EYE_OUTLINE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
#                   22, 23, 24, 25, 26, 27, 28, 29, 30, 56, 110, 112, 130, 190, 247]
RIGHT_EYE_OUTLINE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
#                   252, 253, 254, 255, 256, 257, 258, 259, 260, 286, 339, 359, 414, 467]

indices = np.ones(478)
indices[LEFT_IRIS] = 2
indices[RIGHT_IRIS] = 2
indices[LEFT_EYE_OUTLINE] = 3
indices[RIGHT_EYE_OUTLINE] = 3

PLANE_INDICES = [168, 6, 152, 234, 454] 

SAVE_INTERVAL = 10
OUTPUT_FILENAME = 'save.xyz'

NORMAL_DISTANCE = 180
ORIGIN_X = 320
ORIGIN_Y = 240
ORIGIN_Z = 0
ORIGIN = [ORIGIN_X, ORIGIN_Y, ORIGIN_Z]
Z_CONSTANT = 100

# constants
MODEL_PATH = "face_landmarker.task"

# global vairables
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def draw_landmarks_on_image(rgb_image, detection_result, indices=None):
    """
    Функция для отрисовки ориентиров на изображении.
    """
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    # Перебираем обнаруженные лица
    for face_landmarks in face_landmarks_list:
        if indices is None:
            indices = range(len(face_landmarks))
        # Перебираем ориентиры на лице
        for landmark in [face_landmarks[i] for i in indices]:
            x = int(landmark.x * annotated_image.shape[1])
            y = int(landmark.y * annotated_image.shape[0])
            # Рисуем маленькую зеленую точку на месте каждого ориентира
            cv2.circle(annotated_image, (x, y), 1, (0, 255, 0), -1)
            
    return annotated_image

def process_frame(cap, landmarker) -> bool:
        global off_center_count, frame_counter
        # Чтение кадра с камеры
        success, frame = cap.read()
        h, w, _ = frame.shape
        if not success:
            print("Не удалось получить кадр с камеры.")
            return False
        
        frame_counter += 1
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        face_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
        
        annotated_frame = frame # По умолчанию используем оригинальный кадр
        annotated_frame = draw_landmarks_on_image(frame, face_landmarker_result, indices=LEFT_IRIS+RIGHT_IRIS)
        cv2.imshow('3D Face Landmarks', annotated_frame)
        if off_center_count > ALERT_FRAMES:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.3, annotated_frame, 0.7, 0, annotated_frame)
            cv2.putText(annotated_frame, 'FOCUS!', (w // 3, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4)
        cv2.imshow('3D Face Landmarks', annotated_frame)
        
        
        w, h = get_screen_size()
        blank_frame = np.zeros((h, w, 3), dtype=np.uint8)
        landmarks_array = get_landmarks(face_landmarker_result,
                                        w,
                                        h,
                                        as_array=True)
        eye_direction = process_landmarks(landmarks_array)[1]
        l = np.array(eye_direction)
        modelpath = 'calibration/daniil/params.npz'
        f = ManifoldFitter(calibration_dir='calibration/daniil')
        f.init_from_file(modelpath)
        u, v = f.infer_one(landmarks_array)
        print(u, v, l)
        u = np.clip(u, 0, w)
        v = np.clip(v, 0, h)
        frame = cv2.circle(
            blank_frame,
            (int(u), int(v)),
            10,
            (0,0,255),
            thickness=-1
        )
        cv2.imshow('CALIBRATION', blank_frame)
        
        if frame_counter % SAVE_INTERVAL == 0:
            face_points = pv.PolyData(landmarks_array)
            face_vector = pv.Arrow(start=np.mean(landmarks_array, axis=0), direction=l*500, scale=200)
            
            scene = pv.MultiBlock()
            scene.append(face_points, name="face_points")
            scene.append(face_vector, name="face_vector")
            
            scene.save(f"saved/scene_frame_{(frame_counter // SAVE_INTERVAL):04d}.vtm")
        
        return success

off_center_count = 0
frame_counter = 0

options = FaceLandmarkerOptions(
base_options=BaseOptions(model_asset_path=MODEL_PATH),
running_mode=VisionRunningMode.VIDEO,
num_faces=1  # Ограничиваемся обнаружением одного лица для производительности
)

def main() -> None: 
    success = True
    # main loop
    with FaceLandmarker.create_from_options(options) as landmarker:
        # Инициализация видеозахвата с веб-камеры (0 - обычно встроенная камера)
        cap = cv2.VideoCapture(0)
        while success:
            success = process_frame(cap, landmarker)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()