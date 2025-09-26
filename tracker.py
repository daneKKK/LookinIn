# imports 
from attr import dataclass
import cv2
import mediapipe as mp
import numpy as np
import time
import pyvista as pv
import yaml
from controller import Controller


from attention import AttentionTracker, AffineTransformationModel, EnsembleModel

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

FOCUS_TIME = 0.3
DELAY = 5.0

# constants
MODEL_PATH = "face_landmarker.task"

# global vairables
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


#M = np.load('calibration/daniil/M.npy')
#model = AffineTransformationModel(M)
model = EnsembleModel()
model.load('calibration/demo/M.npy', 'calibration/demo/cb')
attention_tracker = AttentionTracker(model, get_screen_size()[0], get_screen_size()[1])

bracelet = False

def connect_controller():
    global bracelet
    bracelet = Controller()
    return

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
    
def draw_calibration_frame(face_landmarker_result, w, h):
    screen_w, screen_h = get_screen_size()
    u, v = None, None
    try:
        blank_frame = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        landmarks_array = get_landmarks(face_landmarker_result,
                                        w,
                                        h,
                                        as_array=True)
        eye_direction = process_landmarks(landmarks_array)[1] 
        l = np.array(eye_direction)
        
        uv, is_unfocused = track_attention(landmarks_array)
        u = np.clip(uv[0], 0, screen_w)
        v = np.clip(uv[1], 0, screen_h)
        blank_frame = cv2.circle(
            blank_frame,
            (int(u), int(v)),
            10,
            (0,0,255),
            thickness=-1
        )
        if is_unfocused:
            cv2.putText(blank_frame, 'FOCUS', (w // 3, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4)
    except IndexError:
        pass
    except ValueError:
        pass
    except FileNotFoundError:
        pass
    cv2.imshow('CALIBRATION', blank_frame)
    return u, v
    
def track_attention(landmarks_array):
    global attention_tracker
    uv = attention_tracker.eval(landmarks_array)
    is_unfocused = (attention_tracker.attention_check(landmarks_array))
    if is_unfocused:
        print('AAAAAAA')
        if bracelet:
            bracelet.on(FOCUS_TIME, DELAY)
    return uv, is_unfocused

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
        
        draw_calibration_frame(face_landmarker_result, w, h)
        
        
        #if frame_counter % SAVE_INTERVAL == 0:
        #    landmarks_array = get_landmarks(face_landmarker_result,
        #                                    w,
        #                                    h,
        #                                    as_array=True)
        #    _, l, eyeballs, landmarks_array = process_landmarks(landmarks_array, debug=True)
        #    face_points = pv.PolyData(landmarks_array)
        #    face_vector = pv.Arrow(start=np.mean(landmarks_array, axis=0), direction=l*500, scale=200)
        #    left_eye_sphere = pv.Sphere(radius=eyeballs[0][0], center=eyeballs[0][1])
        #    right_eye_sphere = pv.Sphere(radius=eyeballs[1][0], center=eyeballs[1][1])
        #    #
        #    scene = pv.MultiBlock()
        #    scene.append(face_points, name="face_points")
        #    scene.append(face_vector, name="face_vector")
        #    scene.append(left_eye_sphere, name="leye")
        #    scene.append(right_eye_sphere, name="reye")
        #    #
        #    scene.save(f"saved/scene_frame_{(frame_counter // SAVE_INTERVAL):04d}.vtm")
        #    
        #    
        #    # Открываем файл в режиме 'a' (append/добавление)
        #    with open('save.xyz', 'a') as f:
        #        # 1. Записываем количество точек
        #        f.write(f"{len(landmarks_array)}\n")
        #        
        #        # 2. Записываем комментарий с номером кадра
        #        f.write(f"# Frame number {frame_counter}\n")
        #        
        #        # 3. Записываем координаты всех точек
        #        for i, landmark in enumerate(landmarks_array):
        #            # Форматируем строку: "1 x y z"
        #            # Обратите внимание на `1.0 - landmark.y` для инверсии оси Y
        #            line = f"{landmark[0]:.6f} {landmark[1]:.6f} {landmark[2]:.6f}\n"
        #            f.write(line)
        
        return success

off_center_count = 0
frame_counter = 0

options = FaceLandmarkerOptions(
base_options=BaseOptions(model_asset_path=MODEL_PATH),
running_mode=VisionRunningMode.VIDEO,
num_faces=1  # Ограничиваемся обнаружением одного лица для производительности
)

def main() -> None: 
    connect_controller()
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