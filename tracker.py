import cv2
import mediapipe as mp
import numpy as np
import time
import pyvista as pv

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

# Определяем необходимые классы из MediaPipe
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

SAVE_INTERVAL = 10
OUTPUT_FILENAME = 'save.xyz'

NORMAL_DISTANCE = 180
ORIGIN_X = 320
ORIGIN_Y = 240
ORIGIN_Z = 0
ORIGIN = [ORIGIN_X, ORIGIN_Y, ORIGIN_Z]
Z_CONSTANT = 100

def scale(x, origin, scale):
    x = (x - origin) * scale + origin
    return x
    
def fit_sphere(points):
    """Sphere fitting using Linear Least Squares"""

    A = np.column_stack((2*points, np.ones(len(points))))
    b = (points**2).sum(axis=1)
    x, res, _, _ = np.linalg.lstsq(A, b)
    center = x[:3]
    radius = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2 + x[3])
   
    return (center, radius), res
    
def vector_angle(u, v):
    u_m = np.linalg.norm(u)
    v_m = np.linalg.norm(v)
    theta = np.arccos(np.dot(u, v) / (u_m * v_m))
    return theta

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
    
def get_eye_angle(frame, detection_result, iris_idxs, corner_idxs, lid_idxs, outline_idxs):
    # ——— Outline ———
    pts = np.array([
        (face_landmarks.landmark[i].x,
         face_landmarks.landmark[i].y,
         face_landmarks.landmark[i].z
         )
        for i in outline_idxs
    ])
    corners = []
    for i in corner_idxs:
        x = face_landmarks.landmark[i].x
        y = face_landmarks.landmark[i].y
        z = face_landmarks.landmark[i].z
        corners.append((x, y, z))
    corners = np.array(corners)
    lids = []
    for i in lid_idxs:
        x = face_landmarks.landmark[i].x
        y = face_landmarks.landmark[i].y
        z = face_landmarks.landmark[i].z
        lids.append((x, y, z))
    lids = np.array(lids)
    iris_pts = []
    for i in iris_idxs:
        x = int(face_landmarks.landmark[i].x * w)
        y = int(face_landmarks.landmark[i].y * h)
        iris_pts.append((x, y))
    iris_pts = np.array(iris_pts)
    
def fit_plane_and_get_normal(points):
    centroid = points.mean(axis=0)
    centered_points = points - centroid
    _, _, vh = np.linalg.svd(centered_points)
    normal = vh[2, :]
    return normal

def face_normal(face_landmarks_3d, h, w):
    '''
    Функция для расчёта нормали по точкам лица из PLANE_INDICES.
    
    face_landmarks_3d - лэндмарки лица
    h - высота экрана
    w - ширина экрана
    '''
    PLANE_INDICES = [168, 6, 152, 234, 454]
    points = np.array([(face_landmarks_3d[i].x * w, 
                        (1 - face_landmarks_3d[i].y) * h, 
                        face_landmarks_3d[i].z * w) for i in PLANE_INDICES])
    points = np.array([(l.x, l.y, l.z) for l in face_landmarks_3d])
    return fit_plane_and_get_normal(points)
    
def process_face(face_landmarks_3d, h, w, frame_counter):
    norm = face_normal(face_landmarks_3d, h, w)
    face_landmarks_3d = np.array([(l.x, l.y, l.z) for l in face_landmarks_3d])
    face_landmarks_3d[:, 0] *= w
    face_landmarks_3d[:, 1] = h*(1 - face_landmarks_3d[:, 1])
    face_landmarks_3d[:, 2] *= w
    
    face_landmarks_3d[:, 0] -= w/2
    face_landmarks_3d[:, 1] -= h/2
    
    
    scaling_factor = NORMAL_DISTANCE / np.linalg.norm(face_landmarks_3d[127]-face_landmarks_3d[356])
    print(scaling_factor)
    #face_landmarks_3d = scale(face_landmarks_3d, ORIGIN, NORMAL_DISTANCE / scaling_factor)
    face_landmarks_3d *= scaling_factor
    face_landmarks_3d[:, 2] += scaling_factor * NORMAL_DISTANCE
    
    # Fitting eyes to the sphere
    left_eye_sphere_c, left_eye_sphere_r = fit_sphere(face_landmarks_3d[LEFT_EYE_OUTLINE])[0]
    right_eye_sphere_c, right_eye_sphere_r = fit_sphere(face_landmarks_3d[RIGHT_EYE_OUTLINE])[0]
    #print(right_eye_sphere_r * 0.5 + left_eye_sphere_r * 0.5)
    
    # We need to explicitly calculate iris 3d coordinates by taking their projections on the eyes spheres
    for i in LEFT_IRIS:
        face_landmarks_3d[i][2] = (left_eye_sphere_c[2] - np.sqrt(left_eye_sphere_r**2 - 
                                                                 (face_landmarks_3d[i][0]-left_eye_sphere_c[0])**2 - 
                                                                 (face_landmarks_3d[i][1]-left_eye_sphere_c[1])**2))
    #We also calculate eye direction
    left_iris_c = face_landmarks_3d[LEFT_IRIS[0]]
    left_eye_dir = left_iris_c - left_eye_sphere_c
    left_eye_dir /= np.linalg.norm(left_eye_dir)
                                                                 
    for i in RIGHT_IRIS:
        face_landmarks_3d[i][2] = (right_eye_sphere_c[2] - np.sqrt(right_eye_sphere_r**2 - 
                                                                 (face_landmarks_3d[i][0]-right_eye_sphere_c[0])**2 - 
                                                                 (face_landmarks_3d[i][1]-right_eye_sphere_c[1])**2))
    right_iris_c = face_landmarks_3d[RIGHT_IRIS[0]]
    right_eye_dir = right_iris_c - right_eye_sphere_c
    right_eye_dir /= np.linalg.norm(right_eye_dir)
    
    #Calculate face direction by fitting the plane
    face_dir = fit_plane_and_get_normal(face_landmarks_3d[PLANE_INDICES])
    face_dir /= np.linalg.norm(face_dir)
    face_dir *= (1 if face_dir[2] < 0 else -1)
    face_angle = vector_angle(face_dir, np.array([0, 0, -1]))
    face_angle = np.min([face_angle, np.pi - face_angle]) * 180 / np.pi
    
    left_eye_angle = vector_angle(left_eye_dir, np.array([0, 0, -1]))
    left_eye_angle = np.min([left_eye_angle, np.pi-left_eye_angle]) * 180 / np.pi
    right_eye_angle = vector_angle(right_eye_dir, np.array([0, 0, -1]))
    right_eye_angle = np.min([right_eye_angle, np.pi-right_eye_angle]) * 180 / np.pi 
    
    
    left_outline_z = np.mean([face_landmarks_3d[i][2] for i in LEFT_EYE_OUTLINE])
    right_outline_z = np.mean([face_landmarks_3d[i][2] for i in RIGHT_EYE_OUTLINE])
    
    z_descriptor = min(right_eye_sphere_r, left_eye_sphere_r)
    x_left_descriptor, y_left_descriptor = left_eye_dir[0], left_eye_dir[1]
    x_right_descriptor, y_right_descriptor = right_eye_dir[0], right_eye_dir[1]
    face_x_pos_descriptor, face_y_pos_descriptor = np.mean(face_landmarks_3d, axis=0)[0], np.mean(face_landmarks_3d, axis=0)[1]
    
    descriptors = [z_descriptor, x_left_descriptor, y_left_descriptor,
                   x_right_descriptor, y_right_descriptor,
                   face_x_pos_descriptor, face_y_pos_descriptor]
    
    # --- ЛОГИКА СОХРАНЕНИЯ В ФАЙЛ ---
    # Проверяем, настал ли момент для сохранения
    # --- ЛОГИКА СОХРАНЕНИЯ В ФАЙЛ ---
    # Проверяем, настал ли момент для сохранения
    if frame_counter % SAVE_INTERVAL == 0:
        face_points = pv.PolyData(face_landmarks_3d)
        left_eye_sphere = pv.Sphere(radius=left_eye_sphere_r, center=left_eye_sphere_c)
        left_eye_vector = pv.Arrow(start=left_eye_sphere_c, direction=left_eye_dir, scale=left_eye_sphere_r*5)
        right_eye_sphere = pv.Sphere(radius=right_eye_sphere_r, center=right_eye_sphere_c)
        right_eye_vector = pv.Arrow(start=right_eye_sphere_c, direction=right_eye_dir, scale=right_eye_sphere_r*5)
        face_vector = pv.Arrow(start=np.mean(face_landmarks_3d, axis=0), direction=face_dir*500, scale=200)
        
        scene = pv.MultiBlock()
        scene.append(face_points, name="face_points")
        scene.append(left_eye_sphere, name="left_eye_sphere")
        scene.append(right_eye_sphere, name="right_eye_sphere")
        scene.append(left_eye_vector, name="left_eye_vector")
        scene.append(right_eye_vector, name="right_eye_vector")
        scene.append(face_vector, name="face_vector")
        
        scene.save(f"saved/scene_frame_{(frame_counter // SAVE_INTERVAL):04d}.vtm")
        
        # Открываем файл в режиме 'a' (append/добавление)
        with open(OUTPUT_FILENAME, 'a') as f:
            # 1. Записываем количество точек
            f.write(f"{len(face_landmarks_3d)}\n")
            
            # 2. Записываем комментарий с номером кадра
            f.write(f"# Frame number {frame_counter}\n")
            
            # 3. Записываем координаты всех точек
            for i, landmark in enumerate(face_landmarks_3d):
                # Форматируем строку: "1 x y z"
                # Обратите внимание на `1.0 - landmark.y` для инверсии оси Y
                line = f"{indices[i]} {landmark[0]:.6f} {landmark[1]:.6f} {landmark[2]:.6f}\n"
                f.write(line)
    return descriptors, face_landmarks_3d

def main():
    # --- НАСТРОЙКА ДЕТЕКТОРА ---
    # Состояния приложения: IDLE, CALIBRATING, PREDICTING
    app_state = 'IDLE'
    gaze_model = None
    #data_collector = DataCollector()
    
    # Путь к файлу модели
    model_path = 'face_landmarker.task'

    # Настройка опций для FaceLandmarker
    # Используем режим VIDEO для обработки потокового видео
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1  # Ограничиваемся обнаружением одного лица для производительности
    )
    
    # Создаем точки для калибровки (сетка 5x3 с отступами)
    #margin = 0.15
    #x_points = np.linspace(screen_w * margin, screen_w * (1 - margin), 5, dtype=int)
    #y_points = np.linspace(screen_h * margin, screen_h * (1 - margin), 3, dtype=int)
    
    #calibration_points = [(x, y) for y in y_points for x in x_points]
    current_calibration_point = 0

    # --- ОСНОВНОЙ ЦИКЛ ---
    
    # Инициализируем счетчик кадров
    frame_counter = 0
    
    
    off_center_count = 0
    

    # Создаем экземпляр FaceLandmarker внутри 'with' для корректного управления ресурсами
    with FaceLandmarker.create_from_options(options) as landmarker:
        # Инициализация видеозахвата с веб-камеры (0 - обычно встроенная камера)
        cap = cv2.VideoCapture(0)
        
        prediction_screen = None # Для окна с предсказанием
        calibration_screen = None # Для окна с калибровкой
        
        while cap.isOpened():
            # Чтение кадра с камеры
            success, frame = cap.read()
            h, w, _ = frame.shape
            if not success:
                print("Не удалось получить кадр с камеры.")
                break
            
            frame_counter += 1
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            face_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
            annotated_frame = frame # По умолчанию используем оригинальный кадр
            
            descriptors = []

            if face_landmarker_result.face_landmarks:
                # Получаем список ориентиров для первого (и единственного) обнаруженного лица
                descriptors, face_landmarks_3d = process_face(face_landmarker_result.face_landmarks[0], h, w, frame_counter)
                
                X = descriptors
                
                
                
                
                #cv2.putText(annotated_frame, f'Angles: {left_eye_angle:.2f}, {right_eye_angle:.2f}, {face_angle:.2f}', (20, 80),
                #        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
                #cv2.putText(annotated_frame, f'{norm[0]:.2f}, {norm[1]:.2f}, {norm[2]:.2f}', (20, 40),
                #        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
                        
                #if (np.mean([left_eye_angle, right_eye_angle]) > 45):
                #    off_center_count += 1
                #else:
                #    off_center_count = 0
                if app_state == 'CALIBRATING':
                    if calibration_screen is None:
                        root = tk.Tk()
                        screen_w, screen_h = root.winfo_screenwidth(), root.winfo_screenheight()
                        root.destroy()
                        calibration_screen = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
                        cv2.namedWindow('Calibration', cv2.WND_PROP_FULLSCREEN)
                        cv2.setWindowProperty('Calibration', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    x, y = calibration_points[current_calibration_point]
                    # Рисуем белую точку с красным центром
                    cv2.circle(calibration_screen, (x, y), 20, (255, 255, 255), -1)
                    cv2.circle(calibration_screen, (x, y), 5, (0, 0, 255), -1)
                    data_collector.add_data_point(descriptors)
                    cv2.imshow('Calibration', calibration_screen)
                    Y = (x, y)
                elif app_state == 'IDLE':
                    annotated_frame = draw_landmarks_on_image(frame, face_landmarker_result, indices=LEFT_IRIS+RIGHT_IRIS)
                    cv2.imshow('3D Face Landmarks', annotated_frame)
                
                               
                
            
            if off_center_count > ALERT_FRAMES:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.3, annotated_frame, 0.7, 0, annotated_frame)
                cv2.putText(annotated_frame, 'FOCUS!', (w // 3, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4)
                            
                
                

            # Отображаем результат
            cv2.imshow('3D Face Landmarks', annotated_frame)

            # Выход из цикла по нажатию клавиши 'q'
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        # Освобождаем ресурсы
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
