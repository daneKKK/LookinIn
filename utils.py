import cv2
from screeninfo import get_monitors


def get_screen_size():
    """works only with one monitory correctly"""
    h, w = 0, 0
    for monitor in get_monitors():
        w, h = monitor.width, monitor.height
    
    return w, h

def get_landmarks(
        detection_result,
        w : int,
        h : int,
        indices=None,
        ):
    face_landmarks_list = detection_result.face_landmarks
    res = {}
    # Перебираем обнаруженные лица
    for face_landmarks in face_landmarks_list:
        if indices is None:
            indices = range(len(face_landmarks))
        # Перебираем ориентиры на лице
        for ind in indices:
            landmark = face_landmarks[ind]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            z = landmark.z
            res[ind] = (x, y, z)
            # Рисуем маленькую зеленую точку на месте каждого ориентира
    return res
            

