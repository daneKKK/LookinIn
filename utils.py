import cv2
import numpy as np
from screeninfo import get_monitors


def get_screen_size():
    """works only with one monitory correctly"""
    h, w = 0, 0
    for monitor in get_monitors():
        w, h = monitor.width, monitor.height
    
    return w, h
    
NOSE_INDICES = [4, 45, 275, 220, 440, 1, 5, 51, 281, 44, 274, 241, 
                461, 125, 354, 218, 438, 195, 167, 393, 165, 391,
                3, 248]

NORMAL_DISTANCE = 180

LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

LEFT_EYE_OUTLINE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_OUTLINE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

    
def process_landmarks(detection_result, h, w):
    '''
    Gets 3d landmarks. Return two eye vectors and nose vector, all in our abstract coordinates.
    '''
    face_landmarks_3d = np.array([(l.x, l.y, l.z) for l in detection_result.face_landmarks[0]])
    face_landmarks_3d[:, 0] *= w
    face_landmarks_3d[:, 1] = h*(1 - face_landmarks_3d[:, 1])
    face_landmarks_3d[:, 2] *= w
    #Calculate face center as center of a nose
    face_center = np.mean(face_landmarks_3d[NOSE_INDICES], axis=0)
    
    #Scale the faceand move along z to correctly represent it in 3D
    scaling_factor = NORMAL_DISTANCE / np.linalg.norm(face_landmarks_3d[127]-face_landmarks_3d[356])
    face_landmarks_3d = (face_landmarks_3d - face_center) * scaling_factor + face_center
    face_landmarks_3d[:, 2] += scaling_factor * NORMAL_DISTANCE
    
    # Fitting eyes to the sphere
    fitter = Fitter()
    left_eye_sphere_c, left_eye_sphere_r = fitter.fit_circle_in_3d(face_landmarks_3d[LEFT_EYE_OUTLINE])
    right_eye_sphere_c, right_eye_sphere_r = fitter.fit_circle_in_3d(face_landmarks_3d[RIGHT_EYE_OUTLINE])
    
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
    
    
    return np.mean([left_eye_sphere_c, right_eye_sphere_c], axis=0), np.mean([left_eye_dir, right_eye_dir], axis=0)

def get_landmarks(
        detection_result,
        w : int,
        h : int,
        indices=None,
        as_array: bool = True
        ):
    face_landmarks_list = detection_result.face_landmarks
    res = {}
    if as_array:
        res = []
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
            if as_array:
                res.append((x, y, z))
            else:
                res[ind] = (x, y, z)
            # Рисуем маленькую зеленую точку на месте каждого ориентира
        
    return res if not as_array else np.array(res)