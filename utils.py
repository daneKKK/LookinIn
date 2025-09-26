import numpy as np
from screeninfo import get_monitors
from fit import Fitter

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
EYE_SIZE = 1.2 / 17 * NORMAL_DISTANCE

LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

LEFT_EYE_OUTLINE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_OUTLINE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

LEFT_EYE_OUTLINE = [163, 144, 145, 153, 154, 157, 158, 159, 160, 161]
RIGHT_EYE_OUTLINE = [381, 380, 374, 373, 390, 249, 388, 387, 386, 385, 384]

    
def process_landmarks(face_landmarks_3d, debug=False):
    '''
    Gets 3d landmarks. Return two eye vectors and nose vector, all in our abstract coordinates.
    '''
    #Calculate face center as center of a nose
    face_center = np.mean(face_landmarks_3d[NOSE_INDICES], axis=0)
    
    #Scale the faceand move along z to correctly represent it in 3D
    scaling_factor = NORMAL_DISTANCE / np.linalg.norm(face_landmarks_3d[127]-face_landmarks_3d[356])
    face_landmarks_3d = (face_landmarks_3d - face_center) * scaling_factor + face_center
    face_landmarks_3d[:, 2] += scaling_factor * NORMAL_DISTANCE
    
    # Fitting eyes to the sphere
    face_width = np.linalg.norm(face_landmarks_3d)
    fitter = Fitter()
    left_eye_sphere_c, left_eye_sphere_r = fitter.fit_sphere(face_landmarks_3d[LEFT_EYE_OUTLINE])[0]
    right_eye_sphere_c, right_eye_sphere_r = fitter.fit_sphere(face_landmarks_3d[RIGHT_EYE_OUTLINE])[0]
    
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
    eyeballs = [(right_eye_sphere_r, right_eye_sphere_c), (left_eye_sphere_r, left_eye_sphere_c)]
    
    if not debug:
        return np.mean([left_eye_sphere_c, right_eye_sphere_c], axis=0), np.mean([left_eye_dir, right_eye_dir], axis=0)
    else:
        return np.mean([left_eye_sphere_c, right_eye_sphere_c], axis=0), np.mean([left_eye_dir, right_eye_dir], axis=0), eyeballs, face_landmarks_3d
        
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
            y = int((1-landmark.y) * h)
            z = landmark.z * w
            if as_array:
                res.append((x, y, z))
            else:
                res[ind] = (x, y, z)
            # Рисуем маленькую зеленую точку на месте каждого ориентира
        
    return res if not as_array else np.array(res)
    
def find_affine_transform(l, v):
    """
    Finds the best-fit affine transformation matrix from 3D vectors {l_i} to {v_i}.

    Args:
        l: A list of 3D source vectors. Each vector should be a NumPy array.
        v: A list of 3D target vectors. Each vector should be a NumPy array.

    Returns:
        A 4x4 NumPy array representing the affine transformation matrix.
    """
    if len(l) != len(v) or len(l) < 3:
        raise ValueError("At least three corresponding vector pairs are required.")

    # Convert the lists of vectors to NumPy arrays
    l = np.array(l)
    v = np.array(v)

    # Add a homogeneous coordinate to the source vectors
    l_homogeneous = np.hstack((l, np.ones((l.shape[0], 1))))

    # Solve for the transformation matrix using least squares
    # The problem is to find T such that l_homogeneous * T = v
    # This can be solved as T = (l_homogeneous^T * l_homogeneous)^-1 * l_homogeneous^T * v
    T, _, _, _ = np.linalg.lstsq(l_homogeneous, v, rcond=None)
    

    # The result T is a 4x3 matrix. We need to convert it to a 4x4 affine matrix.
    # The last row of the affine matrix is [0, 0, 0, 1]
    added_row = np.zeros(T.shape[0])
    added_row[-1] = 1.0
    affine_matrix = np.vstack((T.T, added_row))

    return affine_matrix