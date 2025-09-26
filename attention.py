import numpy as np
from utils import process_landmarks
from os.path import join
from catboost import CatBoostRegressor


class AffineTransformationModel():
    def __init__(self, M=np.array([[1, 0], [0, 1]])):
        self.M = M
        
    def predict(self, l):
        return (self.M @ np.append(l[:2], 1))[:2]
        
    def fit(self, l, u):
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
        self.M = np.vstack((T.T, added_row))
    
    def save(path):
        np.save(join(path, 'M.npy'), M)
        
    def load(path):
        self.M = np.load(path)
        
class EnsembleModel():
    def __init__(self, M=np.array([[1, 0], [0, 1]]), cb=None):
        self.M = M
        if cb is None:
            self.cb = CatBoostRegressor()
        else:
            self.cb = cb
    
    def predict(self, l):
        Ml = (self.M @ np.append(l[:2], 1))[:2]
        CMl = self.cb.predict(np.array([Ml]))[0]
        return np.mean([CMl, Ml], axis=0)
        
    def load(self, M_path, cb_path):
        self.M = np.load(M_path)
        self.cb.load_model(cb_path)
        

class AttentionTracker():
    GRACE_OUT_OF_FRAME_PERIOD = 50
    GRACE_STARING_FRAME_PERIOD = 600
    GRACE_BLINKING_PERIOD = 50
    SMOOTHING_WINDOW = 10
    
    RIGHT_EYE_OUTLINES = [386, 374]
    LEFT_EYE_OUTLINES = [145, 159]
    
    BLINK_THRESHOLD = 5
    STARING_THRESHOLD = 100

    def __init__(self, model, screen_w, screen_h):
        self.uv_array = np.zeros((self.SMOOTHING_WINDOW, 2))
        self.uv_model = model
        self.w = screen_w
        self.h = screen_h
        
        self.blinking_counter = 0
        self.out_of_frame_counter = 0
        self.staring_frame_counter = 0
        self.uv_staring_array = np.ones((self.GRACE_STARING_FRAME_PERIOD, 2)) * (-100000)
        
    
    def eval(self, landmarks):
        uv = self.uv_model.predict(process_landmarks(landmarks)[1])
        self.uv_array[:-1] = self.uv_array[1:]
        self.uv_array[-1] = uv
        self.uv = np.mean(self.uv_array, axis=0)
        return self.uv
        
    def _out_of_frame_checker(self):
        #print((0 < uv[0] < screen_w), 0 < uv[1] < screen_h)
        is_out_of_frame = int(not ((0 < self.uv[0] < self.w) and (0 < self.uv[1] < self.h)))
        self.out_of_frame_counter = is_out_of_frame * self.out_of_frame_counter + is_out_of_frame
        return self.out_of_frame_counter > self.GRACE_OUT_OF_FRAME_PERIOD
        
    def _blink_checker(self, landmarks):
        is_blinking = ((np.linalg.norm(landmarks[self.RIGHT_EYE_OUTLINES[0]] - 
                                       landmarks[self.RIGHT_EYE_OUTLINES[1]]) < self.BLINK_THRESHOLD) and
                       (np.linalg.norm(landmarks[self.LEFT_EYE_OUTLINES[0]] - 
                                       landmarks[self.LEFT_EYE_OUTLINES[1]]) < self.BLINK_THRESHOLD))
        is_blinking = int(is_blinking)
        self.blinking_counter = is_blinking * self.blinking_counter + is_blinking
        return self.blinking_counter > self.GRACE_BLINKING_PERIOD
        
    def _staring_checker(self, landmarks):
        self.uv_staring_array[:-1] = self.uv_staring_array[1:]
        self.uv_staring_array[-1] = self.uv
        print(np.sum(np.std(self.uv_staring_array[self.uv_staring_array[:, 0] > -50000], axis=0)))
        return np.sum(np.std(self.uv_staring_array[self.uv_staring_array[:, 0] > -50000], axis=0)) < self.STARING_THRESHOLD
        
        
    def attention_check(self, landmarks):
        return (self._out_of_frame_checker() or self._blink_checker(landmarks) or self._staring_checker(landmarks))
        
        
        