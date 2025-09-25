import numpy as np

class Sphere:
    def __init__(self,
                 eye_center: np.ndarray,
                 radius: np.ndarray):
        pass

class Fitter:
    def get_vision_ray(
            self,
            sphere_center: np.ndarray,
            iris_center: np.ndarray) -> np.ndarray:
        # returns 3d vector of eye direction
        return iris_center - sphere_center
    
    @staticmethod
    def fit_circle_in_3d_rough(
                        points, # Nx3
                        ):
        return points.mean(axis=0)
    
    @staticmethod
    def fit_plane_in_3d(
                        points):
        centroid = points.mean(axis=0)
        X = points - centroid
        _, _, vh = np.linalg.svd(X, full_matrices=False)
        normal = vh[2, :]
        u, v = vh[0, :], vh[1, :]
        if np.dot(np.cross(u, v), normal) < 0:
            v = -v
        return centroid, u / np.linalg.norm(u), v / np.linalg.norm(v), normal / np.linalg.norm(normal)

    @staticmethod
    def project_points_to_plane(points, centroid, u, v):
        P = np.asarray(points) - centroid
        x = P.dot(u)
        y = P.dot(v)
        return np.column_stack([x, y])
    
    @staticmethod
    def kasa_circle_fit(xy):
        """
        Algebraic (Kasa) circle fit in 2D.
        xy: Nx2 array
        Returns: center (cx,cy), radius r
        """
        x = xy[:,0]; y = xy[:,1]
        A = np.column_stack([x, y, np.ones_like(x)])
        b = -(x**2 + y**2)
        # solve A*[A,B,C]^T = b in least squares
        abc, *_ = np.linalg.lstsq(A, b, rcond=None)
        Acoef, Bcoef, Ccoef = abc
        cx = -Acoef/2
        cy = -Bcoef/2
        r_sq = cx*cx + cy*cy - Ccoef
        r = np.sqrt(np.abs(r_sq))
        return np.array([cx, cy]), r


    @staticmethod
    def project_points_to_plane(points, centroid, u, v):
        """Project 3D points to 2D coordinates in plane basis (u,v). Returns Nx2 array."""
        P = np.asarray(points) - centroid
        x = P.dot(u)
        y = P.dot(v)
        return np.column_stack([x, y])

    def fit_circle_in_3d(self,
                         pts, # Nx3
                         ):
        # returns 3d coords of a center and a radius
        centroid, u, v, normal = self.fit_plane_in_3d(pts)
        xy = self.project_points_to_plane(pts, centroid, u, v)
        # Kasa initial 2D fit
        center2d, r = self.kasa_circle_fit(xy)
        center3d = centroid + center2d[0]*u + center2d[1]*v
        return center3d, r
    
    def fit_sphere(self, points):
        """Sphere fitting using Linear Least Squares"""
    
        A = np.column_stack((2*points, np.ones(len(points))))
        b = (points**2).sum(axis=1)
        x, res, _, _ = np.linalg.lstsq(A, b)
        center = x[:3]
        radius = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2 + x[3])
    
        return (center, radius), res
    

    
