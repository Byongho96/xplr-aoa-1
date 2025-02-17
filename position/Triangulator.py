from typing import List, Tuple
import numpy as np

"""
Right-handed Cartesian coordinates
"""

class Triangulator:

    @classmethod
    def get_rotation_matrix_from_euler(cls, x: float, y: float, z: float) -> np.ndarray:
        """
        Get the local rotation matrix from the Euler angles (x, y, z) in radians.
        For Example: (0, 0, 0) -> (0, 0, 1) forward vector in the local coordinate system.
        For Example: (0, pi/2, 0) -> (1, 0, 0) right vector in the local coordinate system.
        """

        cx, cy, cz = np.cos(x), np.cos(y), np.cos(z)
        sx, sy, sz = np.sin(x), np.sin(y), np.sin(z)

        # X-axis rotation matrix
        Rx = np.array([
            [1,    0,    0],
            [0,   cx,  -sx],
            [0,   sx,   cx]
        ])

        # Y-axis rotation matrix
        Ry = np.array([
            [ cy, 0, sy],
            [  0, 1,  0],
            [-sy, 0, cy]
        ])

        # Z-axis rotation matrix
        Rz = np.array([
            [cz, -sz, 0],
            [sz,  cz, 0],
            [ 0,   0, 1]
        ])

        # Euler order: X -> Y -> Z
        return Rz @ Ry @ Rx

    @classmethod
    def get_direction_vector_from_angles(cls, elevation: float, azimuth: float) -> np.ndarray:
        """
        Get the unit direction vector in the local coordinate system using the measurement angles (elevation, azimuth).
        For Example: (0, 0) -> (0, 0, 1) forward vector in the local coordinate system.
        For Example: (0, pi/2) -> (1, 0, 0) right vector in the local coordinate system.
        """
        dx = np.cos(elevation) * np.sin(azimuth)
        dy = np.sin(elevation)
        dz = np.cos(elevation) * np.cos(azimuth)
        
        return np.array([dx, dy, dz])

    @classmethod
    def estimate_position(cls, boards: List[dict]) -> Tuple[float, float, float]:
        """
        Calculate the source position using the triangulation method.

        Args:
        - boards (List[dict]): A list of dictionaries containing the following
            - position (Tuple[float, float, float]): A tuple of anchor positions (x, y, z).
            - rotation (Tuple[float, float, float]): A tuple of anchor rotations (roll, pitch, yaw) in radians.
            - angle (Tuple[float, float]): A tuple of measurement angles (elevation, azimuth) in radians.
        """
        if len(boards) < 3:
            raise ValueError("At least 3 anchor positions are required.")
        
        # 3x3 항등 행렬 생성: 투영 행렬 계산에 사용됩니다.
        I = np.eye(3)

        # 누적 행렬 A (3x3)와 벡터 b (3x1)를 0으로 초기화합니다.
        A = np.zeros((3, 3))
        b = np.zeros(3)

        # Iterate for each anchor
        for board in boards:
            pos = board["position"]
            rot = board["rotation"]
            ang = board["angle"]

            P_world = np.array(pos)

            print(P_world)

            r_x, r_y, r_z = map(np.deg2rad, rot)
            print(r_x, r_y, r_z)
            R_local = cls.get_rotation_matrix_from_euler(r_x, r_y, r_z)

            elevation, azimuth = map(np.deg2rad, ang)
            D_local = cls.get_direction_vector_from_angles(elevation, azimuth)
            
            # Get unit direction vector in world coordinate system
            D_world = R_local @ D_local
            D_world /= np.linalg.norm(D_world)

            # np.outer(D_world, D_world) : Projection matrix on D_world
            # I - np.outer(D_world, D_world) : Eliminate the projection on D_world, and get the orthogonal matrix
            D_ortho = I - np.outer(D_world, D_world)
            
            # Cumulate orthogonal matrix to minimize the error
            A += D_ortho
            b += D_ortho @ P_world

        # psuedo-inverse of A
        # Ax = b -> x = A^(-1)b
        source_position = np.linalg.pinv(A) @ b

        # Return as a tuple
        return (float(source_position[0]), float(source_position[1]), float(source_position[2]))


# Example
if __name__ == "__main__":
    """
             Y
             ^
             │
             │   
             │   S 
             O---------1---> x
            /
           /
          2          3
         /  
        Z
    """
    boards = [
        {
            "position": (0, 0, 0),
            "rotation": (0, 45, 0),
            "angle": (45, 0)
        },
        {
            "position": (1, 0, 0),
            "rotation": (0, -45, 0),
            "angle": (45, 0)
        },
        {
            "position": (0, 0, 1),
            "rotation": (0, 135, 0),
            "angle": (45, 0)
        },
        {
            "position": (1, 0, 1),
            "rotation": (0, -135, 0),
            "angle": (45, 0)
        }
    ]

    estimated_position = Triangulator.estimate_position(boards) # (0.5, 0.707, 0.5)
    print("Estimated BLE source position:", estimated_position)
