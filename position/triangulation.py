from typing import List, Tuple
import numpy as np

"""
Right-handed Cartesian coordinates
"""

def get_rotation_matrix_from_euler(x: float, y: float, z: float) -> np.ndarray:
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

def get_direction_vector_from_angles(elevation: float, azimuth: float) -> np.ndarray:
    """
    Get the unit direction vector in the local coordinate system using the measurement angles (elevation, azimuth).
    For Example: (0, 0) -> (0, 0, 1) forward vector in the local coordinate system.
    For Example: (0, pi/2) -> (1, 0, 0) right vector in the local coordinate system.
    """
    dx = np.cos(elevation) * np.sin(azimuth)
    dy = np.sin(elevation)
    dz = np.cos(elevation) * np.cos(azimuth)
    
    return np.array([dx, dy, dz])

def calculate_source_position(positions: List[Tuple[float, float, float]],
                              rotations: List[Tuple[float, float, float]],
                              angles: List[Tuple[float, float]]) -> Tuple[float, float, float]:
    """
    Calculate the source position using the triangulation method.

    Args:
    - positions: A list of anchor positions (x, y, z).
    - rotations: A list of anchor rotations (roll, pitch, yaw) in radians.
    - angles: A list of measurement angles (elevation, azimuth) in radians.
    """
    if len(positions) < 3:
        raise ValueError("At least 3 anchor positions are required.")

    if len(positions) != len(rotations) or len(positions) != len(angles):
        raise ValueError("The number of positions, rotations, and angles must be the same.")

    # 3x3 항등 행렬 생성: 투영 행렬 계산에 사용됩니다.
    I = np.eye(3)

    # 누적 행렬 A (3x3)와 벡터 b (3x1)를 0으로 초기화합니다.
    A = np.zeros((3, 3))
    b = np.zeros(3)

    # Iterate for each anchor
    for pos, rot, ang in zip(positions, rotations, angles):
        P_world = np.array(pos)

        r_x, r_y, r_z = rot
        R_local = get_rotation_matrix_from_euler(r_x, r_y, r_z)

        elevation, azimuth = ang
        D_local = get_direction_vector_from_angles(elevation, azimuth)
        
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
    positions = [
        (0, 0, 0),
        (1, 0, 0),
        (0, 0, 1),
        (1, 0, 1)
    ]
    rotations = [
        (0, np.deg2rad(45), 0),
        (0, np.deg2rad(-45), 0),
        (0, np.deg2rad(135), 0),
        (0, np.deg2rad(-135), 0)
    ]
    angles = [
        (np.deg2rad(45), np.deg2rad(0)),
        (np.deg2rad(45), np.deg2rad(0)),
        (np.deg2rad(45), np.deg2rad(0)),
        (np.deg2rad(45), np.deg2rad(0))
    ]

    estimated_position = calculate_source_position(positions, rotations, angles) # (0.5, 0.707, 0.5)
    print("Estimated BLE source position:", estimated_position)
