import numpy as np
import copy
import math
import cv2

EPS = np.finfo(float).eps * 4.0

# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    "sxyz": (0, 0, 0, 0),
    "sxyx": (0, 0, 1, 0),
    "sxzy": (0, 1, 0, 0),
    "sxzx": (0, 1, 1, 0),
    "syzx": (1, 0, 0, 0),
    "syzy": (1, 0, 1, 0),
    "syxz": (1, 1, 0, 0),
    "syxy": (1, 1, 1, 0),
    "szxy": (2, 0, 0, 0),
    "szxz": (2, 0, 1, 0),
    "szyx": (2, 1, 0, 0),
    "szyz": (2, 1, 1, 0),
    "rzyx": (0, 0, 0, 1),
    "rxyx": (0, 0, 1, 1),
    "ryzx": (0, 1, 0, 1),
    "rxzx": (0, 1, 1, 1),
    "rxzy": (1, 0, 0, 1),
    "ryzy": (1, 0, 1, 1),
    "rzxy": (1, 1, 0, 1),
    "ryxy": (1, 1, 1, 1),
    "ryxz": (2, 0, 0, 1),
    "rzxz": (2, 0, 1, 1),
    "rxyz": (2, 1, 0, 1),
    "rzyz": (2, 1, 1, 1),
}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())


def _compress_obs(obs):
    for key in obs.keys():
        if 'image' in key:
            if obs[key] is not None:
                if len(obs[key].shape) == 3:
                    okay, im_string = cv2.imencode('.jpg', obs[key])
                    assert okay, "image encoding failed!"
                    obs[key] = im_string
        if 'depth_norm' in key:
            assert len(
                obs[key].shape) == 2 and obs[key].dtype == np.uint8, "assumes uint8 greyscale depth image!"
            depth_im = np.tile(obs[key][:, :, None], (1, 1, 3))
            okay, depth_string = cv2.imencode('.jpg', depth_im)
            assert okay, "depth encoding failed!"
            obs[key] = depth_string
    return obs


def _decompress_obs(obs):
    keys = ["image"]
    for key in keys:
        if 'image' in key:
            if obs[key] is not None:
                try:
                    decomp = cv2.imdecode(obs[key], cv2.IMREAD_COLOR)
                    obs[key] = decomp
                except:
                    pass
        if 'depth_norm' in key:
            obs[key] = cv2.imdecode(
                obs[key], cv2.IMREAD_GRAYSCALE).astype(np.uint8)
    return obs


class Trajectory:
    def __init__(self, config_str=None):
        self._data = []
        self._raw_state = []
        self._config_str = None
        self.set_config_str(config_str)

    def append(self, obs, reward=None, done=None, info=None, action=None, raw_state=None):
        """
        Logs observation and rewards taken by environment as well as action taken
        """
        obs, reward, done, info, action, raw_state = [copy.deepcopy(
            x) for x in [obs, reward, done, info, action, raw_state]]

        obs = _compress_obs(obs)
        self._data.append((obs, reward, done, info, action))
        self._raw_state.append(raw_state)

    @ property
    def T(self):
        """
        Returns number of states
        """
        return len(self._data)

    def __getitem__(self, t):
        return self.get(t)

    def get(self, t, decompress=True):
        assert 0 <= t < self.T or - \
            self.T < t <= 0, "index should be in (-T, T)"

        obs_t, reward_t, done_t, info_t, action_t = self._data[t]
        if decompress:
            obs_t = _decompress_obs(obs_t)
        ret_dict = dict(obs=obs_t, reward=reward_t,
                        done=done_t, info=info_t, action=action_t)

        for k in list(ret_dict.keys()):
            if ret_dict[k] is None:
                ret_dict.pop(k)
        return ret_dict

    def change_obs(self, t, obs):
        obs_t, reward_t, done_t, info_t, action_t = self._data[t]
        self._data[t] = obs, reward_t, done_t, info_t, action_t

    def __len__(self):
        return self.T

    def __iter__(self):
        for d in range(self.T):
            yield self.get(d)

    def get_raw_state(self, t):
        assert 0 <= t < self.T or - \
            self.T < t <= 0, "index should be in (-T, T)"
        return copy.deepcopy(self._raw_state[t])

    def set_config_str(self, config_str):
        self._config_str = config_str

    @ property
    def config_str(self):
        return self._config_str


def vec(values):
    """
    Converts value tuple into a numpy vector.

    Args:
        values (n-array): a tuple of numbers

    Returns:
        np.array: vector of given values
    """
    return np.array(values, dtype=np.float32)


def quat2axisangle(quat):
    """
    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero-degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def quat2mat(quaternion):
    """
    Converts given quaternion to matrix.

    Args:
        quaternion (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: 3x3 rotation matrix
    """
    # awkward semantics for use with numba
    inds = np.array([3, 0, 1, 2])
    q = np.asarray(quaternion).copy().astype(np.float32)[inds]

    n = np.dot(q, q)
    if n < EPS:
        return np.identity(3)
    q *= math.sqrt(2.0 / n)
    q2 = np.outer(q, q)
    return np.array(
        [
            [1.0 - q2[2, 2] - q2[3, 3], q2[1, 2] - q2[3, 0], q2[1, 3] + q2[2, 0]],
            [q2[1, 2] + q2[3, 0], 1.0 - q2[1, 1] - q2[3, 3], q2[2, 3] - q2[1, 0]],
            [q2[1, 3] - q2[2, 0], q2[2, 3] + q2[1, 0], 1.0 - q2[1, 1] - q2[2, 2]],
        ]
    )


def mat2euler(rmat, axes="sxyz"):
    """
    Converts given rotation matrix to euler angles in radian.

    Args:
        rmat (np.array): 3x3 rotation matrix
        axes (str): One of 24 axis sequences as string or encoded tuple (see top of this module)

    Returns:
        np.array: (r,p,y) converted euler angles in radian vec3 float
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]

    M = np.asarray(rmat, dtype=np.float32)[:3, :3]
    if repetition:
        sy = math.sqrt(M[i, j] * M[i, j] + M[i, k] * M[i, k])
        if sy > EPS:
            ax = math.atan2(M[i, j], M[i, k])
            ay = math.atan2(sy, M[i, i])
            az = math.atan2(M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(sy, M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i] * M[i, i] + M[j, i] * M[j, i])
        if cy > EPS:
            ax = math.atan2(M[k, j], M[k, k])
            ay = math.atan2(-M[k, i], cy)
            az = math.atan2(M[j, i], M[i, i])
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(-M[k, i], cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return vec((ax, ay, az))


def quat2euler(quat):
    """
    Converts given quaternion to euler angles.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (r,p,y) converted euler angles in radian vec3 float
    """
    return mat2euler(rmat=quat2mat(quaternion=quat))


def euler2mat(euler):
    """
    Converts euler angles into rotation matrix form

    Args:
        euler (np.array): (r,p,y) angles

    Returns:
        np.array: 3x3 rotation matrix

    Raises:
        AssertionError: [Invalid input shape]
    """

    euler = np.asarray(euler, dtype=np.float64)
    assert euler.shape[-1] == 3, "Invalid shaped euler {}".format(euler)

    ai, aj, ak = -euler[..., 2], -euler[..., 1], -euler[..., 0]
    si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak)
    ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    mat = np.empty(euler.shape[:-1] + (3, 3), dtype=np.float64)
    mat[..., 2, 2] = cj * ck
    mat[..., 2, 1] = sj * sc - cs
    mat[..., 2, 0] = sj * cc + ss
    mat[..., 1, 2] = cj * sk
    mat[..., 1, 1] = sj * ss + cc
    mat[..., 1, 0] = sj * cs - sc
    mat[..., 0, 2] = -sj
    mat[..., 0, 1] = cj * si
    mat[..., 0, 0] = cj * ci
    return mat


def mat2quat(rmat):
    """
    Converts given rotation matrix to quaternion.

    Args:
        rmat (np.array): 3x3 rotation matrix

    Returns:
        np.array: (x,y,z,w) float quaternion angles
    """
    M = np.asarray(rmat).astype(np.float32)[:3, :3]

    m00 = M[0, 0]
    m01 = M[0, 1]
    m02 = M[0, 2]
    m10 = M[1, 0]
    m11 = M[1, 1]
    m12 = M[1, 2]
    m20 = M[2, 0]
    m21 = M[2, 1]
    m22 = M[2, 2]
    # symmetric matrix K
    K = np.array(
        [
            [m00 - m11 - m22, np.float32(0.0), np.float32(0.0), np.float32(0.0)],
            [m01 + m10, m11 - m00 - m22, np.float32(0.0), np.float32(0.0)],
            [m02 + m20, m12 + m21, m22 - m00 - m11, np.float32(0.0)],
            [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
        ]
    )
    K /= 3.0
    # quaternion is Eigen vector of K that corresponds to largest eigenvalue
    w, V = np.linalg.eigh(K)
    inds = np.array([3, 0, 1, 2])
    q1 = V[inds, np.argmax(w)]
    if q1[0] < 0.0:
        np.negative(q1, q1)
    inds = np.array([1, 2, 3, 0])
    return q1[inds]


def euler2quat(euler):
    """
    Converts given euler angles to quaternion.

    Args:
        euler (np.array): (r,p,y) angles

    Returns:
        np.array: (x,y,z,w) float quaternion angles
    """
    return mat2quat(euler2mat(euler=euler))


def create_T(R, xyz, scaling):
    """
    Creates the transformation matrix by using rotation matrix, translation column
    and scaling row.

    Args:
        R: (np.array): 3x3 rotation matrix
        xyz: (np.array): 3x1 translation row
        scaling: (np.array) 1x4 scaling row

    Returns:
        np.array: 4x4 transformation matrix
    """
    T = np.vstack((np.hstack((R, xyz)), scaling))

    return T


def convert_Berkeley(observation):
    # getting current eef position
    position = np.vstack(observation['robot_state'].numpy()[6:9])  # for matrix creation

    # getting current eef quaternion
    quat = observation['robot_state'].numpy()[9:13]

    # 180° of rotation around the z-axis
    R_tcp_berkeley_tcp_unisa = np.array([
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]
    ])

    # -90° of rotation around the z-axis
    R_base_berkeley_base_unisa = np.array([
        [0, 1, 0],
        [-1, 0, 0],
        [0, 0, 1]
    ])

    R_base_berkeley_tcp_unisa = quat2mat(quat)
    R_base_berkeley_tcp_unisa = np.matmul(R_base_berkeley_tcp_unisa, R_tcp_berkeley_tcp_unisa)
    R_tcp_unisa_base_berkeley = np.linalg.inv(R_base_berkeley_tcp_unisa)
    R_tcp_unisa_base_unisa = np.matmul(R_tcp_unisa_base_berkeley, R_base_berkeley_base_unisa)
    R_base_unisa_tcp_unisa = np.linalg.inv(R_tcp_unisa_base_unisa)

    origin_position_column = np.vstack((0, 0, 0))  # 3x1, useful when there is no translation
    no_scaling_row = np.hstack((0, 0, 0, 1))  # 1x4, useful when there is no scaling

    T_base_berkeley_tcp_berkeley = create_T(R=R_base_berkeley_tcp_unisa, xyz=position,
                                            scaling=no_scaling_row)  # 4x4
    T_tcp_berkeley_tcp_unisa = create_T(R=R_tcp_berkeley_tcp_unisa, xyz=origin_position_column,
                                        scaling=no_scaling_row)  # 4x4
    T_base_berkeley_tcp_unisa = np.matmul(T_base_berkeley_tcp_berkeley, T_tcp_berkeley_tcp_unisa)  # 4x4

    position_base_berkeley_tcp_unisa = np.vstack((T_base_berkeley_tcp_unisa[0, 3],
                                                  T_base_berkeley_tcp_unisa[1, 3],
                                                  T_base_berkeley_tcp_unisa[2, 3]))  # 3x1
    T_tcp_unisa_base_unisa = create_T(R=np.linalg.inv(R_base_berkeley_tcp_unisa),
                                      xyz=np.matmul(np.linalg.inv(R_base_berkeley_tcp_unisa),
                                                    position_base_berkeley_tcp_unisa),
                                      scaling=no_scaling_row)  # 4x4

    T_base_berkeley_base_unisa = create_T(R=R_base_berkeley_base_unisa, xyz=origin_position_column,
                                          scaling=no_scaling_row)  # 4x4
    T_tcp_unisa_base_unisa = np.matmul(T_tcp_unisa_base_unisa, T_base_berkeley_base_unisa)  # 4x4
    position_tcp_unisa_base_unisa = np.vstack(
        (T_tcp_unisa_base_unisa[0, 3], T_tcp_unisa_base_unisa[1, 3], T_tcp_unisa_base_unisa[2, 3]))  # 3x1

    T_base_unisa_tcp_unisa = create_T(R=R_base_unisa_tcp_unisa,
                                      xyz=np.matmul(R_base_unisa_tcp_unisa, position_tcp_unisa_base_unisa),
                                      scaling=no_scaling_row)

    # getting p from last transformation matrix
    position_base_unisa_tcp_unisa = np.concatenate((T_base_unisa_tcp_unisa[0, 3],
                                                    T_base_unisa_tcp_unisa[1, 3],
                                                    T_base_unisa_tcp_unisa[2, 3]),
                                                   axis=0)  # 3x1

    # saving quaternion of current waypoint
    quat_base_unisa_tcp_unisa = mat2quat(R_base_unisa_tcp_unisa)

    return position_base_unisa_tcp_unisa, quat_base_unisa_tcp_unisa
