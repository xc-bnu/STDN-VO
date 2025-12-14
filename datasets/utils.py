import numpy as np
import functools


def rotation_to_euler(M, cy_thresh=None, seq='zyx'):
    M = np.asarray(M)
    if cy_thresh is None:
        try:
            cy_thresh = np.finfo(M.dtype).eps * 4
        except ValueError:
            cy_thresh = np.finfo(float).eps * 4.0  # _FLOAT_EPS_4
    r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat
    # cy: sqrt((cos(y)*cos(z))**2 + (cos(x)*cos(y))**2)
    cy = np.sqrt(r33 * r33 + r23 * r23)
    if seq == 'zyx':
        if cy > cy_thresh:  # cos(y) not close to zero, standard form
            z = np.arctan2(-r12, r11)  # atan2(cos(y)*sin(z), cos(y)*cos(z))
            y = np.arctan2(r13, cy)  # atan2(sin(y), cy)
            x = np.arctan2(-r23, r33)  # atan2(cos(y)*sin(x), cos(x)*cos(y))
        else:  # cos(y) (close to) zero, so x -> 0.0 (see above)
            # so r21 -> sin(z), r22 -> cos(z) and
            z = np.arctan2(r21, r22)
            y = np.arctan2(r13, cy)  # atan2(sin(y), cy)
            x = 0.0
    elif seq == 'xyz':
        if cy > cy_thresh:
            y = np.arctan2(-r31, cy)
            x = np.arctan2(r32, r33)
            z = np.arctan2(r21, r11)
        else:
            z = 0.0
            if r31 < 0:
                y = np.pi / 2
                x = np.arctan2(r12, r13)
            else:
                y = -np.pi / 2
    else:
        raise Exception('Sequence not recognized')
    return [z, y, x]


def euler_to_rotation(z=0, y=0, x=0, isRadian=True, seq='zyx'):
    if seq != 'xyz' and seq != 'zyx':
        raise Exception('Sequence not recognized')

    if not isRadian:
        z = ((np.pi) / 180.) * z
        y = ((np.pi) / 180.) * y
        x = ((np.pi) / 180.) * x
    if z < -np.pi:
        while z < -np.pi:
            z += 2 * np.pi
    if z > np.pi:
        while z > np.pi:
            z -= 2 * np.pi
    if y < -np.pi:
        while y < -np.pi:
            y += 2 * np.pi
    if y > np.pi:
        while y > np.pi:
            y -= 2 * np.pi
    if x < -np.pi:
        while x < -np.pi:
            x += 2 * np.pi
    if x > np.pi:
        while x > np.pi:
            x -= 2 * np.pi
    assert z >= (-np.pi) and z < np.pi, 'Inappropriate z: %f' % z
    assert y >= (-np.pi) and y < np.pi, 'Inappropriate y: %f' % y
    assert x >= (-np.pi) and x < np.pi, 'Inappropriate x: %f' % x

    Ms = []

    if seq == 'zyx':

        if z:
            cosz = np.cos(z)
            sinz = np.sin(z)
            Ms.append(np.array(
                [[cosz, -sinz, 0],
                 [sinz, cosz, 0],
                 [0, 0, 1]]))
        if y:
            cosy = np.cos(y)
            siny = np.sin(y)
            Ms.append(np.array(
                [[cosy, 0, siny],
                 [0, 1, 0],
                 [-siny, 0, cosy]]))
        if x:
            cosx = np.cos(x)
            sinx = np.sin(x)
            Ms.append(np.array(
                [[1, 0, 0],
                 [0, cosx, -sinx],
                 [0, sinx, cosx]]))
        if Ms:
            return functools.reduce(np.dot, Ms[::-1])
        return np.eye(3)

    elif seq == 'xyz':

        if x:
            cosx = np.cos(x)
            sinx = np.sin(x)
            Ms.append(np.array(
                [[1, 0, 0],
                 [0, cosx, -sinx],
                 [0, sinx, cosx]]))
        if y:
            cosy = np.cos(y)
            siny = np.sin(y)
            Ms.append(np.array(
                [[cosy, 0, siny],
                 [0, 1, 0],
                 [-siny, 0, cosy]]))
        if z:
            cosz = np.cos(z)
            sinz = np.sin(z)
            Ms.append(np.array(
                [[cosz, -sinz, 0],
                 [sinz, cosz, 0],
                 [0, 0, 1]]))

        if Ms:
            return functools.reduce(np.dot, Ms[::-1])
        return np.eye(3)
