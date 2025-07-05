# Copyright (C) 2019 Chao Wen, Yinda Zhang, Zhuwen Li, Yanwei Fu
# All rights reserved.
# This code is licensed under BSD 3-Clause License.

import sys
import numpy as np
import os

if __name__ == '__main__':
    xyz_path = sys.argv[1]
    obj_path = xyz_path.replace('.xyz', '.obj')
    face_path = '../data/face2.obj'  # Adjust if needed

    xyzf = np.loadtxt(xyz_path)
    v = np.full((xyzf.shape[0], 1), 'v')
    face = np.loadtxt(face_path, dtype='|S32')
    out = np.vstack((np.hstack((v, xyzf)), face))
    np.savetxt(obj_path, out, fmt='%s', delimiter=' ')
