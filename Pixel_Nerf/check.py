import glob, numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

poses = []
for f in sorted(glob.glob('./data/pollen_train/*/pose/*.txt')[:20]):
    M = np.loadtxt(f).reshape(4,4)
    poses.append(M[:3,3])
poses = np.array(poses)


fig = plt.figure(); ax = fig.add_subplot(111, projection='3d')
ax.scatter(poses[:,0], poses[:,1], poses[:,2], c='r')

# draw origin
ax.scatter([0],[0],[0], c='b')
ax.set_box_aspect((1,1,1))
plt.show()
print(f"Min Z: {poses[:,2].min():.2f}, Max Z: {poses[:,2].max():.2f}")
