consult https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md

on windows:
1. download the CUB library .zip from https://github.com/NVIDIA/cub/releases and unpack it in any directory. Then set the environment variable `CUB_HOME` to the path of the unpacked directory (where CMakeLists.txt is located). Make sure to restart your terminal after setting the environment variable.
2. run 
```
pip install --upgrade setuptools wheel
```
3. build and install the package using the following command:
```
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```