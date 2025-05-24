import trimesh
mesh = trimesh.load('./data/meshes/17767_Common_knapweed_Centaurea_nigra_pollen_grain.stl')
print(mesh.centroid)  # Should be ~ [0, 0, 0]