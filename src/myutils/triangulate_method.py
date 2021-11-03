# sphinx_gallery_thumbnail_number = 2
import pyvista as pv
import numpy as np
import open3d as o3d



points = np.load('221000_Xrec.npy')[0, :]

# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(points)
# o3d.visualization.draw_geometries([pcd])

lines = ['Readme', 'How to write text files in Python']
with open('readme.txt', 'w') as f:
    for point in points:
        for cooridinate in point:
            f.write(str(cooridinate))
            f.write("\t")
        f.write('\n')

# # simply pass the numpy points to the PolyData constructor
# cloud = pv.PolyData(points)
# cloud.plot(point_size=15)
#
# surf = cloud.delaunay_3d(alpha=0.8)
# surf.plot(show_edges=True)