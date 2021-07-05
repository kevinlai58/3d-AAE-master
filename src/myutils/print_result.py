import numpy as np
import open3d as o3d
index = 50
epoch = '00020'
source_data = np.load('%s_X.npy' %epoch)
print_data_X = source_data[index,:]
point_cloud1 = o3d.geometry.PointCloud()
point_cloud1.points = o3d.utility.Vector3dVector(print_data_X)


source_data = np.load('%s_Xrec.npy'%epoch)
print_data_recX = source_data[index,:]
point_cloud2 = o3d.geometry.PointCloud()
point_cloud2.points = o3d.utility.Vector3dVector(print_data_recX)

source_data = np.load('%s_Xg_0.npy'%epoch)
print_data_g1 = source_data[index,:]
point_cloud3 = o3d.geometry.PointCloud()
point_cloud3.points = o3d.utility.Vector3dVector(print_data_g1)

source_data = np.load('%s_Xg_1.npy'%epoch)
print_data_g2 = source_data[index,:]
point_cloud4 = o3d.geometry.PointCloud()
point_cloud4.points = o3d.utility.Vector3dVector(print_data_g2)

source_data = np.load('%s_Xg_2.npy'%epoch)
print_data_g3 = source_data[index,:]
point_cloud5 = o3d.geometry.PointCloud()
point_cloud5.points = o3d.utility.Vector3dVector(print_data_g3)


vis = o3d.visualization.Visualizer()
vis.create_window(window_name='TopLeft', width=960, height=540, left=0, top=0)
vis.add_geometry(point_cloud1)

vis2 = o3d.visualization.Visualizer()
vis2.create_window(window_name='TopRight', width=960, height=540, left=960, top=0)
vis2.add_geometry(point_cloud2)

vis3 = o3d.visualization.Visualizer()
vis3.create_window(window_name='TopRight', width=500, height=540, left=0, top=540)
vis3.add_geometry(point_cloud3)

vis4 = o3d.visualization.Visualizer()
vis4.create_window(window_name='TopRight', width=500, height=540, left=500, top=540)
vis4.add_geometry(point_cloud4)

vis5 = o3d.visualization.Visualizer()
vis5.create_window(window_name='TopRight', width=500, height=540, left=1000, top=540)
vis5.add_geometry(point_cloud5)



while True:
    if not vis.poll_events():
        break
    if not vis2.poll_events():
        break
    if not vis3.poll_events():
        break
    if not vis4.poll_events():
        break
    if not vis5.poll_events():
        break



# vis.destroy_window()
# vis2.destroy_window()

