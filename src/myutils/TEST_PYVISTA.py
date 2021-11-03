import pyvista as pv
from pyvista import examples
import PVGeo
from h5_loader import load_data_h5, MakeBatchData
import h5py
import numpy as np
from normalize_points import rescale

# Load data
config = {}
config["index_num"] = 5
config["amplified_ratio"] = 1.05
config["parameter_feature"] = 15


def main(eval_config):

    print(config["index_num"])
    filename = "D:/Git/3d-AAE/data/ScaR_F2P_UNI_2048\\scaphoid_models_aligned.h5"
    with h5py.File(filename, "r") as f:
        # List all groups
        # print("Keys: %s" % f.keys())
        a_group_key = list(f.keys())

        # Get the data
        faces_list = list(f[a_group_key[0]])
        points_list = list(f[a_group_key[1]])

        normalized_points_list = []
        for ele in points_list:
            normalized_points, _ = rescale(ele)
            normalized_points_list.append(normalized_points)

    vertices_ = normalized_points_list[config["index_num"]]
    faces = faces_list[config["index_num"]]

    # manipulate faces to the required structure
    # e.g mesh faces:
    # faces = np.hstack([[4, 0, 1, 2, 3],  # square
    #                    [3, 0, 1, 4],     # triangle
    #                    [3, 1, 2, 4]])    # triangle
    new_faces = np.array([], dtype=int)
    for ele in faces:
        new_ele = np.array([], dtype=int)
        new_ele = np.append(new_ele, 3)
        for e in ele:
            new_ele = np.append(new_ele, e)
        new_faces = np.append(new_faces, new_ele)

    # manipulate faces to the required structure

    new_faces = np.hstack(new_faces)
    mesh = pv.PolyData(vertices_, new_faces)
    # plot original graph
    mesh.plot()

    # plot featured graph
    edges = mesh.extract_feature_edges(config["parameter_feature"])
    p = pv.Plotter()
    p.add_mesh(mesh, color=True)
    p.add_mesh(edges, color="red", line_width=5)
    # p.camera_position = [(9.5, 3.0, 5.5), (2.5, 1, 0), (0, 1, 0)]
    p.show()

    # exaggerate featured region
    def ndarray2string(ndarray):
        string_value = ''
        for ele in ndarray:
            string_value = string_value + str(ele)
        return string_value

    featured_points = np.array(edges.points)
    dictionary = {}

    for i, points in enumerate(featured_points):
        dictionary[ndarray2string(points)] = i

    index_list = []
    for i, vertice in enumerate(vertices_):
        if ndarray2string(vertice) in dictionary:
            index_list.append(i)

    changed_vertices_ = np.array(vertices_)
    for index in index_list:
        changed_vertices_[index] = config["amplified_ratio"] * changed_vertices_[index]

    mesh = pv.PolyData(changed_vertices_, new_faces)
    mesh.plot()


#
# mesh = examples.download_cow()
#
# edges = mesh.extract_feature_edges(20)
#
# p = pv.Plotter()
# p.add_mesh(mesh, color=True)
# p.add_mesh(edges, color="red", line_width=5)
# p.camera_position = [(9.5, 3.0, 5.5), (2.5, 1, 0), (0, 1, 0)]
# p.show()
#
#
# # mesh points
# vertices = np.array([[0, 0, 0],
#                      [1, 0, 0],
#                      [1, 1, 0],
#                      [0, 1, 0],
#                      [0.5, 0.5, -1]])
#
# # mesh faces
#
# faces = np.hstack([[4, 0, 1, 2, 3],  # square
#                    [3, 0, 1, 4],     # triangle
#                    [3, 1, 2, 4]])    # triangle
#
#
# surf = pv.PolyData(vertices, faces)
#
# # plot each face with a different color
# surf.plot()

if __name__ == '__main__':
    main(config)

