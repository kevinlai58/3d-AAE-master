import pyvista as pv
import h5py
import numpy as np
from normalize_points import rescale

# parameter
config = {"index_num": 5, "amplified_ratio": 1.05, "parameter_feature": 15, "plot": True}

# load data
filename = "D:/Git/3d-AAE/data/ScaR_F2P_UNI_2048\\scaphoid_models_aligned_amplified_ratio1.025.h5"
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

def show_all_image(eval_config):

    print("index number = " + str(config["index_num"]))
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
    if config["plot"]:
        mesh.plot()



def manipulate_a_bone(eval_config):
    print("index number = " + str(config["index_num"]))
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
    if config["plot"]:
        mesh.plot()

    # plot featured graph
    edges = mesh.extract_feature_edges(config["parameter_feature"])
    p = pv.Plotter()
    p.add_mesh(mesh, color=True)
    p.add_mesh(edges, color="red", line_width=5)
    # p.camera_position = [(9.5, 3.0, 5.5), (2.5, 1, 0), (0, 1, 0)]
    if config["plot"]:
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

    if config["plot"]:
        mesh = pv.PolyData(changed_vertices_, new_faces)
        mesh.plot()

    return changed_vertices_


if __name__ == '__main__':
    length = len(normalized_points_list)
    for i in range(length):
        config["index_num"] = i
        show_all_image(config)

    # length = len(normalized_points_list)
    # list_changed_vertices_ = []
    # for i in range(length):
    #     config["index_num"] = i
    #     changed_vertices_ = manipulate_a_bone(config)
    #     list_changed_vertices_.append(changed_vertices_)
    #
    # hf = h5py.File('D:/Git/3d-AAE/data/ScaR_F2P_UNI_2048/scaphoid_models_aligned_amplified_ratio'+str(config["amplified_ratio"])+'.h5', 'w')
    # hf.create_dataset('points', data=list_changed_vertices_)
    # hf.create_dataset('faces', data=faces_list)
    #
    # hf.close()
