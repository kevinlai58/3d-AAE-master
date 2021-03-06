import pymeshlab
import numpy as np
import h5py
from normalize_points import rescale


def loadh5file(filename):
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
        return faces_list, normalized_points_list


def example_import_mesh_from_arrays():
    filename = "D:/Git/3d-AAE/data/ScaR_F2P_UNI_2048\\scaphoid_models_aligned.h5"
    faces_list, points_list = loadh5file(filename)

    # create a new Mesh with the two arrays
    m = pymeshlab.Mesh(points_list[0], faces_list[0])
    # create a new MeshSet
    ms = pymeshlab.MeshSet()

    # add the mesh to the MeshSet
    ms.add_mesh(m, "original")

    # save the current mesh
    ms.save_current_mesh("original.ply")

def export():

    points_list = np.load('221000_Xrec.npy')[0,:]
    filename = "D:/Git/3d-AAE/data/ScaR_F2P_UNI_2048\\scaphoid_models_aligned.h5"
    faces_list, _ = loadh5file(filename)

    # create a new Mesh with the two arrays
    m = pymeshlab.Mesh(points_list, faces_list[0])
    # create a new MeshSet
    ms = pymeshlab.MeshSet()

    # add the mesh to the MeshSet
    ms.add_mesh(m, "rec")

    # save the current mesh
    ms.save_current_mesh("rectruction.ply")




def test():
    # lines needed to run this specific example
    print('\n')
    output_path = ""

    # create a numpy 8x3 array of vertices
    # columns are the coordinates (x, y, z)
    # every row represents a vertex
    verts = numpy.array([
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [-0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5]])

    # create a numpy 12x3 array of faces
    # every row represents a face (triangle in this case)
    # for every triangle, the index of the vertex
    # in the vertex array
    faces = numpy.array([
        [2, 1, 0],
        [1, 2, 3],
        [4, 2, 0],
        [2, 4, 6],
        [1, 4, 0],
        [4, 1, 5],
        [6, 5, 7],
        [5, 6, 4],
        [3, 6, 7],
        [6, 3, 2],
        [5, 3, 7],
        [3, 5, 1]])

    # create a new Mesh with the two arrays
    m = pymeshlab.Mesh(verts, faces)

    assert m.vertex_number() == 8
    assert m.face_number() == 12

    # create a new MeshSet
    ms = pymeshlab.MeshSet()

    # add the mesh to the MeshSet
    ms.add_mesh(m, "cube_mesh")

    # save the current mesh
    ms.save_current_mesh(output_path + "saved_cube_from_array.ply")

    # create a 1D numpy array of 8 elements to store per vertex quality
    vert_quality = numpy.array([
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8])

    # create a 1D numpy array of 12 elements to store per face quality
    face_quality = numpy.array([
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12])

    # create a new mesh with the selected arrays
    m1 = pymeshlab.Mesh(
        vertex_matrix=verts,
        face_matrix=faces,
        v_quality_array=vert_quality,
        f_quality_array=face_quality)

    # add the mesh to the MeshSet
    ms.add_mesh(m1, "cube_quality_mesh")

    # colorize the cube according to the per face and per vertex quality
    ms.colorize_by_vertex_quality()
    ms.colorize_by_face_quality()

    # save the mesh
    ms.save_current_mesh(output_path + "colored_cube_from_array.ply")


if __name__ == '__main__':
    # example_import_mesh_from_arrays()
    export()