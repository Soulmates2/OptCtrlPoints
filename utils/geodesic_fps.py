import numpy as np
import trimesh
import potpourri3d as pp3d
# https://pypi.org/project/potpourri3d/#mesh-distance

def geodesic_farthest_point_sampling_heat(vertices, faces, num_points):
    # = Stateful solves (much faster if computing distance many times)
    solver = pp3d.MeshHeatMethodDistanceSolver(vertices, faces)

    vids = []

    for i in range(num_points):
        if not vids:
            # Use index 0 vertex as the initial vertex.
            dist = solver.compute_distance(0)
        else:
            dist = solver.compute_distance_multisource(vids)
        vids.append(np.argmax(dist))

    samples = vertices[vids]

    return samples, vids



if __name__ == "__main__":
    num_points = 16
    mesh_filename = "/home/kaist984/docker/dataset/NeuralKeypoints/SMPL/upper_template/tet_full.mesh__sf.obj"
    #mesh_filename = "stanford-bunny.obj"

    print("Loading '{}'...".format(mesh_filename))

    mesh = trimesh.load(mesh_filename, process=False)
    vertices, faces = mesh.vertices, mesh.faces
    n_vertices = np.shape(vertices)[0]
    n_faces = np.shape(faces)[0]
    print("# vertices: {}".format(n_vertices))
    print("# faces: {}".format(n_faces))

    samples, indices = geodesic_farthest_point_sampling_heat(
        vertices, faces, num_points)

    print(samples)
    out_filename = "samples.npy"
    np.save(out_filename, samples)
    print("Saved '{}'.".format(out_filename))

