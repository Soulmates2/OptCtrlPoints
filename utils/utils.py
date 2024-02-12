import os, sys
import numpy as np
import pytorch3d.io
import torch
import torch.nn.functional as F
from einops import repeat
import trimesh
from PIL import Image


renderer_path = '/home/kaist984/libigl-renderer/build/OSMesaRenderer'


def sample_farthest_points(points, num_samples, return_index=False):
    b, c, n = points.shape
    sampled = torch.zeros((b, 3, num_samples), device=points.device, dtype=points.dtype)
    indexes = torch.zeros((b, num_samples), device=points.device, dtype=torch.int64)

    index = torch.randint(n, [b], device=points.device)

    gather_index = repeat(index, 'b -> b c 1', c=c)
    sampled[:, :, 0] = torch.gather(points, 2, gather_index)[:, :, 0]
    indexes[:, 0] = index
    dists = torch.norm(sampled[:, :, 0][:, :, None] - points, dim=1)

    # iteratively sample farthest points
    for i in range(1, num_samples):
        _, index = torch.max(dists, dim=1)
        gather_index = repeat(index, 'b -> b c 1', c=c)
        sampled[:, :, i] = torch.gather(points, 2, gather_index)[:, :, 0]
        indexes[:, i] = index
        dists = torch.min(dists, torch.norm(sampled[:, :, i][:, :, None] - points, dim=1))

    if return_index:
        return sampled, indexes
    else:
        return sampled


def nn_without_replacement(pred_keypoints, points):
    b, n, c = points.shape
    _, k, _ = pred_keypoints.shape

    indexes = torch.zeros((b, k), device=points.device, dtype=torch.int64)
    dists = torch.cdist(pred_keypoints, points) ## (b , k , n)

    ### Randomize order of assigning neighbors to keypoints
    idx_order = torch.randperm(k, device=points.device)

    # iteratively sample farthest points
    for i in range(k):

        _, nn_idx = torch.min(dists[:, idx_order[i], :], dim=-1)

        ## set indexes
        indexes[:, idx_order[i]] = nn_idx

        ## set distance to max for 100.0 for indices already selected
        dists = dists.scatter_(dim=-1, index=nn_idx.unsqueeze(-1).unsqueeze(-1).repeat(1,k,1), value=100.0)

    # indexes = F.one_hot(indexes, num_classes=n)
    # logits = F.softmax(dists, dim=-1)

    # ### VQVAE trick
    # indexes = logits + (indexes - logits).detach()
    # indexes = torch.argmax(indexes, dim=-1)

    return indexes

##########################################################################################
############################### Visualization ############################################
##########################################################################################
def render_output(mesh_file, face_labels_file, point_cloud_file, point_labels_file, snapshot_file, outfile=False, filehandle_=None):
    g_renderer = renderer_path
    g_azimuth_deg = -70
    g_elevation_deg = 20
    g_theta_deg = 0
    point_radius = 0.03

    # g_azimuth_deg = 110
    # g_elevation_deg = 25
    # g_theta_deg = 0
    # point_radius = 0.045

    if not outfile:
        cmd = g_renderer + ' \\\n'
        cmd += ' --mesh=' + mesh_file + ' \\\n'
        cmd += ' --face_labels=' + face_labels_file + ' \\\n'
        cmd += ' --point_cloud=' + point_cloud_file + ' \\\n'
        cmd += ' --point_labels=' + point_labels_file + ' \\\n'
        cmd += ' --snapshot=' + snapshot_file + ' \\\n'
        cmd += ' --azimuth_deg=' + str(g_azimuth_deg) + ' \\\n'
        cmd += ' --elevation_deg=' + str(g_elevation_deg) + ' \\\n'
        cmd += ' --theta_deg=' + str(g_theta_deg) + ' \\\n'
        cmd += ' --point_radius=' + str(point_radius) + ' \\\n'
        cmd += ' >/dev/null 2>&1'
    else:
        cmd = g_renderer + ' --mesh=' + mesh_file + ' --face_labels=' + face_labels_file + ' --point_cloud=' + point_cloud_file \
                + ' --point_labels=' + point_labels_file + ' --snapshot=' + snapshot_file \
                +' --camera_distance=' + str(3) + ' --azimuth_deg=' + str(g_azimuth_deg) + ' --elevation_deg=' + str(g_elevation_deg) \
                + ' --theta_deg=' + str(g_theta_deg) + ' --point_radius=' + str(point_radius) + ' --auto_adjust_camera=false' + ' >/dev/null 2>&1'

    if outfile:
        filehandle_.write(cmd+'\n')
    else:
        os.system(cmd)
    snapshot_file += '.png'
    print("Saved '{}'.".format(snapshot_file))


def render_front(mesh_file, face_labels_file, point_cloud_file, point_labels_file, snapshot_file, outfile=False, filehandle_=None):
    g_renderer = renderer_path
    g_azimuth_deg = 0
    g_elevation_deg = 20
    g_theta_deg = 0
    point_radius = 0.03

    if not outfile:
        cmd = g_renderer + ' \\\n'
        cmd += ' --mesh=' + mesh_file + ' \\\n'
        cmd += ' --face_labels=' + face_labels_file + ' \\\n'
        cmd += ' --point_cloud=' + point_cloud_file + ' \\\n'
        cmd += ' --point_labels=' + point_labels_file + ' \\\n'
        cmd += ' --snapshot=' + snapshot_file + ' \\\n'
        cmd += ' --azimuth_deg=' + str(g_azimuth_deg) + ' \\\n'
        cmd += ' --elevation_deg=' + str(g_elevation_deg) + ' \\\n'
        cmd += ' --theta_deg=' + str(g_theta_deg) + ' \\\n'
        cmd += ' --point_radius=' + str(point_radius) + ' \\\n'
        cmd += ' >/dev/null 2>&1'
    else:
        cmd = g_renderer + ' --mesh=' + mesh_file + ' --face_labels=' + face_labels_file + ' --point_cloud=' + point_cloud_file \
                + ' --point_labels=' + point_labels_file + ' --snapshot=' + snapshot_file \
                +' --camera_distance=' + str(3) + ' --azimuth_deg=' + str(g_azimuth_deg) + ' --elevation_deg=' + str(g_elevation_deg) \
                + ' --theta_deg=' + str(g_theta_deg) + ' --point_radius=' + str(point_radius) + ' --auto_adjust_camera=false' + ' >/dev/null 2>&1'

    if outfile:
        filehandle_.write(cmd+'\n')
    else:
        os.system(cmd)
    snapshot_file += '.png'
    print("Saved '{}'.".format(snapshot_file))


def render_back(mesh_file, face_labels_file, point_cloud_file, point_labels_file, snapshot_file, outfile=False, filehandle_=None):
    g_renderer = renderer_path
    g_azimuth_deg = 180
    g_elevation_deg = 20
    g_theta_deg = 0
    point_radius = 0.03

    if not outfile:
        cmd = g_renderer + ' \\\n'
        cmd += ' --mesh=' + mesh_file + ' \\\n'
        cmd += ' --face_labels=' + face_labels_file + ' \\\n'
        cmd += ' --point_cloud=' + point_cloud_file + ' \\\n'
        cmd += ' --point_labels=' + point_labels_file + ' \\\n'
        cmd += ' --snapshot=' + snapshot_file + ' \\\n'
        cmd += ' --azimuth_deg=' + str(g_azimuth_deg) + ' \\\n'
        cmd += ' --elevation_deg=' + str(g_elevation_deg) + ' \\\n'
        cmd += ' --theta_deg=' + str(g_theta_deg) + ' \\\n'
        cmd += ' --point_radius=' + str(point_radius) + ' \\\n'
        cmd += ' >/dev/null 2>&1'
    else:
        cmd = g_renderer + ' --mesh=' + mesh_file + ' --face_labels=' + face_labels_file + ' --point_cloud=' + point_cloud_file \
                + ' --point_labels=' + point_labels_file + ' --snapshot=' + snapshot_file \
                +' --camera_distance=' + str(3) + ' --azimuth_deg=' + str(g_azimuth_deg) + ' --elevation_deg=' + str(g_elevation_deg) \
                + ' --theta_deg=' + str(g_theta_deg) + ' --point_radius=' + str(point_radius) + ' --auto_adjust_camera=false' + ' >/dev/null 2>&1'

    if outfile:
        filehandle_.write(cmd+'\n')
    else:
        os.system(cmd)
    snapshot_file += '.png'
    print("Saved '{}'.".format(snapshot_file))


def render_point_cloud(point_cloud_file, point_labels_file, snapshot_file, outfile=False, filehandle_=None):
    g_renderer = renderer_path

    g_azimuth_deg = -70
    g_elevation_deg = 20
    g_theta_deg = 0

    # g_azimuth_deg = 110
    # g_elevation_deg = 25
    # g_theta_deg = 0

    if not outfile:
        cmd = g_renderer + ' \\\n'
        cmd += ' --point_cloud=' + point_cloud_file + ' \\\n'
        cmd += ' --point_labels=' + point_labels_file + ' \\\n'
        cmd += ' --snapshot=' + snapshot_file + ' \\\n'
        cmd += ' --azimuth_deg=' + str(g_azimuth_deg) + ' \\\n'
        cmd += ' --camera_distance=' + str(2.5) + ' \\\n'
        cmd += ' --elevation_deg=' + str(g_elevation_deg) + ' \\\n'
        cmd += ' --theta_deg=' + str(g_theta_deg) + ' \\\n'
        cmd += ' >/dev/null 2>&1'
    else:
        cmd = g_renderer + ' --point_cloud=' + point_cloud_file + ' --point_labels=' + point_labels_file + ' --snapshot=' + snapshot_file \
                +' --camera_distance=' + str(2.5) + ' --azimuth_deg=' + str(g_azimuth_deg) + ' --elevation_deg=' + str(g_elevation_deg) + ' --theta_deg=' + str(g_theta_deg) \
                + ' --theta_deg=' + str(g_theta_deg) + ' >/dev/null 2>&1'

    if outfile:
        filehandle_.write(cmd+'\n')
    else:
        os.system(cmd)
    snapshot_file += '.png'
    print("Saved '{}'.".format(snapshot_file))


def visualize_deformation_v2(model_id, output_folder, src_vertex, src_faces, keypoints, target_pc, deformed_vertex, def_keypoints, filehandle_1, filehandle_2):
    #### Source
    out_mesh_file = os.path.join(output_folder, model_id+'_source.obj')
    mesh_source = trimesh.Trimesh(vertices=src_vertex, faces=src_faces)
    mesh_source.export(out_mesh_file, os.path.splitext(out_mesh_file)[1][1:])
    print("Saved '{}'.".format(out_mesh_file))

    # Save vertex ids.
    out_vertex_ids_file = os.path.join(output_folder,  model_id + '_source_vertex_ids.txt')
    np.savetxt(out_vertex_ids_file, mesh_source.vertices, fmt='%d')
    print("Saved '{}'.".format(out_vertex_ids_file))

    # # Save face ids.
    out_face_ids_file = os.path.join(output_folder,  model_id + '_source_face_ids.txt')

    ### No segmentation
    face_labels = np.zeros(src_faces.shape[0])
    np.savetxt(out_face_ids_file, face_labels, fmt='%d')
    print("Saved '{}'.".format(out_face_ids_file))

    srckpt_point_cloud_file = os.path.join(output_folder, 'point_cloud', model_id+'_src_kpt.xyz')
    np.savetxt(srckpt_point_cloud_file, keypoints, delimiter=' ', fmt='%f')
    print("Saved '{}'.".format(srckpt_point_cloud_file))

    srckpt_ids_file = os.path.join(output_folder, 'point_cloud', model_id+'_src_label_ids.txt')
    label = np.arange(keypoints.shape[0])
    np.savetxt(srckpt_ids_file, label, fmt='%d')
    print("Saved '{}'.".format(srckpt_ids_file))


    ## Target
    target_point_cloud_file = os.path.join(output_folder, 'point_cloud', model_id+'_target.xyz')
    np.savetxt(target_point_cloud_file, target_pc, delimiter=' ', fmt='%f')
    print("Saved '{}'.".format(target_point_cloud_file))

    target_ids_file = os.path.join(output_folder, 'point_cloud', model_id+'_target_label_ids.txt')
    label = np.zeros(target_pc.shape[0])
    np.savetxt(target_ids_file, label, fmt='%d')
    print("Saved '{}'.".format(target_ids_file))

    ## Deformed
    def_mesh_file = os.path.join(output_folder, model_id+'_deformed.obj')
    mesh = trimesh.Trimesh(vertices=deformed_vertex, faces=src_faces)
    mesh.export(def_mesh_file, os.path.splitext(def_mesh_file)[1][1:])
    print("Saved '{}'.".format(def_mesh_file))

    # Save vertex ids.
    def_vertex_ids_file = os.path.join(output_folder,  model_id + '_def_vertex_ids.txt')
    np.savetxt(def_vertex_ids_file, mesh.vertices, fmt='%d')
    print("Saved '{}'.".format(def_vertex_ids_file))

    defkpt_point_cloud_file = os.path.join(output_folder, 'point_cloud', model_id+'_def_kpt.xyz')
    np.savetxt(defkpt_point_cloud_file, def_keypoints, delimiter=' ', fmt='%f')
    print("Saved '{}'.".format(defkpt_point_cloud_file))

    defkpt_ids_file = os.path.join(output_folder, 'point_cloud', model_id+'_src_label_ids.txt')
    label = np.arange(def_keypoints.shape[0])
    np.savetxt(defkpt_ids_file, label, fmt='%d')
    print("Saved '{}'.".format(defkpt_ids_file))

    # Render source.
    source_snapshot_file = os.path.join(output_folder,  model_id+'_source_kp')
    render_output(out_mesh_file, out_face_ids_file, srckpt_point_cloud_file, srckpt_ids_file, source_snapshot_file, outfile=True, filehandle_=filehandle_1)

    # Render target.
    target_snapshot_file = os.path.join(output_folder, 'tmp', model_id+'_target')
    render_point_cloud(target_point_cloud_file, target_ids_file,
            target_snapshot_file, outfile=True, filehandle_=filehandle_1)

    # Render deformed
    deformed_snapshot_file = os.path.join(output_folder,  model_id+'_def_kp')
    render_output(def_mesh_file, out_face_ids_file, defkpt_point_cloud_file, defkpt_ids_file, deformed_snapshot_file, outfile=True, filehandle_=filehandle_1)


    # Save filename to combine
    line = source_snapshot_file + ".png" + " " + target_snapshot_file + ".png" + " " + deformed_snapshot_file + ".png"+ "\n"
    filehandle_2.write(line)

    return


def visualize_deformation_all(model_id, output_folder, src_vertex, src_faces, keypoints, tgt_vertex, tgt_faces, offsets, deformed_vertex, def_keypoints, filehandle_1, filehandle_2):
    #### Source
    out_mesh_file = os.path.join(output_folder, model_id+'_source.obj')
    mesh_source = trimesh.Trimesh(vertices=src_vertex, faces=src_faces)
    mesh_source.export(out_mesh_file, os.path.splitext(out_mesh_file)[1][1:])
    print("Saved '{}'.".format(out_mesh_file))

    # Save vertex ids.
    out_vertex_ids_file = os.path.join(output_folder,  model_id + '_source_vertex_ids.txt')
    np.savetxt(out_vertex_ids_file, mesh_source.vertices, fmt='%d')
    print("Saved '{}'.".format(out_vertex_ids_file))

    # # Save face ids.
    out_face_ids_file = os.path.join(output_folder,  model_id + '_source_face_ids.txt')

    ### No segmentation
    face_labels = np.zeros(src_faces.shape[0])
    np.savetxt(out_face_ids_file, face_labels, fmt='%d')
    print("Saved '{}'.".format(out_face_ids_file))

    srckpt_point_cloud_file = os.path.join(output_folder, 'point_cloud', model_id+'_src_kpt.xyz')
    np.savetxt(srckpt_point_cloud_file, keypoints, delimiter=' ', fmt='%f')
    print("Saved '{}'.".format(srckpt_point_cloud_file))

    srckpt_ids_file = os.path.join(output_folder, 'point_cloud', model_id+'_src_label_ids.txt')
    label = np.arange(keypoints.shape[0])
    np.savetxt(srckpt_ids_file, label, fmt='%d')
    print("Saved '{}'.".format(srckpt_ids_file))


    ## Target
    tgt_mesh_file = os.path.join(output_folder, model_id+'_target.obj')
    mesh_target = trimesh.Trimesh(vertices=tgt_vertex, faces=tgt_faces)
    mesh_target.export(tgt_mesh_file, os.path.splitext(tgt_mesh_file)[1][1:])
    print("Saved '{}'.".format(tgt_mesh_file))

    # Save vertex ids.
    tgt_vertex_ids_file = os.path.join(output_folder,  model_id + '_target_vertex_ids.txt')
    np.savetxt(tgt_vertex_ids_file, mesh_target.vertices, fmt='%d')
    print("Saved '{}'.".format(tgt_vertex_ids_file))

    # # Save face ids.
    tgt_face_ids_file = os.path.join(output_folder,  model_id + '_target_face_ids.txt')

    ### No segmentation
    face_labels = np.zeros(tgt_faces.shape[0])
    np.savetxt(tgt_face_ids_file, face_labels, fmt='%d')
    print("Saved '{}'.".format(tgt_face_ids_file))

    tgtkpt_point_cloud_file = os.path.join(output_folder, 'point_cloud', model_id+'_tgt_kpt.xyz')
    np.savetxt(tgtkpt_point_cloud_file, offsets, delimiter=' ', fmt='%f')
    print("Saved '{}'.".format(tgtkpt_point_cloud_file))

    tgtkpt_ids_file = os.path.join(output_folder, 'point_cloud', model_id+'_tgt_label_ids.txt')
    label = np.arange(offsets.shape[0])
    np.savetxt(tgtkpt_ids_file, label, fmt='%d')
    print("Saved '{}'.".format(tgtkpt_ids_file))

    target_point_cloud_file = os.path.join(output_folder, 'point_cloud', model_id+'_target.xyz')
    np.savetxt(target_point_cloud_file, tgt_vertex, delimiter=' ', fmt='%f')
    print("Saved '{}'.".format(target_point_cloud_file))

    target_ids_file = os.path.join(output_folder, 'point_cloud', model_id+'_target_label_ids.txt')
    label = np.zeros(tgt_vertex.shape[0])
    np.savetxt(target_ids_file, label, fmt='%d')
    print("Saved '{}'.".format(target_ids_file))


    ## Deformed
    def_mesh_file = os.path.join(output_folder, model_id+'_deformed.obj')
    mesh = trimesh.Trimesh(vertices=deformed_vertex, faces=src_faces)
    mesh.export(def_mesh_file, os.path.splitext(def_mesh_file)[1][1:])
    print("Saved '{}'.".format(def_mesh_file))

    # Save vertex ids.
    def_vertex_ids_file = os.path.join(output_folder,  model_id + '_def_vertex_ids.txt')
    np.savetxt(def_vertex_ids_file, mesh.vertices, fmt='%d')
    print("Saved '{}'.".format(def_vertex_ids_file))

    defkpt_point_cloud_file = os.path.join(output_folder, 'point_cloud', model_id+'_def_kpt.xyz')
    np.savetxt(defkpt_point_cloud_file, def_keypoints, delimiter=' ', fmt='%f')
    print("Saved '{}'.".format(defkpt_point_cloud_file))

    defkpt_ids_file = os.path.join(output_folder, 'point_cloud', model_id+'_src_label_ids.txt')
    label = np.arange(def_keypoints.shape[0])
    np.savetxt(defkpt_ids_file, label, fmt='%d')
    print("Saved '{}'.".format(defkpt_ids_file))

    # Render source.
    source_snapshot_file = os.path.join(output_folder,  model_id+'_source_kp')
    render_output(out_mesh_file, out_face_ids_file, srckpt_point_cloud_file, srckpt_ids_file, source_snapshot_file, outfile=True, filehandle_=filehandle_1)

    # Render target.
    target_snapshot_file = os.path.join(output_folder,  model_id+'_target_kp')
    render_output(tgt_mesh_file, tgt_face_ids_file, tgtkpt_point_cloud_file, tgtkpt_ids_file, target_snapshot_file, outfile=True, filehandle_=filehandle_1)
    # target_snapshot_file = os.path.join(output_folder, 'tmp', model_id+'_target')
    # render_point_cloud(target_point_cloud_file, target_ids_file,
    #         target_snapshot_file, outfile=True, filehandle_=filehandle_1)

    # Render deformed
    deformed_snapshot_file = os.path.join(output_folder,  model_id+'_def_kp')
    render_output(def_mesh_file, out_face_ids_file, defkpt_point_cloud_file, defkpt_ids_file, deformed_snapshot_file, outfile=True, filehandle_=filehandle_1)

    # Save filename to combine
    line = source_snapshot_file + ".png" + " " + target_snapshot_file + ".png" + " " + deformed_snapshot_file + ".png"+ "\n"
    filehandle_2.write(line)

    return


def visualize_source_front_and_back(model_id, output_folder, src_vertex, src_faces, keypoints, tgt_vertex, tgt_faces, offsets, deformed_vertex, def_keypoints, filehandle_1, filehandle_2):
    #### Source
    out_mesh_file = os.path.join(output_folder, model_id+'_source.obj')
    mesh_source = trimesh.Trimesh(vertices=src_vertex, faces=src_faces)
    mesh_source.export(out_mesh_file, os.path.splitext(out_mesh_file)[1][1:])
    print("Saved '{}'.".format(out_mesh_file))

    # Save vertex ids.
    out_vertex_ids_file = os.path.join(output_folder,  model_id + '_source_vertex_ids.txt')
    np.savetxt(out_vertex_ids_file, mesh_source.vertices, fmt='%d')
    print("Saved '{}'.".format(out_vertex_ids_file))

    # # Save face ids.
    out_face_ids_file = os.path.join(output_folder,  model_id + '_source_face_ids.txt')

    ### No segmentation
    face_labels = np.zeros(src_faces.shape[0])
    np.savetxt(out_face_ids_file, face_labels, fmt='%d')
    print("Saved '{}'.".format(out_face_ids_file))

    srckpt_point_cloud_file = os.path.join(output_folder, 'kpts', model_id+'_src_kpt.xyz')
    np.savetxt(srckpt_point_cloud_file, keypoints, delimiter=' ', fmt='%f')
    print("Saved '{}'.".format(srckpt_point_cloud_file))

    srckpt_ids_file = os.path.join(output_folder, 'kpts', model_id+'_src_label_ids.txt')
    label = np.arange(keypoints.shape[0])
    np.savetxt(srckpt_ids_file, label, fmt='%d')
    print("Saved '{}'.".format(srckpt_ids_file))


    ## Target
    tgt_mesh_file = os.path.join(output_folder, model_id+'_target.obj')
    mesh_target = trimesh.Trimesh(vertices=tgt_vertex, faces=tgt_faces)
    mesh_target.export(tgt_mesh_file, os.path.splitext(tgt_mesh_file)[1][1:])
    print("Saved '{}'.".format(tgt_mesh_file))

    # Save vertex ids.
    tgt_vertex_ids_file = os.path.join(output_folder,  model_id + '_target_vertex_ids.txt')
    np.savetxt(tgt_vertex_ids_file, mesh_target.vertices, fmt='%d')
    print("Saved '{}'.".format(tgt_vertex_ids_file))

    # # Save face ids.
    tgt_face_ids_file = os.path.join(output_folder,  model_id + '_target_face_ids.txt')

    ### No segmentation
    face_labels = np.zeros(tgt_faces.shape[0])
    np.savetxt(tgt_face_ids_file, face_labels, fmt='%d')
    print("Saved '{}'.".format(tgt_face_ids_file))

    tgtkpt_point_cloud_file = os.path.join(output_folder, 'kpts', model_id+'_tgt_kpt.xyz')
    np.savetxt(tgtkpt_point_cloud_file, offsets, delimiter=' ', fmt='%f')
    print("Saved '{}'.".format(tgtkpt_point_cloud_file))

    tgtkpt_ids_file = os.path.join(output_folder, 'kpts', model_id+'_tgt_label_ids.txt')
    label = np.arange(offsets.shape[0])
    np.savetxt(tgtkpt_ids_file, label, fmt='%d')
    print("Saved '{}'.".format(tgtkpt_ids_file))

    target_point_cloud_file = os.path.join(output_folder, 'kpts', model_id+'_target.xyz')
    np.savetxt(target_point_cloud_file, tgt_vertex, delimiter=' ', fmt='%f')
    print("Saved '{}'.".format(target_point_cloud_file))

    target_ids_file = os.path.join(output_folder, 'kpts', model_id+'_target_label_ids.txt')
    label = np.zeros(tgt_vertex.shape[0])
    np.savetxt(target_ids_file, label, fmt='%d')
    print("Saved '{}'.".format(target_ids_file))


    ## Deformed
    def_mesh_file = os.path.join(output_folder, model_id+'_deformed.obj')
    mesh = trimesh.Trimesh(vertices=deformed_vertex, faces=src_faces)
    mesh.export(def_mesh_file, os.path.splitext(def_mesh_file)[1][1:])
    print("Saved '{}'.".format(def_mesh_file))

    # Save vertex ids.
    def_vertex_ids_file = os.path.join(output_folder,  model_id + '_def_vertex_ids.txt')
    np.savetxt(def_vertex_ids_file, mesh.vertices, fmt='%d')
    print("Saved '{}'.".format(def_vertex_ids_file))

    defkpt_point_cloud_file = os.path.join(output_folder, 'kpts', model_id+'_def_kpt.xyz')
    np.savetxt(defkpt_point_cloud_file, def_keypoints, delimiter=' ', fmt='%f')
    print("Saved '{}'.".format(defkpt_point_cloud_file))

    defkpt_ids_file = os.path.join(output_folder, 'kpts', model_id+'_src_label_ids.txt')
    label = np.arange(def_keypoints.shape[0])
    np.savetxt(defkpt_ids_file, label, fmt='%d')
    print("Saved '{}'.".format(defkpt_ids_file))

    # Render source.
    source_snapshot_file = os.path.join(output_folder,  model_id+'_source_kp_front')
    render_front(out_mesh_file, out_face_ids_file, srckpt_point_cloud_file, srckpt_ids_file, source_snapshot_file, outfile=True, filehandle_=filehandle_1)
    source_snapshot_file = os.path.join(output_folder,  model_id+'_source_kp_back')
    render_back(out_mesh_file, out_face_ids_file, srckpt_point_cloud_file, srckpt_ids_file, source_snapshot_file, outfile=True, filehandle_=filehandle_1)

    # Render target.
    target_snapshot_file = os.path.join(output_folder,  model_id+'_target_kp_front')
    render_front(tgt_mesh_file, tgt_face_ids_file, tgtkpt_point_cloud_file, tgtkpt_ids_file, target_snapshot_file, outfile=True, filehandle_=filehandle_1)
    target_snapshot_file = os.path.join(output_folder,  model_id+'_target_kp_back')
    render_back(tgt_mesh_file, tgt_face_ids_file, tgtkpt_point_cloud_file, tgtkpt_ids_file, target_snapshot_file, outfile=True, filehandle_=filehandle_1)

    # Render deformed
    deformed_snapshot_file = os.path.join(output_folder,  model_id+'_def_kp_front')
    render_front(def_mesh_file, out_face_ids_file, defkpt_point_cloud_file, defkpt_ids_file, deformed_snapshot_file, outfile=True, filehandle_=filehandle_1)
    deformed_snapshot_file = os.path.join(output_folder,  model_id+'_def_kp_back')
    render_back(def_mesh_file, out_face_ids_file, defkpt_point_cloud_file, defkpt_ids_file, deformed_snapshot_file, outfile=True, filehandle_=filehandle_1)

    return



def visualize_deformation_images(model_id, output_folder, src_vertex, src_faces, keypoints, target_id, deformed_vertex, def_keypoints, filehandle_1, filehandle_2):
    #### Source
    out_mesh_file = os.path.join(output_folder, model_id+'_source.obj')
    mesh_source = trimesh.Trimesh(vertices=src_vertex, faces=src_faces)
    mesh_source.export(out_mesh_file, os.path.splitext(out_mesh_file)[1][1:])
    print("Saved '{}'.".format(out_mesh_file))

    # Save vertex ids.
    out_vertex_ids_file = os.path.join(output_folder,  model_id + '_source_vertex_ids.txt')
    np.savetxt(out_vertex_ids_file, mesh_source.vertices, fmt='%d')
    print("Saved '{}'.".format(out_vertex_ids_file))

    # # Save face ids.
    out_face_ids_file = os.path.join(output_folder,  model_id + '_source_face_ids.txt')

    ### No segmentation
    face_labels = np.zeros(src_faces.shape[0])
    np.savetxt(out_face_ids_file, face_labels, fmt='%d')
    print("Saved '{}'.".format(out_face_ids_file))

    srckpt_point_cloud_file = os.path.join(output_folder, 'point_cloud', model_id+'_src_kpt.xyz')
    np.savetxt(srckpt_point_cloud_file, keypoints, delimiter=' ', fmt='%f')
    print("Saved '{}'.".format(srckpt_point_cloud_file))

    srckpt_ids_file = os.path.join(output_folder, 'point_cloud', model_id+'_src_label_ids.txt')
    label = np.arange(keypoints.shape[0])
    np.savetxt(srckpt_ids_file, label, fmt='%d')
    print("Saved '{}'.".format(srckpt_ids_file))

    ### Target
    # Load the image view
    height = 1080
    width = 1920
    IMAGE_BASE_DIR = "/orion/downloads/partnet_dataset/partnet_rgb_masks_chair/"
    img_filename = os.path.join(IMAGE_BASE_DIR, str(int(target_id)), "view-"+str(17).zfill(2), "shape-rgb.png")

    img = Image.open(img_filename)
    old_size = img.size
    ratio = float(height)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    img_resized = img.resize(new_size, Image.ANTIALIAS)
    padded_img = Image.new("RGBA", (width, height))
    padded_img.paste(img_resized, ((width-new_size[0])//2, (height-new_size[1])//2))
    target_snapshot_file = os.path.join(output_folder, 'tmp', model_id+'_target.png')
    padded_img.save(target_snapshot_file)

    ## Deformed
    def_mesh_file = os.path.join(output_folder, model_id+'_deformed.obj')
    mesh = trimesh.Trimesh(vertices=deformed_vertex, faces=src_faces)
    mesh.export(def_mesh_file, os.path.splitext(def_mesh_file)[1][1:])
    print("Saved '{}'.".format(def_mesh_file))

    # Save vertex ids.
    def_vertex_ids_file = os.path.join(output_folder,  model_id + '_def_vertex_ids.txt')
    np.savetxt(def_vertex_ids_file, mesh.vertices, fmt='%d')
    print("Saved '{}'.".format(def_vertex_ids_file))

    defkpt_point_cloud_file = os.path.join(output_folder, 'point_cloud', model_id+'_def_kpt.xyz')
    np.savetxt(defkpt_point_cloud_file, def_keypoints, delimiter=' ', fmt='%f')
    print("Saved '{}'.".format(defkpt_point_cloud_file))

    defkpt_ids_file = os.path.join(output_folder, 'point_cloud', model_id+'_src_label_ids.txt')
    label = np.arange(def_keypoints.shape[0])
    np.savetxt(defkpt_ids_file, label, fmt='%d')
    print("Saved '{}'.".format(defkpt_ids_file))

    # Render source.
    source_snapshot_file = os.path.join(output_folder,  model_id+'_source_kp')
    render_output(out_mesh_file, out_face_ids_file, srckpt_point_cloud_file, srckpt_ids_file, source_snapshot_file, outfile=True, filehandle_=filehandle_1)

    # Render deformed
    deformed_snapshot_file = os.path.join(output_folder,  model_id+'_def_kp')
    render_output(def_mesh_file, out_face_ids_file, defkpt_point_cloud_file, defkpt_ids_file, deformed_snapshot_file, outfile=True, filehandle_=filehandle_1)


    # Save filename to combine
    line = source_snapshot_file + ".png" + " " + target_snapshot_file + " " + deformed_snapshot_file + ".png"+ "\n"
    filehandle_2.write(line)

    return


def visualize_deformation(model_id, output_folder, src_surface, kpidx, target_pc, deformed_pc, filehandle_1, filehandle_2):
    ###
    ### Point Cloud ###
    ## Source.
    srcsurface_point_cloud_file = os.path.join(output_folder, 'point_cloud', model_id+'_src_surface.xyz')
    np.savetxt(srcsurface_point_cloud_file, src_surface, delimiter=' ', fmt='%f')
    print("Saved '{}'.".format(srcsurface_point_cloud_file))

    srcsurface_ids_file = os.path.join(output_folder, 'point_cloud', model_id+'_src_label_ids.txt')
    label = np.zeros(src_surface.shape[0])
    label[kpidx] = 2
    np.savetxt(srcsurface_ids_file, label, fmt='%d')
    print("Saved '{}'.".format(srcsurface_ids_file))

    ## Target
    target_point_cloud_file = os.path.join(output_folder, 'point_cloud', model_id+'_target.xyz')
    np.savetxt(target_point_cloud_file, target_pc, delimiter=' ', fmt='%f')
    print("Saved '{}'.".format(target_point_cloud_file))

    target_ids_file = os.path.join(output_folder, 'point_cloud', model_id+'_target_label_ids.txt')
    label = np.zeros(target_pc.shape[0])
    np.savetxt(target_ids_file, label, fmt='%d')
    print("Saved '{}'.".format(target_ids_file))

    ## Deformed
    deformed_point_cloud_file = os.path.join(output_folder, 'point_cloud', model_id+'_deformed.xyz')
    np.savetxt(deformed_point_cloud_file, deformed_pc, delimiter=' ', fmt='%f')
    print("Saved '{}'.".format(deformed_point_cloud_file))

    deformed_ids_file = os.path.join(output_folder, 'point_cloud', model_id+'_deformed_label_ids.txt')
    label = np.zeros(deformed_pc.shape[0])
    np.savetxt(deformed_ids_file, label, fmt='%d')
    print("Saved '{}'.".format(deformed_ids_file))

    # Render source.
    source_snapshot_file = os.path.join(output_folder, 'tmp', model_id+'_source')
    render_point_cloud(srcsurface_point_cloud_file, srcsurface_ids_file,
            source_snapshot_file, outfile=True, filehandle_=filehandle_1)

    # Render target.
    target_snapshot_file = os.path.join(output_folder, 'tmp', model_id+'_target')
    render_point_cloud(target_point_cloud_file, target_ids_file,
            target_snapshot_file, outfile=True, filehandle_=filehandle_1)

    # Render deformed
    deformed_snapshot_file = os.path.join(output_folder, 'tmp', model_id+'_deformed')
    render_point_cloud(deformed_point_cloud_file, deformed_ids_file,
            deformed_snapshot_file, outfile=True, filehandle_=filehandle_1)

    # Save filename to combine
    line = source_snapshot_file + ".png" + " " + target_snapshot_file + ".png" + " " + deformed_snapshot_file + ".png"+ "\n"
    filehandle_2.write(line)

    return


def visualize_reconstruction(model_id, output_folder, src_surface, kpidx, curr_pred_keypoints, curr_recons_pc, filehandle_1, filehandle_2):
    ###
    ### Point Cloud ###
    ## Source.
    srcsurface_point_cloud_file = os.path.join(output_folder, 'point_cloud', model_id+'_src_surface_recons.xyz')
    np.savetxt(srcsurface_point_cloud_file, src_surface, delimiter=' ', fmt='%f')
    print("Saved '{}'.".format(srcsurface_point_cloud_file))

    srcsurface_ids_file = os.path.join(output_folder, 'point_cloud', model_id+'_src_label_ids_recons.txt')
    label = np.zeros(src_surface.shape[0])
    label[kpidx] = 2
    np.savetxt(srcsurface_ids_file, label, fmt='%d')
    print("Saved '{}'.".format(srcsurface_ids_file))

    ## Target
    stacked_pc = np.vstack((curr_pred_keypoints, src_surface))

    predkp_point_cloud_file = os.path.join(output_folder, 'point_cloud', model_id+'_predkp.xyz')
    np.savetxt(predkp_point_cloud_file, stacked_pc, delimiter=' ', fmt='%f')
    print("Saved '{}'.".format(predkp_point_cloud_file))

    predkp_ids_file = os.path.join(output_folder, 'point_cloud', model_id+'_predkp_label_ids.txt')
    label = np.zeros(stacked_pc.shape[0])
    label[:curr_pred_keypoints.shape[0]] = 1
    np.savetxt(predkp_ids_file, label, fmt='%d')
    print("Saved '{}'.".format(predkp_ids_file))

    ## Deformed
    recons_point_cloud_file = os.path.join(output_folder, 'point_cloud', model_id+'_recons.xyz')
    np.savetxt(recons_point_cloud_file, curr_recons_pc, delimiter=' ', fmt='%f')
    print("Saved '{}'.".format(recons_point_cloud_file))

    recons_ids_file = os.path.join(output_folder, 'point_cloud', model_id+'_recons_label_ids.txt')
    label = np.zeros(curr_recons_pc.shape[0])
    np.savetxt(recons_ids_file, label, fmt='%d')
    print("Saved '{}'.".format(recons_ids_file))

    # Render source.
    source_snapshot_file = os.path.join(output_folder, 'tmp', model_id+'_sourcerecons')
    render_point_cloud(srcsurface_point_cloud_file, srcsurface_ids_file,
            source_snapshot_file, outfile=True, filehandle_=filehandle_1)

    # Render target.
    predkp_snapshot_file = os.path.join(output_folder, 'tmp', model_id+'_kprecons')
    render_point_cloud(predkp_point_cloud_file, predkp_ids_file,
            predkp_snapshot_file, outfile=True, filehandle_=filehandle_1)

    # Render deformed
    recons_snapshot_file = os.path.join(output_folder, 'tmp', model_id+'_pcrecons')
    render_point_cloud(recons_point_cloud_file, recons_ids_file,
            recons_snapshot_file, outfile=True, filehandle_=filehandle_1)

    # Save filename to combine
    line = source_snapshot_file + ".png" + " " + predkp_snapshot_file + ".png" + " " + recons_snapshot_file + ".png"+ "\n"
    filehandle_2.write(line)

    return


### Render mesh for FID score
def render_mesh(mesh_file, face_labels_file, snapshot_file, outfile=False, filehandle_=None):
    g_renderer = renderer_path
    g_azimuth_deg = -70
    g_elevation_deg = 20
    g_theta_deg = 0

    if not outfile:
        cmd = g_renderer + ' \\\n'
        cmd += ' --mesh=' + mesh_file + ' \\\n'
        cmd += ' --face_labels=' + face_labels_file + ' \\\n'
        cmd += ' --snapshot=' + snapshot_file + ' \\\n'
        cmd += ' --azimuth_deg=' + str(g_azimuth_deg) + ' \\\n'
        cmd += ' --elevation_deg=' + str(g_elevation_deg) + ' \\\n'
        cmd += ' --theta_deg=' + str(g_theta_deg) + ' \\\n'
        cmd += ' >/dev/null 2>&1'
    else:
        cmd = g_renderer + ' --mesh=' + mesh_file + ' --face_labels=' + face_labels_file + ' --snapshot=' + snapshot_file +\
                ' --azimuth_deg=' + str(g_azimuth_deg) + ' --elevation_deg=' + str(g_elevation_deg) + ' --theta_deg=' + str(g_theta_deg) + ' >/dev/null 2>&1'

    if outfile:
        filehandle_.write(cmd+'\n')
    else:
        os.system(cmd)
    snapshot_file += '.png'
    print("Saved '{}'.".format(snapshot_file))

def output_mesh(model_id, output_folder, src_vertex, src_faces, filehandle_1):
    out_mesh_file = os.path.join(output_folder, "../tmp", model_id+'_mesh.obj')
    mesh = trimesh.Trimesh(vertices=src_vertex, faces=src_faces)
    mesh.export(out_mesh_file, os.path.splitext(out_mesh_file)[1][1:])
    print("Saved '{}'.".format(out_mesh_file))

    # Save vertex ids.
    out_vertex_ids_file = os.path.join(output_folder, "../tmp",  model_id + '_vertex_ids.txt')
    np.savetxt(out_vertex_ids_file, mesh.vertices, fmt='%d')
    print("Saved '{}'.".format(out_vertex_ids_file))

    # # Save face ids.
    out_face_ids_file = os.path.join(output_folder, "../tmp",  model_id + '_face_ids.txt')

    ### No segmentation
    face_labels = np.zeros(src_faces.shape[0])

    np.savetxt(out_face_ids_file, face_labels, fmt='%d')
    print("Saved '{}'.".format(out_face_ids_file))

    # Render mesh.
    final_mesh_snapshot_file = os.path.join(output_folder,  model_id+'_mesh')

    render_mesh(out_mesh_file, out_face_ids_file, final_mesh_snapshot_file, outfile=True, filehandle_=filehandle_1)


