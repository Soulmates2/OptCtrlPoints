# Kunho Kim (kaist984@kaist.ac.kr)
import os
import io
import sys
import argparse
import datetime
import numpy as np
import torch
import torch.nn.functional as F
import pytorch3d
import trimesh
import gc
from pytorch3d.loss import chamfer_distance
from pytorch3d.io import load_obj, save_obj
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import linear_sum_assignment

from deformation import *
from dataloader.dataloader import *
from utils.utils import *
from utils.loss import *
from utils.geodesic_fps import *


def seed_everything(seed=2023):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log_string(out_str):
	LOG.write(out_str+'\n')
	LOG.flush()
	print(out_str)   


def make_output_dump_dir(out_path):
    if not os.path.exists(out_path): os.makedirs(out_path)
    kpts_dir = os.path.join(out_path, "kpts")
    if not os.path.exists(kpts_dir): os.mkdir(kpts_dir)
    matrix_dir = os.path.join(out_path, "mat")
    if not os.path.exists(matrix_dir): os.mkdir(matrix_dir)
    point_cloud_dir = os.path.join(out_path, "point_cloud")
    if not os.path.exists(point_cloud_dir): os.mkdir(point_cloud_dir)
        

def make_output_render_file(out_path):
    f1 = open(os.path.join(out_path, "render.sh"), "w")
    g1 = open(os.path.join(out_path, "image_files.sh"), "w")
    return f1, g1


def nearest_neighbor(src, dst):
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def matching_indices(src_vertices, dst_vertices):
    '''
    src-to-dst matching with nearest_neighbor
    '''
    dist, src2dst_idx = nearest_neighbor(src_vertices.squeeze(0).detach().to("cpu").numpy(), dst_vertices.squeeze(0).detach().to("cpu").numpy())
    return src2dst_idx


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument('--source_data_path', type=str, default='/scratch/NeuralKeypoints/SMAL/smal_source_tetmesh/hippo', help='dataset path')
    parser.add_argument('--target_data_path', type=str, default='/scratch/NeuralKeypoints/SMAL/smal_target_tetmesh/hippo', help='dataset path')
    parser.add_argument('--log', type=eval, default=True, choices=[True, False], help='logging train result')
    parser.add_argument('--log_dir', type=str, default='log/SMAL/hippo_16_partition_random', help='log dir path')
    parser.add_argument('--dump_dir', type=str, default='dump/SMAL/hippo_16_partition_random', help='dump dir path')

    # model parameter
    parser.add_argument('--num_keypoints', type=int, default=16, help='the number key points') 
    parser.add_argument('--num_candidates', type=int, default=5000, help='farthest point sampling of source point cloud')
    parser.add_argument('--init_method', default='fps', choices=['fps'], help='initialization methods (default: %(default)s)') 
    parser.add_argument('--calc_w_method', default='prefactor', choices=['prefactor', 'pinv', 'original'], help='how to calculate the biharmonic weight W')
    parser.add_argument('--eval_metric', default='l2', choices=['l2', 'chamfer'], help='evaluation metric')


    # train parameter
    parser.add_argument('--seed', type=int, default=2023, help='seed number')
    parser.add_argument('--gpu', type=str, default='0', help='specify GPU devices')
    
    args = parser.parse_args()

    # fix seed
    seed_everything(args.seed)

    # set gpu
    device = f'cuda:{args.gpu}'
    print(f'Using GPU {args.gpu}')

    # make log dir
    if not os.path.exists(args.log_dir): os.makedirs(args.log_dir, exist_ok=True)
    LOG = open(os.path.join(args.log_dir, 'log.txt'), 'w')
    LOG.write(str(args)+'\n')
    LOG_loss = open(os.path.join(args.log_dir, 'partition_loss.txt'), 'w')
    os.system(f'cp {os.path.abspath(__file__)} {args.log_dir}')
    make_output_dump_dir(args.dump_dir)
    f1, g1 = make_output_render_file(args.dump_dir)


    # load source data and target data
    source_dataset = SourceData(args.source_data_path, device)
    src_norm_tet_vertices, src_norm_tet_faces, src_surface_vertices, src_surface_faces, src_tet_A, sur2tet_idx, _ = source_dataset.get_item()
    print("source tet vertices shape:", src_norm_tet_vertices.shape)
    print("source surface vertices shape:", src_surface_vertices.shape)

    ### pre-compute ##
    A_LU, pivots = torch.lu(src_tet_A)
    P, L, U = torch.lu_unpack(A_LU, pivots)

    print("P shape:", P.shape)
    print("L shape:", L.shape)
    print("U shape:", U.shape)

    np_src_tet_A = src_tet_A.to("cpu").numpy()
    A_inv = torch.from_numpy(np.linalg.pinv(np_src_tet_A, rcond=1e-20)).to(device, dtype=torch.float64)
    #################

    target_dataset = TargetData(args.target_data_path, device)
    tgt_vertices, tgt_faces = target_dataset.get_item()
    print("target vertices shape:", tgt_vertices.shape)
    tgt_batch_size = tgt_vertices.shape[0]
    num_tet_vertices = src_tet_A.shape[1]
    

    # geodesic FPS on surface mesh
    np_src_surface_vers = src_surface_vertices.squeeze(0).detach().to("cpu").numpy()
    np_src_surface_faces = src_surface_faces.squeeze(0).detach().to("cpu").numpy()
    src_surface_FPS_pc, src_surface_FPS_idx = geodesic_farthest_point_sampling_heat(np_src_surface_vers, np_src_surface_faces, args.num_keypoints)
    
    # map surface mesh FPS to tetmesh FPS
    sur2tet_idx = sur2tet_idx.squeeze().detach().to("cpu").numpy()
    src_tet_FPS_idx = sur2tet_idx[src_surface_FPS_idx]
    sur2tet_idx = torch.from_numpy(sur2tet_idx).unsqueeze(0).to(device, dtype=torch.int64)

    # make partitions of source tetmesh based on FPS points
    FPS_points = src_norm_tet_vertices[:, src_tet_FPS_idx, :]
    np_src_norm_tet_vertices = src_norm_tet_vertices
    tet2FPS_idx = matching_indices(np_src_norm_tet_vertices, FPS_points)
    
    # calculate the centroids of each partitions
    centroid_idx = torch.zeros((args.num_keypoints), dtype=torch.float64).to(device)
    for i in range(args.num_keypoints):
        partition_points = src_norm_tet_vertices[:, tet2FPS_idx == i, :]
        log_string(f'Partition {i} has {partition_points.shape[1]} points')
        centroid_i = torch.mean(partition_points, dim=1).unsqueeze(0)
        np_centroid_idx = matching_indices(centroid_i, np_src_norm_tet_vertices)
        centroid_idx[i] = torch.from_numpy(np_centroid_idx).to(device, dtype=torch.int64)
    log_string(f"Centroid indices {centroid_idx}")

    # make candidate matrix M (m x n)
    candidate_idx = np.arange(0, args.num_candidates)
    candidate_idx = torch.Tensor(candidate_idx).unsqueeze(0).to(device, dtype=torch.int64)
    M = torch.zeros((1, args.num_candidates, num_tet_vertices), dtype=torch.float64).to(device)
    M = M.scatter_(index=candidate_idx.unsqueeze(-1), dim=-1, value=1.0).to(torch.float64)
        
    # FPS initialize
    initial_kpts_idx = src_tet_FPS_idx
    initial_kpts_idx = torch.Tensor(initial_kpts_idx).unsqueeze(0).to(device, dtype=torch.int64)
    
    print("Initial keypoints indices")
    print(initial_kpts_idx)
    print(initial_kpts_idx.shape)
    print()


    # set initial s to one-hot
    s_ = torch.zeros(1, args.num_keypoints, args.num_candidates).to(device, dtype=torch.float64)
    s = s_.scatter_(index=initial_kpts_idx.unsqueeze(-1), dim=-1, value=1.0).to(torch.float64)
    S = torch.bmm(s, M).to(device, dtype=torch.float64)
    
    W_reformulated_full = calculate_W(S, src_tet_A)
    # W_reformulated_full = calculate_W_prefactor(S, P, L, U)
    # W_reformulated_full = calculate_W_pseudo_np(S, A_inv)
    W_reformulated_surface = torch.gather(W_reformulated_full, 1, sur2tet_idx.unsqueeze(-1).repeat(1,1,args.num_keypoints))
    
    # get source keypoints SV = C
    src_keypoints = torch.bmm(S, src_norm_tet_vertices).to(dtype=torch.float)
    
    # get target keypoints from mapping
    pred_idx = torch.argmax(S, dim=-1)
    tgt_keypoints = torch.gather(tgt_vertices, 1, pred_idx.unsqueeze(-1).repeat(tgt_batch_size,1,3)).to(device)
    def_src_keypoints = tgt_keypoints

    # deformation V = WC
    deformed_vertices = torch.matmul(W_reformulated_full.float(), def_src_keypoints)
    deformed_surface_vertices = torch.matmul(W_reformulated_surface.float(), def_src_keypoints)
    
    # find target surface vertices for calculating loss
    tgt_surface_vertices = torch.gather(tgt_vertices, 1, sur2tet_idx.unsqueeze(-1).repeat(tgt_batch_size,1,3)).to(dtype=torch.float)

    # reconstruction loss
    if args.eval_metric == 'l2':
        l2_loss = torch.nn.MSELoss()
        recon_loss = l2_loss(deformed_surface_vertices, tgt_surface_vertices).item()
    elif args.eval_metric == 'chamfer':
        recon_loss, _ = chamfer_distance(deformed_vertices, tgt_vertices).item()
    else:
        raise NotImplementedError

    with open(os.path.join(args.log_dir, "fps_loss.txt"), "w") as FPS_loss:
        for j in range(tgt_batch_size):
            FPS_loss.write(str(l2_loss(deformed_surface_vertices[j], tgt_surface_vertices[j]).item())+ '\n')
        FPS_loss.write(f'mean FPS loss: {recon_loss}')


    initial_fps_loss = recon_loss
    best_loss = recon_loss
    best_S = None
    curr_kpts_idx = initial_kpts_idx.clone()

    start_time = datetime.datetime.now()
    print(f"Partition Search start at {start_time}")
    #####################################################################################################
    for i in range(args.num_keypoints):
        best_partition_loss = best_loss
        best_partition_idx = None
        # 1. Find the best partition of the source tetmesh that minimizes the loss
        for k in range(args.num_keypoints):
            # replace and check if loss decreases
            partition_points_idx = np.where(tet2FPS_idx == k)[0]
            replaced_idx = curr_kpts_idx.clone()
            replaced_idx[:, i] = np.random.choice(partition_points_idx, 1)[0]

            # check whether current vertex index was already previously selected
            if torch.sum(replaced_idx[:, i] == curr_kpts_idx) > 0:
                continue

            # deform and calculate loss
            s_ = torch.zeros(1, args.num_keypoints, args.num_candidates).to(device, dtype=torch.float64)
            s = s_.scatter_(index=replaced_idx.unsqueeze(-1), dim=-1, value=1.0).to(torch.float64)
            S = torch.bmm(s, M).to(device, dtype=torch.float64)
            
            if args.calc_w_method == 'prefactor':
                W_reformulated_full = calculate_W_prefactor(S, P, L, U)
            elif args.calc_w_method == 'pinv':
                W_reformulated_full = calculate_W_pseudo_np(S, A_inv)
            elif args.calc_w_method == 'original':
                W_reformulated_full = original_calculate_W(S, src_tet_A)
            else:
                raise NotImplementedError

            W_reformulated_surface = torch.gather(W_reformulated_full, 1, sur2tet_idx.unsqueeze(-1).repeat(1,1,args.num_keypoints))
            
            # get source keypoints SV = C
            src_keypoints = torch.bmm(S, src_norm_tet_vertices).to(dtype=torch.float)
            
            # get target keypoints from mapping
            pred_idx = torch.argmax(S, dim=-1)
            tgt_keypoints = torch.gather(tgt_vertices, 1, pred_idx.unsqueeze(-1).repeat(tgt_batch_size,1,3)).to(device)
            def_src_keypoints = tgt_keypoints

            # deformation V = WC
            deformed_vertices = torch.matmul(W_reformulated_full.float(), def_src_keypoints)
            deformed_surface_vertices = torch.matmul(W_reformulated_surface.float(), def_src_keypoints)
            
            # find target surface vertices for calculating loss
            tgt_surface_vertices = torch.gather(tgt_vertices, 1, sur2tet_idx.unsqueeze(-1).repeat(tgt_batch_size,1,3)).to(dtype=torch.float)

            # reconstruction loss
            if args.eval_metric == 'l2':
                l2_loss = torch.nn.MSELoss()
                recon_loss = l2_loss(deformed_surface_vertices, tgt_surface_vertices).item()
            elif args.eval_metric == 'chamfer':
                recon_loss, _ = chamfer_distance(deformed_vertices, tgt_vertices).item()
            else:
                raise NotImplementedError
            # print(f"{i}-th Partition loss: {recon_loss}")
            
            if recon_loss <= best_partition_loss:
                best_partition_idx = k
                curr_kpts_idx = replaced_idx
                best_partition_loss = recon_loss
                best_loss = recon_loss
                best_S = S.clone().detach()

        # if no partition was found, use the initial partition
        if best_partition_idx is None:
            best_partition_idx = i
        # log_string(f'The best partition is {best_partition_idx}. loss: {best_partition_loss}')

        # 2. Find the best control point in the best partition
        partition_points = src_norm_tet_vertices[:, tet2FPS_idx == best_partition_idx, :]
        partition_points_idx = np.where(tet2FPS_idx == best_partition_idx)[0]
        for j in range(partition_points.shape[1]):
            # check whether current vertex index was already previously selected
            if torch.sum(partition_points_idx[j]==curr_kpts_idx) > 0:
                continue

            # replace and check if loss decreases
            replaced_idx = curr_kpts_idx.clone()
            replaced_idx[:, i] = partition_points_idx[j]

            # deform and calculate loss
            s_ = torch.zeros(1, args.num_keypoints, args.num_candidates).to(device, dtype=torch.float64)
            s = s_.scatter_(index=replaced_idx.unsqueeze(-1), dim=-1, value=1.0).to(torch.float64)
            S = torch.bmm(s, M).to(device, dtype=torch.float64)
            
            if args.calc_w_method == 'prefactor':
                W_reformulated_full = calculate_W_prefactor(S, P, L, U)
            elif args.calc_w_method == 'pinv':
                W_reformulated_full = calculate_W_pseudo_np(S, A_inv)
            elif args.calc_w_method == 'original':
                W_reformulated_full = original_calculate_W(S, src_tet_A)
            else:
                raise NotImplementedError
            W_reformulated_surface = torch.gather(W_reformulated_full, 1, sur2tet_idx.unsqueeze(-1).repeat(1,1,args.num_keypoints))
            
            # get source keypoints SV = C
            src_keypoints = torch.bmm(S, src_norm_tet_vertices).to(dtype=torch.float)
            
            # get target keypoints from mapping
            pred_idx = torch.argmax(S, dim=-1)
            tgt_keypoints = torch.gather(tgt_vertices, 1, pred_idx.unsqueeze(-1).repeat(tgt_batch_size,1,3)).to(device)
            def_src_keypoints = tgt_keypoints

            # deformation V = WC
            deformed_vertices = torch.matmul(W_reformulated_full.float(), def_src_keypoints)
            deformed_surface_vertices = torch.matmul(W_reformulated_surface.float(), def_src_keypoints)
            
            # find target surface vertices for calculating loss
            tgt_surface_vertices = torch.gather(tgt_vertices, 1, sur2tet_idx.unsqueeze(-1).repeat(tgt_batch_size,1,3)).to(dtype=torch.float)

            # reconstruction loss
            if args.eval_metric == 'l2':
                l2_loss = torch.nn.MSELoss()
                recon_loss = l2_loss(deformed_surface_vertices, tgt_surface_vertices).item()
            elif args.eval_metric == 'chamfer':
                recon_loss, _ = chamfer_distance(deformed_vertices, tgt_vertices).item()
            else:
                raise NotImplementedError
            

            if recon_loss < best_loss:
                # log_string(f"Updated keypoint {i} with {replaced_idx}. loss: {best_loss}")
                curr_kpts_idx = replaced_idx
                best_loss = recon_loss
                best_S = S.clone().detach()
        
        # log_string(f"Done with keypoint {i}")
        now = datetime.datetime.now()
        duration = (now - start_time).total_seconds()
        # log_string(f"> Elapsed time: {duration}s\n")
    #####################################################################################################
    print("================== optimizing done! ==================")
    pred_idx = torch.argmax(best_S, dim=-1)        
    S = torch.zeros((1, args.num_keypoints, num_tet_vertices), dtype=torch.float64).to(device)
    S = S.scatter_(index=pred_idx.unsqueeze(-1), dim=-1, value=1.0).to(torch.float64)
    
    if args.calc_w_method == 'prefactor':
        W_reformulated_full = calculate_W_prefactor(S, P, L, U)
    elif args.calc_w_method == 'pinv':
        W_reformulated_full = calculate_W_pseudo_np(S, A_inv)
    elif args.calc_w_method == 'original':
        W_reformulated_full = original_calculate_W(S, src_tet_A)
    else:
        raise NotImplementedError
    W_reformulated_surface = torch.gather(W_reformulated_full, 1, sur2tet_idx.unsqueeze(-1).repeat(1,1,args.num_keypoints))
    
    src_keypoints = torch.bmm(S, src_norm_tet_vertices).to(dtype=torch.float)
    tgt_keypoints = torch.gather(tgt_vertices, 1, pred_idx.unsqueeze(-1).repeat(tgt_batch_size,1,3)).to(device)
    def_src_keypoints = tgt_keypoints

    deformed_vertices = torch.matmul(W_reformulated_full.float(), def_src_keypoints)
    deformed_surface_vertices = torch.matmul(W_reformulated_surface.float(), def_src_keypoints)
    tgt_surface_vertices = torch.gather(tgt_vertices, 1, sur2tet_idx.unsqueeze(-1).repeat(tgt_batch_size,1,3)).to(dtype=torch.float)

    l2_loss = torch.nn.MSELoss()
    our_loss = l2_loss(deformed_surface_vertices, tgt_surface_vertices).item()

    for j in range(tgt_batch_size):
        LOG_loss.write(str(l2_loss(deformed_surface_vertices[j], tgt_surface_vertices[j]).item())+ '\n')
    
    print("Initial keypoints indices")
    print(initial_kpts_idx)
    LOG_loss.write(f'Initial keypoints indices: {initial_kpts_idx}\n')
    print()
    print("Final keypoints indices")
    print(curr_kpts_idx)
    LOG_loss.write(f'Final keypoints indices: {curr_kpts_idx}\n')

    print(f'Initial FPS l2 loss: {initial_fps_loss}\n')
    LOG_loss.write(f'Initial FPS l2 loss: {initial_fps_loss}\n')

    final_loss = best_loss
    print(f'Partition l2 loss: {best_loss}\n')
    LOG_loss.write(f'Partition l2 loss: {best_loss}\n')
    
    now = datetime.datetime.now()
    duration = (now - start_time).total_seconds()
    print(f"> Total Partition deformation time: {duration}s |")
    LOG_loss.write(f"> Total Partition deformation time: {duration}s |")
    LOG_loss.close()
    LOG.close()

    # visualize
    # for j in range(tgt_batch_size):
    #     curr_src_vertices = src_norm_tet_vertices[0].detach().to("cpu").numpy()
    #     curr_src_faces = src_norm_tet_faces[0].detach().to("cpu").numpy()
    #     curr_src_keypoints = src_keypoints[0].detach().to("cpu").numpy()
    #     curr_tgt_vertices = tgt_vertices[j].detach().to("cpu").numpy()
    #     curr_tgt_faces = tgt_faces[j].detach().to("cpu").numpy()
    #     curr_tgt_keypoints = tgt_keypoints[j].detach().to("cpu").numpy()
    #     curr_def_vertices = deformed_vertices[j].detach().to("cpu").numpy()
    #     curr_def_keypoints = def_src_keypoints[j].detach().to("cpu").numpy()
    #     visualize_source_front_and_back(f"heuristic{j}", args.dump_dir, curr_src_vertices, curr_src_faces, curr_src_keypoints, curr_tgt_vertices, curr_tgt_faces, curr_tgt_keypoints, curr_def_vertices, curr_def_keypoints, f1, g1)
    
    #     curr_src_surface_vertices = torch.gather(src_norm_tet_vertices, 1, sur2tet_idx.unsqueeze(-1).repeat(1,1,3))
    #     curr_src_surface_vertices = curr_src_surface_vertices[0].detach().to("cpu").numpy()
    #     curr_src_surface_faces = src_surface_faces[0].detach().to("cpu").numpy()
    #     curr_tgt_surface_vertices = tgt_surface_vertices[j].detach().to("cpu").numpy()
    #     curr_def_surface_vertices = deformed_surface_vertices[j].detach().to("cpu").numpy()
    #     visualize_source_front_and_back(f"heuristic_surface{j}", args.dump_dir, curr_src_surface_vertices, curr_src_surface_faces, curr_src_keypoints, curr_tgt_surface_vertices, curr_src_surface_faces, curr_tgt_keypoints, curr_def_surface_vertices, curr_def_keypoints, f1, g1)

    #     vertex2vertex_error = []
    #     for i in range(deformed_surface_vertices.shape[1]):
    #         vertex2vertex_error.append(l2_loss(deformed_surface_vertices[j][i], tgt_surface_vertices[j][i]).item())
    #     vertex2vertex_error = np.array(vertex2vertex_error)
    #     np.savetxt(f'{args.dump_dir}/heuristic_surface{j}_error.txt', vertex2vertex_error, fmt='%f')

    print(f'===================== partition writing mesh done!\n')
