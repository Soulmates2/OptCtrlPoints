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
import matplotlib.pyplot as plt

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


def make_output_dump_dir(out_path):
    if not os.path.exists(out_path): os.mkdir(out_path)
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
    dist, src2dst_idx = nearest_neighbor(src_vertices.squeeze().detach().to("cpu").numpy(), dst_vertices.squeeze().detach().to("cpu").numpy())
    return src2dst_idx    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument('--source_data_path', type=str, default='/scratch/NeuralKeypoints/SMAL/smal_source_tetmesh/fox', help='dataset path')
    parser.add_argument('--target_data_path', type=str, default='/scratch/NeuralKeypoints/SMAL/smal_target_tetmesh/fox', help='dataset path')
    parser.add_argument('--log', type=eval, default=True, choices=[True, False], help='logging train result')
    parser.add_argument('--log_dir', type=str, default='log/SMAL', help='log dir path')
    parser.add_argument('--dump_dir', type=str, default='dump/SMAL', help='dump dir path')

    # model parameter
    parser.add_argument('--num_keypoints', type=int, default=16, help='the number key points') 
    parser.add_argument('--num_candidates', type=int, default=5000, help='farthest point sampling of source point cloud')
    parser.add_argument('--init_method', default='fps', choices=['fps'], help='initialization methods (default: %(default)s)') 

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
    os.system(f'cp {os.path.abspath(__file__)} {args.log_dir}')
    make_output_dump_dir(args.dump_dir)
    f1, g1 = make_output_render_file(args.dump_dir)


    # load source data and target data
    source_dataset = SourceData(args.source_data_path, device)
    src_norm_tet_vertices, src_norm_tet_faces, src_surface_vertices, src_surface_faces, src_tet_A, sur2tet_idx, _ = source_dataset.get_item()
    print("source tet vertices shape:", src_norm_tet_vertices.shape)
    print("source surface vertices shape:", src_surface_vertices.shape)

    ### Pre-factor ##
    A_LU, pivots = torch.lu(src_tet_A)
    P, L, U = torch.lu_unpack(A_LU, pivots)

    print("P shape:", P.shape)
    print("L shape:", L.shape)
    print("U shape:", U.shape)
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
    # recon_loss, _ = chamfer_distance(deformed_vertices, tgt_vertices)
    l2_loss = torch.nn.MSELoss()
    recon_loss = l2_loss(deformed_surface_vertices, tgt_surface_vertices).item()
    print("Initial FPS loss:", recon_loss)

    with open(os.path.join(args.log_dir, "fps_loss.txt"), "w") as FPS_loss:
        for j in range(tgt_batch_size):
            FPS_loss.write(str(l2_loss(deformed_surface_vertices[j], tgt_surface_vertices[j]).item())+ '\n')
        FPS_loss.write(f'mean FPS loss: {recon_loss}')

    # for j in range(tgt_batch_size):
    #     curr_src_vertices = src_norm_tet_vertices[0].detach().to("cpu").numpy()
    #     curr_src_faces = src_norm_tet_faces[0].detach().to("cpu").numpy()
    #     curr_src_keypoints = src_keypoints[0].detach().to("cpu").numpy()
    #     curr_tgt_vertices = tgt_vertices[j].detach().to("cpu").numpy()
    #     curr_tgt_faces = tgt_faces[j].detach().to("cpu").numpy()
    #     curr_tgt_keypoints = tgt_keypoints[j].detach().to("cpu").numpy()
    #     curr_def_vertices = deformed_vertices[j].detach().to("cpu").numpy()
    #     curr_def_keypoints = def_src_keypoints[j].detach().to("cpu").numpy()
    #     visualize_source_front_and_back(f"FPS{j}", args.dump_dir, curr_src_vertices, curr_src_faces, curr_src_keypoints, curr_tgt_vertices, curr_tgt_faces, curr_tgt_keypoints, curr_def_vertices, curr_def_keypoints, f1, g1)
    
    #     curr_src_surface_vertices = torch.gather(src_norm_tet_vertices, 1, sur2tet_idx.unsqueeze(-1).repeat(1,1,3))
    #     curr_src_surface_vertices = curr_src_surface_vertices[0].detach().to("cpu").numpy()
    #     curr_src_surface_faces = src_surface_faces[0].detach().to("cpu").numpy()
    #     curr_tgt_surface_vertices = tgt_surface_vertices[j].detach().to("cpu").numpy()
    #     curr_def_surface_vertices = deformed_surface_vertices[j].detach().to("cpu").numpy()
    #     visualize_source_front_and_back(f"FPS_surface{j}", args.dump_dir, curr_src_surface_vertices, curr_src_surface_faces, curr_src_keypoints, curr_tgt_surface_vertices, curr_src_surface_faces, curr_tgt_keypoints, curr_def_surface_vertices, curr_def_keypoints, f1, g1)

    #     vertex2vertex_error = []
    #     for i in range(deformed_surface_vertices.shape[1]):
    #         vertex2vertex_error.append(l2_loss(deformed_surface_vertices[j][i], tgt_surface_vertices[j][i]).item())
    #     vertex2vertex_error = np.array(vertex2vertex_error)
    #     np.savetxt(f'{args.dump_dir}/FPS_surface{j}_error.txt', vertex2vertex_error, fmt='%f')

    print(f'===================== FPS writing mesh done!\n')
