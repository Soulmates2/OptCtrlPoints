import os
import numpy as np
import torch
import scipy.sparse
from pytorch3d.io import load_obj, load_ply


class CenterAlign(object):
    def __init__(self):
        super().__init__()

    @staticmethod
    def normalize(pcl, center=None, scale=None):
        if center is None:
            p_max = pcl.max(dim=0, keepdim=True)[0]
            p_min = pcl.min(dim=0, keepdim=True)[0]
            center = (p_max + p_min) / 2    # (1, 3)
        pcl = pcl - center
        if scale is None:
            scale = (pcl ** 2).sum(dim=1, keepdim=True).sqrt().max(dim=0, keepdim=True)[0]  # (1, 1)
        pcl = pcl / scale
        return pcl, center, scale

    def __call__(self, data):
        return self.normalize(data)


class SourceData:
    def __init__(self, root, device):
        self.root = root
        src_tet_vertices, src_tet_faces, _ = load_obj(os.path.join(self.root, "norm_tet_full_appended.obj"))
        src_surface_vertices, src_surface_faces, _ = load_obj(os.path.join(self.root, "tet_full.mesh__sf.obj"))
        tet2sur_idx = np.loadtxt(os.path.join(self.root, "tet2sur_idx.npy"), skiprows=1, dtype=int)
        A = scipy.sparse.load_npz(os.path.join(self.root, "A_sparse_double.npz")).toarray()
        src_scale_factor = np.load(os.path.join(self.root, "src_scale.npy"))

        src_tet_vertices = src_tet_vertices.unsqueeze(0).to(device, dtype=torch.float64)
        src_tet_faces = src_tet_faces.verts_idx.unsqueeze(0).to(device, dtype=torch.int64)
        src_surface_vertices = src_surface_vertices.unsqueeze(0).to(device, dtype=torch.float64)
        src_surface_faces = src_surface_faces.verts_idx.unsqueeze(0).to(device, dtype=torch.int64)
        sur2tet_idx = torch.from_numpy(tet2sur_idx).unsqueeze(0).to(device, dtype=torch.int64)
        src_A = torch.from_numpy(A).unsqueeze(0).to(device, dtype=torch.float64)
        
        self.src_tet_vertices = src_tet_vertices
        self.src_tet_faces = src_tet_faces
        self.src_surface_vertices = src_surface_vertices
        self.src_surface_faces = src_surface_faces
        self.sur2tet_idx = sur2tet_idx
        self.src_A = src_A
        self.src_scale_factor = src_scale_factor
        
    def get_item(self):
        return self.src_tet_vertices, self.src_tet_faces, self.src_surface_vertices, self.src_surface_faces, self.src_A, self.sur2tet_idx, self.src_scale_factor



class TargetData:
    def __init__(self, root, device):
        self.root = root
        filelist = sorted(os.listdir(root), key=lambda x: int(os.path.splitext(x)[0]))
        filelist = np.array(filelist)

        tgt_vertices = []
        tgt_faces = []
        for fn in filelist:
            if os.path.splitext(os.path.join(root, fn))[1] == '.ply':
                verts, faces = load_ply(os.path.join(root, fn))
                tgt_vertices.append(verts)
                tgt_faces.append(faces)
            if os.path.splitext(os.path.join(root, fn))[1] == '.obj':
                verts, faces, _ = load_obj(os.path.join(root, fn))
                tgt_vertices.append(verts)
                tgt_faces.append(faces.verts_idx)
        
        tgt_vertices = torch.stack(tgt_vertices).to(device, dtype=torch.float)
        tgt_faces = torch.stack(tgt_faces).to(device, dtype=torch.int64)
        
        self.vertices = tgt_vertices
        self.faces = tgt_faces

    def get_item(self):
        return self.vertices, self.faces
