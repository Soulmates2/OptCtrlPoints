import torch


def triangle_area(v, f):
    A = torch.gather(v, 0, f[:, 0].unsqueeze(-1).expand(-1, 3))
    B = torch.gather(v, 0, f[:, 1].unsqueeze(-1).expand(-1, 3))
    C = torch.gather(v, 0, f[:, 2].unsqueeze(-1).expand(-1, 3))
    normal = torch.cross((B - A), (C - A), dim = -1)
    return normal.norm(p = 2, dim = -1)


def normal_loss(src_meshes, def_meshes):
    N = len(src_meshes)
    src_v = src_meshes.verts_packed()  
    def_v = def_meshes.verts_packed() 
    src_f = src_meshes.faces_packed()
    
    num_faces_per_mesh = src_meshes.num_faces_per_mesh()  
    faces_packed_idx = src_meshes.faces_packed_to_mesh_idx() 
    w = num_faces_per_mesh.gather(0, faces_packed_idx)  
    w = 1.0 / w.float()
    
    cos = torch.nn.CosineSimilarity(dim = -1, eps = 1e-18)
    src_A = torch.gather(src_v, 0, src_f[:, 0].unsqueeze(-1).expand(-1, 3))
    src_B = torch.gather(src_v, 0, src_f[:, 1].unsqueeze(-1).expand(-1, 3))
    src_C = torch.gather(src_v, 0, src_f[:, 2].unsqueeze(-1).expand(-1, 3))
    def_A = torch.gather(def_v, 0, src_f[:, 0].unsqueeze(-1).expand(-1, 3))
    def_B = torch.gather(def_v, 0, src_f[:, 1].unsqueeze(-1).expand(-1, 3))
    def_C = torch.gather(def_v, 0, src_f[:, 2].unsqueeze(-1).expand(-1, 3))
    src_normal = torch.cross((src_B - src_A), (src_C - src_A))
    def_normal = torch.cross((def_B - def_A), (def_C - def_A))
    with torch.no_grad():
        weight = triangle_area(src_v, src_f) + triangle_area(def_v, src_f)
        weight = (weight / weight.mean() + 1) / 2
    loss = (1 - cos(src_normal, def_normal)) * weight
    loss = loss * w
    return loss.sum() / N


def laplacian_loss(meshes, def_meshes, method: str = "cot", batch_reduction=True):
    if meshes.isempty():
        return torch.tensor(
            [0.0], dtype=torch.float32, device=meshes.device, requires_grad=True
        )

    N = len(meshes)
    verts_packed = meshes.verts_packed()  # (sum(V_n), 3)
    def_verts_packed = def_meshes.verts_packed()  # (sum(V_n), 3)
    num_verts_per_mesh = meshes.num_verts_per_mesh()  # (N,)
    verts_packed_idx = meshes.verts_packed_to_mesh_idx()  # (sum(V_n),)
    weights = num_verts_per_mesh.gather(0, verts_packed_idx)  # (sum(V_n),)
    weights = 1.0 / weights.float()

    # We don't want to backprop through the computation of the Laplacian;
    # just treat it as a magic constant matrix that is used to transform
    # verts into normals
    with torch.no_grad():
        if method == "uniform":
            L = meshes.laplacian_packed()
        elif method in ["cot", "cotcurv"]:
            L, inv_areas = laplacian_cot(meshes)
            if method == "cot":
                norm_w = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
                idx = norm_w > 0
                norm_w[idx] = 1.0 / norm_w[idx]
            else:
                norm_w = 0.25 * inv_areas
        else:
            raise ValueError("Method should be one of {uniform, cot, cotcurv}")

    v = def_verts_packed - verts_packed
    if method == "uniform":
        loss = L.mm(verts_packed)
    elif method == "cot":
        loss = L.mm(v) * norm_w - v
    elif method == "cotcurv":
        loss = (L.mm(verts_packed) - verts_packed) * norm_w
    loss = loss.norm(dim=1)

    loss = loss * weights
    if batch_reduction:
    	return loss.sum() / N
    else:
    	return loss


def laplacian_cot(meshes):
    """
    Returns the Laplacian matrix with cotangent weights and the inverse of the
    face areas.
    Args:
        meshes: Meshes object with a batch of meshes.
    Returns:
        2-element tuple containing
        - **L**: FloatTensor of shape (V,V) for the Laplacian matrix (V = sum(V_n))
            Here, L[i, j] = cot a_ij + cot b_ij iff (i, j) is an edge in meshes.
            See the description above for more clarity.
        - **inv_areas**: FloatTensor of shape (V,) containing the inverse of sum of
            face areas containing each vertex
    """
    verts_packed = meshes.verts_packed()  # (sum(V_n), 3)
    faces_packed = meshes.faces_packed()  # (sum(F_n), 3)
    # V = sum(V_n), F = sum(F_n)
    V, F = verts_packed.shape[0], faces_packed.shape[0]

    face_verts = verts_packed[faces_packed]
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

    # Side lengths of each triangle, of shape (sum(F_n),)
    # A is the side opposite v1, B is opposite v2, and C is opposite v3
    A = (v1 - v2).norm(dim=1)
    B = (v0 - v2).norm(dim=1)
    C = (v0 - v1).norm(dim=1)

    # Area of each triangle (with Heron's formula); shape is (sum(F_n),)
    s = 0.5 * (A + B + C)
    # note that the area can be negative (close to 0) causing nans after sqrt()
    # we clip it to a small positive value
    area = (s * (s - A) * (s - B) * (s - C)).clamp_(min=1e-12).sqrt()

    # Compute cotangents of angles, of shape (sum(F_n), 3)
    A2, B2, C2 = A * A, B * B, C * C
    cota = (B2 + C2 - A2) / area
    cotb = (A2 + C2 - B2) / area
    cotc = (A2 + B2 - C2) / area
    cot = torch.stack([cota, cotb, cotc], dim=1)
    cot /= 4.0

    # Construct a sparse matrix by basically doing:
    # L[v1, v2] = cota
    # L[v2, v0] = cotb
    # L[v0, v1] = cotc
    ii = faces_packed[:, [1, 2, 0]]
    jj = faces_packed[:, [2, 0, 1]]
    idx = torch.stack([ii, jj], dim=0).view(2, F * 3)
    L = torch.sparse.FloatTensor(idx, cot.view(-1), (V, V))

    # Make it symmetric; this means we are also setting
    # L[v2, v1] = cota
    # L[v0, v2] = cotb
    # L[v1, v0] = cotc
    L += L.t()

    # For each vertex, compute the sum of areas for triangles containing it.
    idx = faces_packed.view(-1)
    inv_areas = torch.zeros(V, dtype=torch.float32, device=meshes.device)
    val = torch.stack([area] * 3, dim=1).view(-1)
    inv_areas.scatter_add_(0, idx, val)
    idx = inv_areas > 0
    inv_areas[idx] = 1.0 / inv_areas[idx]
    inv_areas = inv_areas.view(-1, 1)

    return L, inv_areas
