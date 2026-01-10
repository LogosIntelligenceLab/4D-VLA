import torch
from torch.nn import functional as F


def unproject(intrinsics, poses, depths):
    """
    Inputs:
        intrinsics: B X V X 3 X 3
        poses: B X V X 4 X 4 (torch.tensor)
        depths: B X V X H X W (torch.tensor)
    
    Outputs:
        world_coords: B X V X H X W X 3
    """
    # (B, V, 336, 336)
    B, V, H, W = depths.shape 
    # (B, V, 1)
    fx, fy, px, py = intrinsics[..., 0, 0][..., None], intrinsics[..., 1, 1][..., None], intrinsics[..., 0, 2][..., None], intrinsics[..., 1, 2][..., None]

    # y = torch.arange(0, H).to(depths.device)
    # x = torch.arange(0, W).to(depths.device)
    y = torch.arange(0, H).to(depths.device)
    x = torch.arange(0, W).to(depths.device)
    y = y + 0.5
    x = x + 0.5
    y = y.to(torch.bfloat16)
    x = x.to(torch.bfloat16)

    y, x = torch.meshgrid(y, x)

    x = x[None, None].repeat(B, V, 1, 1).flatten(2)  # (B, V, H*W)
    y = y[None, None].repeat(B, V, 1, 1).flatten(2)  # (B, V, H*W)
    z = depths.flatten(2)  # (B, V, H*W)
    x = (x - px) * z / fx
    y = (y - py) * z / fy
    cam_coords = torch.stack([
        x, y, z, torch.ones_like(x)
    ], -1)

    world_coords = (poses @ cam_coords.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
    world_coords = world_coords[..., :3] / world_coords[..., 3][..., None]

    world_coords = world_coords.reshape(B, V, H, W, 3)

    return world_coords

def backproject_depth(depths, poses, intrinsics=None):
    B, V, H, W = depths.shape
    if intrinsics.shape[1] != V:
        intrinsics = intrinsics.repeat(1, V, 1, 1)
    xyz = unproject(intrinsics, poses, depths)
    return xyz  # (B X V X H X W X 3)

def interpolate_depth(depth, feature_shape, depth_mask, intrinsic, method="nearest"):
    B, V, H, W = depth.shape
    h, w = feature_shape[2:]

    depth_interpolate = torch.nn.functional.interpolate(
        depth.reshape(B*V, 1, H, W), size=(h, w),
        mode='bilinear', align_corners=False).reshape(B, V, h, w)
    
    depth_mask_interpolate = torch.nn.functional.interpolate(
        depth_mask.float().reshape(B*V, 1, H, W), size=(h, w),
        mode='bilinear', align_corners=False).reshape(B, V, h, w).to(dtype=depth_interpolate.dtype)

    depth_mask = depth_mask_interpolate > 0.1
    depth_interpolate[depth_mask] = depth_interpolate[depth_mask] / depth_mask_interpolate[depth_mask]

    if intrinsic.shape[1] != V:
        intrinsic = intrinsic.repeat(1, V, 1, 1)
    scale_x = w / 448
    scale_y = h / 448
    # scale_x = w / W
    # scale_y = h / H
    fx = intrinsic[:, :, 0, 0]  # (B, V)
    fy = intrinsic[:, :, 1, 1]
    cx = intrinsic[:, :, 0, 2]
    cy = intrinsic[:, :, 1, 2]

    new_fx = fx * scale_x
    new_fy = fy * scale_y
    new_cx = cx * scale_x
    new_cy = cy * scale_y

    new_intrinsic = torch.stack([
        new_fx, torch.zeros_like(new_fx), new_cx,
        torch.zeros_like(new_fy), new_fy, new_cy,
        torch.zeros_like(new_fx), torch.zeros_like(new_fy), torch.ones_like(new_fy)
    ], dim=-1).reshape(B, V, 3, 3).to(depth.device)

    return depth_interpolate, depth_mask, new_intrinsic

def backprojector_dataloader(
    multi_scale_features, depths, depth_mask, poses,
    intrinsics=None, method='nearest', 
    padding=None):
    """
    Inputs:
        multi_scale_features: list
            [B*V, 1024, 24, 24], [B*V, 1024, 48, 48], [B*V, 1024, 96, 96]
        depths: tensor [B, 5, 336, 336]
        poses: tensor [B, 5, 4, 4]
        intrinsics: tensor [B, 5, 4, 4]

    Outputs:
        list: []
            B, V, H, W, 3
    """
    new_depth_interpolate, new_depth_mask, new_intrinsic = interpolate_depth(
        depths, multi_scale_features[0].shape, depth_mask, intrinsics
    )

    # (B, V, H, W, 3)
    new_xyz = backproject_depth(
        new_depth_interpolate, poses, new_intrinsic)

    return new_xyz, None, new_depth_mask

def voxelize(xyz, voxel_size=0.28):
    """
    Inputs: 
        xyz: list of tensors [B, V, H, W, 3]
        voxel_size: voxel size

    Outputs:
        N=V*H*W
        p2v: tensors [B, N]
    """
    B, V, H, W, _ = xyz.shape
    xyz = xyz.reshape(B, V*H*W, 3)
    p2v = voxelization(xyz, voxel_size)
    return p2v


def ravel_hash_vec(arr):
    """
    Ravel the coordinates after subtracting the min coordinates.
    """
    assert len(arr.shape) == 3
    arr -= arr.min(1, keepdims=True)[0].to(torch.long)
    arr_max = arr.max(1, keepdims=True)[0].to(torch.long) + 1

    keys = torch.zeros(arr.shape[0], arr.shape[1], dtype=torch.long).to(arr.device)

    # Fortran style indexing
    for j in range(arr.shape[2] - 1):
        keys += arr[..., j]
        keys *= arr_max[..., j + 1]
    keys += arr[..., -1]
    return keys

def voxelization(xyz, voxel_size):
    """
    Inputs:
        xyz: tensor [B, N, 3]
        voxel_size: float
    Outputs: 
        point_to_voxel_all: tensor [B, N], is the mapping from original point cloud to voxel
    """
    B, N, _ = xyz.shape
    xyz = xyz / voxel_size
    xyz = torch.round(xyz).long()
    xyz = xyz - xyz.min(1, keepdim=True)[0]

    keys = ravel_hash_vec(xyz)

    point_to_voxel = torch.stack(
        [torch.unique(keys[b], return_inverse=True)[1] for b in range(B)], 0)
    return point_to_voxel


def voxel_map_to_source(voxel_map, poin2voxel):
    """
    Input:
        voxel_map (B, N1, C)
        point2voxel (B, N)
    Output:
        src_new (B, N, C)
    """
    bs, n, c = voxel_map.shape
    src_new = torch.stack([voxel_map[i, poin2voxel[i]] for i in range(bs)])
    return src_new