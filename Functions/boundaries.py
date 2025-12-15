import numpy as np
from Functions.visuals import apply_solid_damping

def enforce_solid_boundary(vx, vy, mask, sdf):
    vx, vy = apply_solid_damping(vx, vy, sdf)
    for _ in range(2):
        vx[:, 1:] = np.where(~mask[:, 1:] & mask[:, :-1], 0, vx[:, 1:])
        vx[:, :-1] = np.where(~mask[:, :-1] & mask[:, 1:], 0, vx[:, :-1])
        vy[1:, :] = np.where(~mask[1:, :] & mask[:-1, :], 0, vy[1:, :])
        vy[:-1, :] = np.where(~mask[:-1, :] & mask[1:, :], 0, vy[:-1, :])
    return vx, vy

def enforceNoPenetration(vx, vy, sdf, nx_sdf, ny_sdf):
    near = (sdf < 1.5) & (sdf > -1.5)
    vn = vx * nx_sdf + vy * ny_sdf
    vx[near] -= vn[near] * nx_sdf[near]
    vy[near] -= vn[near] * ny_sdf[near]
    return vx, vy
