import numpy as np

def apply_solid_damping(vx, vy, sdf, thickness=2.0):
    mask = sdf < thickness
    damping = np.clip((sdf / thickness), 0, 1)
    vx[mask] *= damping[mask]
    vy[mask] *= damping[mask]
    return vx, vy

def project(vx, vy, p, fluid_mask, sdf, iterations=80):
    div = np.zeros_like(p)
    div[1:-1,1:-1] = (
        vx[1:-1,2:] - vx[1:-1,0:-2] +
        vy[2:,1:-1] - vy[0:-2,1:-1]
    ) * 0.5

    div[~fluid_mask] = 0
    p[:] = 0

    for _ in range(iterations):
        p[1:-1,1:-1] = (
            p[1:-1,2:] + p[1:-1,0:-2] +
            p[2:,1:-1] + p[0:-2,1:-1] - div[1:-1,1:-1]
        ) * 0.25
        p[~fluid_mask] = p[~fluid_mask]

    vx[1:-1,1:-1] -= 0.5 * (p[1:-1,2:] - p[1:-1,0:-2])
    vy[1:-1,1:-1] -= 0.5 * (p[2:,1:-1] - p[0:-2,1:-1])

    vx, vy = apply_solid_damping(vx, vy, sdf)
    return vx, vy

def particles(px, py, particle_active, particle_intake, intake_positions, num_particles):
    for intake, (x, y) in enumerate(intake_positions):
        for i in range(num_particles):
            if not particle_active[i]:
                px[i] = x + 0.5
                py[i] = y
                particle_active[i] = True
                particle_intake[i] = intake
                break
