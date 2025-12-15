import numpy as np
from scipy.ndimage import convolve

def advect(vx, vy, dt, fluid_mask):
    ny, nx = vx.shape
    vx_new = np.zeros_like(vx)
    vy_new = np.zeros_like(vy)

    Y, X = np.mgrid[0:ny, 0:nx]
    Xb = X - vx * dt
    Yb = Y - vy * dt

    Xb = np.clip(Xb, 0, nx-1.001)
    Yb = np.clip(Yb, 0, ny-1.001)

    x0 = Xb.astype(int)
    y0 = Yb.astype(int)
    x1 = np.clip(x0 + 1, 0, nx-1)
    y1 = np.clip(y0 + 1, 0, ny-1)

    sx = Xb - x0
    sy = Yb - y0

    for field, field_new in [(vx, vx_new), (vy, vy_new)]:
        field_new[:] = (
            (1-sx)*(1-sy)*field[y0, x0] +
            sx*(1-sy)*field[y0, x1] +
            (1-sx)*sy*field[y1, x0] +
            sx*sy*field[y1, x1]
        )

    vx_new[~fluid_mask] = 0
    vy_new[~fluid_mask] = 0

    return vx_new, vy_new

def interpolate_velocity(x, y, vx, vy):
    ny, nx = vx.shape
    x = np.clip(x, 0, nx-1.0001)
    y = np.clip(y, 0, ny-1.0001)
    ix0, iy0 = int(x), int(y)
    ix1, iy1 = min(ix0+1, nx-1), min(iy0+1, ny-1)
    fx = x - ix0
    fy = y - iy0

    vx_val = (1-fx)*(1-fy)*vx[iy0, ix0] + fx*(1-fy)*vx[iy0, ix1] + \
             (1-fx)*fy*vx[iy1, ix0] + fx*fy*vx[iy1, ix1]
    vy_val = (1-fx)*(1-fy)*vy[iy0, ix0] + fx*(1-fy)*vy[iy0, ix1] + \
             (1-fx)*fy*vy[iy1, ix0] + fx*fy*vy[iy1, ix1]
    return vx_val, vy_val

def reset(vx, vy, airfoil_mask):
    vx[airfoil_mask == 1] = 0
    vy[airfoil_mask == 1] = 0

def compute_pressure(p, vx, vy, fluid_mask, tol=1e-5):
    rhs = - (np.gradient(vx, axis=1) + np.gradient(vy, axis=0))
    rhs[~fluid_mask] = 0

    for _ in range(200):
        p_new = convolve(p, np.array([[0, 1, 0],[1, 0, 1],[0, 1, 0]])/4, mode='nearest')
        p_new[fluid_mask] += rhs[fluid_mask]/2

        # Neumann boundary conditions
        p[0, :] = p[1, :]
        p[-1, :] = p[-2, :]
        p[:, 0] = p[:, 1]
        p[:, -1] = p[:, -2]

        if np.max(np.abs(p_new - p)) < tol:
            break
        p[fluid_mask] = p_new[fluid_mask]

    p[~fluid_mask] = 0
    return p

def runge_kutta(px, py, vx, vy, dt, num_particles, particle_active, particle_intake, intake_positions, airfoil_mask):
    px_new, py_new = px.copy(), py.copy()
    ny, nx = vx.shape

    for i in range(num_particles):
        if particle_active[i]:
            vx_p, vy_p = interpolate_velocity(px[i], py[i], vx, vy)
            k1x, k1y = vx_p*dt, vy_p*dt
            vx2, vy2 = interpolate_velocity(px[i]+0.5*k1x, py[i]+0.5*k1y, vx, vy)
            k2x, k2y = vx2*dt, vy2*dt
            vx3, vy3 = interpolate_velocity(px[i]+0.5*k2x, py[i]+0.5*k2y, vx, vy)
            k3x, k3y = vx3*dt, vy3*dt
            vx4, vy4 = interpolate_velocity(px[i]+k3x, py[i]+k3y, vx, vy)
            k4x, k4y = vx4*dt, vy4*dt

            px_new[i] += (k1x + 2*k2x + 2*k3x + k4x)/6
            py_new[i] += (k1y + 2*k2y + 2*k3y + k4y)/6

            # Handle collisions with airfoil
            ix, iy = int(px_new[i]), int(py_new[i])
            if 0 <= ix < nx and 0 <= iy < ny and airfoil_mask[iy, ix] == 1:
                intake_idx = particle_intake[i]
                x, y = intake_positions[intake_idx]
                px_new[i] = x + 0.5
                py_new[i] = y

            # Keep inside domain
            px_new[i] = np.clip(px_new[i], 0, nx-1)
            py_new[i] = np.clip(py_new[i], 0, ny-1)

    return px_new, py_new
