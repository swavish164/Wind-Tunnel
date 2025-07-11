import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from scipy.ndimage import convolve

# Grid and simulation parameters
nx, ny = 500, 200    
dx, dy = 1.0, 1.0  
nt = 300          
dt = 0.02         
viscosity = 0.1    

# Load and process image
image = cv.imread('f1CarSide.jpg')
bw = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
blurred = cv.GaussianBlur(bw, (5, 5), 0)
edges = cv.Canny(blurred, threshold1=30, threshold2=100)
kernel = np.ones((5, 5), np.uint8)
closed_edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)
contours, _ = cv.findContours(closed_edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

filled_image = np.zeros_like(bw)
cv.drawContours(filled_image, contours, -1, 255, thickness=cv.FILLED)

def imageResize(image, max_height, max_width):
    height, width = image.shape[:2]
    scale = min(max_height / height, max_width / width*0.75)
    new_size = (int(width * scale), int(height * scale))
    return cv.resize(image, new_size)

# Resize and floor-align the car mask
max_airfoil_height, max_airfoil_width = int(0.5 * ny), int(0.8 * nx)
resized = imageResize(filled_image, max_airfoil_height, max_airfoil_width)
resized = (resized > 0).astype(np.uint8)

# Flip resized mask vertically (so top of resized becomes bottom)
resized_flipped = np.flipud(resized)

# Place at top-left corner (y=0)
top_left_y = 0

airfoil_mask = np.zeros((ny, nx), dtype=np.uint8)
top_left_x = int(0.1 * nx)
airfoil_mask[top_left_y:top_left_y + resized.shape[0], top_left_x:top_left_x + resized.shape[1]] = resized_flipped
fluid_mask = (airfoil_mask == 0)

# Initialize fields
vx = np.ones((ny, nx)) *4
vy = np.zeros((ny, nx))
p = np.zeros((ny, nx))

num_particles = 500
particle_spawn_rate = 10
num_intakes = 15
intake_positions = []
space = int(ny / num_intakes)
for i in range (1,num_intakes+2):
    intake_positions.append((0,(i*space)-1))
    
px = np.zeros(num_particles)
py = np.zeros(num_particles)
particle_intake = np.zeros(num_particles, dtype=int)
particle_active = np.zeros(num_particles, dtype = bool)

# Distribute particles in straight lines at each intake

def particles():
    for intake , (x,y) in enumerate(intake_positions):
        for i in range(num_particles):
            if not particle_active[i]:
                px[i] = x + 0.5
                py[i] = y
                particle_active[i] = True
                particle_intake[i] = intake
                break


def compute_velocity(vx, vy, p, dt, viscosity):
    fluid_mask = (airfoil_mask == 0)
    dpdx, dpdy = np.gradient(p)
    vx -= dt * dpdx
    vy -= dt * dpdy
    laplacian_vx = convolve(vx, np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]), mode='nearest', cval=0)
    laplacian_vy = convolve(vy, np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]) , mode='nearest', cval=0)
    vx[~fluid_mask] = 0
    vy[~fluid_mask] = 0
    vx += viscosity * laplacian_vx * dt
    vy += viscosity * laplacian_vy * dt
    vx = np.clip(vx, -10,10)
    vy = np.clip(vy,-5,5)
    return vx, vy

def divergence(vx, vy,mask):
    dx = np.gradient(vx, axis=1)
    dy = np.gradient(vy, axis=0)
    div = dy + dx
    div[~mask] = 0
    return div

def compute_pressure(p, vx, vy, tol=1e-5):
    rhs = -divergence(vx, vy, fluid_mask)
    for _ in range(200):
        p_new = convolve(p, np.array([[0, 1, 0],
                                      [1, 0, 1],
                                      [0, 1, 0]])/4, mode='nearest')
        p_new[fluid_mask] += rhs[fluid_mask] / 2

        # Neumann boundary conditions (zero normal gradient)
        p[0, :] = p[1, :]
        p[-1, :] = p[-2, :]
        p[:, 0] = p[:, 1]
        p[:, -1] = p[:, -2]

        if np.max(np.abs(p_new - p)) < tol:
            break

        p[fluid_mask] = p_new[fluid_mask]

    # Zero pressure inside the car (optional but cleaner)
    p[~fluid_mask] = 0

    return p


def interpolate_velocity(x, y, vx, vy):
    # Clamp coordinates to stay within bounds
    x = np.clip(x, 0, nx-1.0001)
    y = np.clip(y, 0, ny-1.0001)
    # Floor coordinates for interpolation
    ix0, iy0 = int(x), int(y)
    ix1, iy1 = min(ix0 + 1, nx-1), min(iy0 + 1, ny-1)
    # Bilinear interpolation weights
    fx = x - ix0
    fy = y - iy0
    # Interpolate velocity
    vx_val = (1-fx)*(1-fy)*vx[iy0, ix0] + fx*(1-fy)*vx[iy0, ix1] + \
             (1-fx)*fy*vx[iy1, ix0] + fx*fy*vx[iy1, ix1]
    
    vy_val = (1-fx)*(1-fy)*vy[iy0, ix0] + fx*(1-fy)*vy[iy0, ix1] + \
             (1-fx)*fy*vy[iy1, ix0] + fx*fy*vy[iy1, ix1]
    
    return vx_val, vy_val

def runge_kutta(px, py, vx, vy, dt):
    px_new, py_new = px.copy(), py.copy()
    
    for i in range(num_particles):
        if particle_active[i]:
            # Get velocity at particle position
            vx_p, vy_p = interpolate_velocity(px[i], py[i], vx, vy)
            
            # RK4 integration
            k1x, k1y = vx_p * dt, vy_p * dt
            vx2, vy2 = interpolate_velocity(px[i] + 0.5*k1x, py[i] + 0.5*k1y, vx, vy)
            k2x, k2y = vx2 * dt, vy2 * dt
            vx3, vy3 = interpolate_velocity(px[i] + 0.5*k2x, py[i] + 0.5*k2y, vx, vy)
            k3x, k3y = vx3 * dt, vy3 * dt
            vx4, vy4 = interpolate_velocity(px[i] + k3x, py[i] + k3y, vx, vy)
            k4x, k4y = vx4 * dt, vy4 * dt
            
            px_new[i] += (k1x + 2*k2x + 2*k3x + k4x) / 6
            py_new[i] += (k1y + 2*k2y + 2*k3y + k4y) / 6
            
            # If particle hits the car, make it swirl around
            if (0 <= px_new[i] < nx and 0 <= py_new[i] < ny and 
                airfoil_mask[int(py_new[i]), int(px_new[i])] == 1):
                # Push particle back with some randomness
                px_new[i] = px[i] - 0.3*vx_p*dt
                py_new[i] = py[i] - 0.3*vy_p*dt + np.random.uniform(-0.5, 0.5)
            ix = int(px_new[i])
            iy = int(py_new[i])
            if 0 <= ix < nx and 0 <= iy < ny and airfoil_mask[iy, ix] == 1:
            #if airfoil_mask[int(py_new[i]), int(px_new[i])] == 1:
    # Reset particle to intake immediately
                intake_idx = particle_intake[i]
                x, y = intake_positions[intake_idx]
                px_new[i] = x + 0.5
                py_new[i] = y
            
            # If particle exits right boundary, recycle it to its intake
            if px_new[i] >= nx + 20:
                intake_idx = particle_intake[i]
                x, y = intake_positions[intake_idx]
                angles = np.random.uniform(-np.pi/2, np.pi/2)
                px_new[i] = x + 0.5 + 0.8 * (1 + np.cos(angles))
                py_new[i] = y + 0.8 * np.sin(angles)
            
            # If particle hits top/bottom, bounce it
            if py_new[i] <= 0 or py_new[i] >= ny-1:
                py_new[i] = np.clip(py_new[i], 0, ny-1)
                vy_p *= -0.3  # Reverse vertical velocity with damping
                
            # Ensure particles don't go behind the wall
            px_new[i] = max(px_new[i], 0)
            
    return px_new, py_new

# Main simulation loop
plt.figure(figsize=(12, 6))
for t in range(nt):
    vx, vy = compute_velocity(vx, vy, p, dt, viscosity)
    p = compute_pressure(p, vx, vy)
    vx[airfoil_mask == 1] = 0
    vy[airfoil_mask == 1] = 0
    # For each point adjacent to the car boundary, set velocity to zero or mirror it to enforce no-slip

    # Left wall boundary - no flow through wall except at intakes
    vx[:, 0] = 4.0
    vy[:, 0] = 0
    
    # Boundary conditions
    vx[:, -1] = vx[:, -2] + 5  # Right boundary - open
    vy[:, -1] = vy[:, -2]
    
    
    # Top/bottom boundaries - partial slip
    vx[0, :] = vx[1, :] 
    vx[-1, :] = vx[-2, :] 
    vy[0, :] = -vy[1,:]*0.2
    vy[-1, :] = -vy[-2,:]*0.2
    
    vx[0, 0] = vx[1, 0]
    vx[-1, 0] = vx[-2, 0]
    vx[0, -1] = vx[1, -1]
    vx[-1, -1] = vx[-2, -1]

    vy[0, 0] = vy[0, 1]
    vy[-1, 0] = vy[-1, 1]
    vy[0, -1] = vy[0, -2]
    vy[-1, -1] = vy[-1, -2]
    
    # No-slip boundary at car
    vx[~fluid_mask] = 0
    vy[~fluid_mask] = 0
    
    vx[:, -1] = vx[:, -2]
    vy[:, -1] = vy[:, -2]
    p[:, -1] = p[:, -2]

    
    # Move particles
    if t%particle_spawn_rate == 0:
        particles()
    px, py = runge_kutta(px, py, vx, vy, dt)

    for i in range(num_particles):
        if particle_active[i] and px[i] > nx:
            particle_active[i] = False
    # Visualization every 10 frames
    #create_particles()
    if t % 10 == 0:
        plt.clf()
            
            # Background velocity magnitude
        speed = np.sqrt(vx**2 + vy**2)
        speed[airfoil_mask == 1] = np.nan  # Hide car area
        plt.imshow(speed, cmap='viridis', origin='lower', vmin=0, vmax=10.0)  # origin='lower' flips y-axis
            
        plt.colorbar(label='Velocity Magnitude')

            # Car outline
        car_outline = np.ma.masked_where(airfoil_mask == 0, airfoil_mask)
        plt.imshow(car_outline, cmap='Greys', origin='lower', alpha=0.7)  # also origin='lower'
            
            # Streamlines
        Y, X = np.mgrid[0:ny, 0:nx]
        plt.streamplot(X, Y, vx, vy, color='white', linewidth=0.8, 
                        density=1.0, arrowsize=0.6, minlength=0.5)
            
            # Particles
        valid = particle_active & (px >= 0) & (px < nx) & (py >= 0) & (py < ny)
        #plt.scatter(px[valid], py[valid], s=8, color='red', alpha=0.7)
            
        plt.title(f"Wind Tunnel Simulation (t = {t*dt:.2f}s)\n")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.xlim(0, nx)
        plt.ylim(0, ny)
        plt.tight_layout()
        plt.pause(0.01)
        plt.draw()

plt.show()

