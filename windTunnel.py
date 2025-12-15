import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from scipy.ndimage import distance_transform_edt, convolve

from Functions.calculations import advect, interpolate_velocity, reset, compute_pressure, runge_kutta
from Functions.boundaries import enforce_solid_boundary, enforceNoPenetration
from Functions.visuals import apply_solid_damping, project, particles

# Grid and simulation parameters
nx, ny = 500, 200    
dt = 0.02
nt = 200
viscosity = 0.01

# Load and process car image
image = cv.imread('f1CarSide.jpg')
bw = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
blurred = cv.GaussianBlur(bw, (5, 5), 0)
edges = cv.Canny(blurred, 30, 100)
kernel = np.ones((5,5), np.uint8)
closed_edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)
contours, _ = cv.findContours(closed_edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
filled_image = np.zeros_like(bw)
cv.drawContours(filled_image, contours, -1, 255, thickness=cv.FILLED)

def image_resize(image, max_height, max_width):
    height, width = image.shape[:2]
    scale = min(max_height / height, max_width / width*0.75)
    new_size = (int(width * scale), int(height * scale))
    return cv.resize(image, new_size)

max_airfoil_height, max_airfoil_width = int(0.5*ny), int(0.8*nx)
resized = image_resize(filled_image, max_airfoil_height, max_airfoil_width)
resized = (resized > 0).astype(np.uint8)
resized_flipped = np.flipud(resized)

top_left_y = int(0.1 * ny)
top_left_x = int(0.1 * nx)
airfoil_mask = np.zeros((ny, nx), dtype=np.uint8)
airfoil_mask[top_left_y:top_left_y+resized.shape[0], top_left_x:top_left_x+resized.shape[1]] = resized_flipped
fluid_mask = (airfoil_mask==0)

sdf = distance_transform_edt(airfoil_mask==0) - distance_transform_edt(airfoil_mask==1)
ny_sdf, nx_sdf = np.gradient(sdf)
norm = np.sqrt(nx_sdf**2 + ny_sdf**2)+1e-6
nx_sdf /= norm
ny_sdf /= norm

vx = np.ones((ny,nx))*4
vy = np.zeros((ny,nx))
p = np.zeros((ny,nx))

num_particles = 250
particle_spawn_rate = 10
num_intakes = 15
intake_positions = [(0, i*int(ny/num_intakes)-1) for i in range(1, num_intakes+2)]
px = np.zeros(num_particles)
py = np.zeros(num_particles)
particle_intake = np.zeros(num_particles, dtype=int)
particle_active = np.zeros(num_particles, dtype=bool)

laplacian_kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]])

plt.figure(figsize=(12,6))

for t in range(nt):
    vx, vy = advect(vx, vy, dt, fluid_mask)
    vx, vy = enforce_solid_boundary(vx, vy, fluid_mask, sdf)

    vx += viscosity*convolve(vx, laplacian_kernel, mode='nearest')*dt
    vy += viscosity*convolve(vy, laplacian_kernel, mode='nearest')*dt

    vx, vy = apply_solid_damping(vx, vy, sdf)
    vx, vy = project(vx, vy, p, fluid_mask, sdf)
    vx, vy = enforceNoPenetration(vx, vy, sdf, nx_sdf, ny_sdf)

    vx[:,0] = 4.0
    vy[:,0] = 0.0
    vx[:, -1] = vx[:, -2]
    vy[:, -1] = vy[:, -2]
    vx[0,:] = vx[1,:]
    vx[-1,:] = vx[-2,:]
    vy[0,:] = 0
    vy[-1,:] = 0
    vx[0,:] *= 0.95

    if t % particle_spawn_rate == 0:
        particles(px, py, particle_active, particle_intake, intake_positions, num_particles)

    px, py = runge_kutta(px, py, vx, vy, dt, num_particles, particle_active, particle_intake, intake_positions, airfoil_mask)

    if t % 10 == 0:
        plt.clf()
        speed = np.sqrt(vx**2 + vy**2)
        speed[airfoil_mask==1] = np.nan
        plt.imshow(speed, cmap='viridis', origin='lower', vmin=0, vmax=8)
        plt.colorbar(label='Velocity Magnitude')
        plt.contour(sdf, levels=[0], colors='white', linewidths=2)
        car_outline = np.ma.masked_where(airfoil_mask==0, airfoil_mask)
        plt.imshow(car_outline, cmap='gray', origin='lower', alpha=0.6)
        Y, X = np.mgrid[0:ny, 0:nx]
        plt.streamplot(X,Y,vx,vy,color='white',density=1.2,linewidth=0.8)
        plt.title(f"Wind Tunnel Simulation (t={t*dt:.2f}s)")
        plt.xlim(0,nx)
        plt.ylim(0,ny)
        plt.pause(0.01)

plt.show()
