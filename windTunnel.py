import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from scipy.ndimage import convolve

# Domain size and parameters
nx, ny = 200, 150    
dx, dy = 1.0, 1.0  
nt = 500          
dt = 0.01          
viscosity = 0.01    
ar = 5

# Load and process the airfoil image
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
    scale = min(max_height / height, max_width / width)
    new_size = (int(width * scale), int(height * scale))
    return cv.resize(image, new_size)

max_airfoil_height, max_airfoil_width = int(0.8 * ny), int(0.8 * nx)
resized_filled_image = imageResize(filled_image, max_airfoil_height, max_airfoil_width)

airfoil_mask = np.zeros((ny, nx), dtype=np.uint8)
resized_height, resized_width = resized_filled_image.shape[:2]
top_left_y = (ny - resized_height) // 2
top_left_x = (nx - resized_width) // 2

airfoil_mask[top_left_y:top_left_y + resized_height, top_left_x:top_left_x + resized_width] = \
    (resized_filled_image > 0).astype(np.uint8)

# Initialize velocity and pressure fields
vx = np.zeros((ny, nx))
vy = np.zeros((ny, nx))
p = np.zeros((ny, nx))

# Set inlet conditions - horizontal flow from the left side
inletHeight = ny // 3
inletStart = (ny - inletHeight) // 3
inletEnd = (inletStart + inletHeight) 

# Create parabolic velocity profile for the inlet
y_inlet = np.linspace(-1, 1, inletHeight)
vx[inletStart:inletEnd, 0] = 2.0 * (1 - y_inlet**2)  # horizontal velocity at inlet
vy[inletStart:inletEnd, 0] = 0.0  # zero vertical component at inlet

# Outlet conditions
vx[:, -1] = 0.0  # Prevent any inflow by setting outlet velocity to zero
vy[:, -1] = 0.0  # Ensure no vertical flow at the outlet

# Set no-flow conditions on top and bottom walls
vx[0, :] = 0.0  # Top wall
vy[0, :] = 0.0  # Top wall
vx[-1, :] = 0.0  # Bottom wall
vy[-1, :] = 0.0  # Bottom wall

# Apply wall boundary conditions on airfoil
vx[airfoil_mask == 1] = 0
vy[airfoil_mask == 1] = 0

# Particle positions for visualization
num_particles = 100
px = np.random.rand(num_particles) * nx  
py = np.random.rand(num_particles) * ny

# Function to compute velocity based on pressure gradient
def compute_velocity(vx, vy, p, dt, viscosity):
    dpdx, dpdy = np.gradient(p)
    vx -= dt * dpdx
    vy -= dt * dpdy
    # Adding viscosity diffusion term
    laplacian_vx = convolve(vx, np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]) / 4, mode='constant', cval=0)
    laplacian_vy = convolve(vy, np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]) / 4, mode='constant', cval=0)
    vx += viscosity * laplacian_vx * dt
    vy += viscosity * laplacian_vy * dt
    return vx, vy

def divergence(vx, vy):
    dx = np.gradient(vx, axis=1)
    dy = np.gradient(vy, axis=0)
    return dx + dy

# Jacobi iterative method for pressure correction
J = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / 4


def compute_pressure(p, vx, vy,tol = 1e-5):
    rhs = -divergence(vx, vy)
    for _ in range(100):
        p_new = convolve(p, J, mode='constant', cval=0) + rhs / 2
        if np.max(np.abs(p_new - p)) < tol:
            break
        p = p_new
    return p



def interpolate_velocity(x, y, vx, vy):
    ix, iy = int(x), int(y)
    ix = np.clip(ix, 0, vx.shape[1] - 1)
    iy = np.clip(iy, 0, vy.shape[0] - 1)
    return vx[iy, ix], vy[iy, ix]

def runge_kutta(px, py, vx, vy, dt):
    px2, py2 = px.copy(), py.copy()
    for i in range(len(px)):
        vx1, vy1 = interpolate_velocity(px[i], py[i], vx, vy)
        k1x, k1y = vx1 * dt, vy1 * dt
        vx2, vy2 = interpolate_velocity(px[i] + 0.5 * k1x, py[i] + 0.5 * k1y, vx, vy)
        k2x, k2y = vx2 * dt, vy2 * dt
        vx3, vy3 = interpolate_velocity(px[i] + 0.5 * k2x, py[i] + 0.5 * k2y, vx, vy)
        k3x, k3y = vx3 * dt, vy3 * dt
        vx4, vy4 = interpolate_velocity(px[i] + k3x, py[i] + k3y, vx, vy)
        k4x, k4y = vx4 * dt, vy4 * dt
        px2[i] += (k1x + 2 * k2x + 2 * k3x + k4x) / 6
        py2[i] += (k1y + 2 * k2y + 2 * k3y + k4y) / 6
    return px2, py2
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
# Main time-stepping loop
for t in range(nt):
    vx, vy = compute_velocity(vx, vy, p, dt, viscosity)
    p = compute_pressure(p, vx, vy)
    px, py = runge_kutta(px, py, vx, vy, dt)

    # Resetting boundary conditions after velocity updates
    vx[airfoil_mask == 1] = 0
    vy[airfoil_mask == 1] = 0

    # Ensure walls do not allow for any inflow
    vx[0, :] = 0.0  # No flow at top
    vx[-1, :] = 0.0  # No flow at bottom
    vy[:, 0] = 0.0  # No vertical flow at left
    vy[:, -1] = 0.0  # No vertical flow at right

    if t % 20 == 0:
        # Plot Velocity Magnitude
        velocity_magnitude = np.sqrt(vx**2 + vy**2)
        im1 = axs[0].imshow(velocity_magnitude, cmap='viridis', origin='lower')
        axs[0].set_title('Velocity Magnitude')
        axs[0].set_xlabel('X Position')
        axs[0].set_ylabel('Y Position')
        if 'cb1' in locals(): cb1.remove()  # Remove old colorbar if it exists
        cb1 = plt.colorbar(im1, ax=axs[0])  # Create colorbar for the velocity plot

        # Plot Pressure Field
        im2 = axs[1].imshow(p, cmap='viridis', origin='lower')
        axs[1].set_title('Pressure Field')
        axs[1].set_xlabel('X Position')
        if 'cb2' in locals(): cb2.remove()  # Remove old colorbar if it exists
        cb2 = plt.colorbar(im2, ax=axs[1])  # Create colorbar for the pressure plot

        plt.suptitle(f"Time = {t * dt:.2f} s")
        plt.pause(0.01)

plt.ioff()
plt.show()