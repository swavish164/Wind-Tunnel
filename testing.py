import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from scipy.ndimage import convolve

# Parameters
nx, ny = 400, 150    # Grid size
dx, dy = 1.0, 1.0  # Grid spacing
nt = 500           # Number of time steps
dt = 0.01          # Time step
viscosity = 0.1    # Viscosity of the fluid
ar = 5

# Load the image
image = cv.imread('f1CarSide.jpg')

# Convert to grayscale and process the image to get edges
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
blurred = cv.GaussianBlur(gray, (5, 5), 0)
edges = cv.Canny(blurred, threshold1=30, threshold2=100)
kernel = np.ones((5, 5), np.uint8)  # Define a 5x5 kernel
closed_edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)
contours, _ = cv.findContours(closed_edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Create a filled image
filled_image = np.zeros_like(gray)
cv.drawContours(filled_image, contours, -1, (255), thickness=cv.FILLED)

# Resize the filled image to match the grid resolution (ny, nx)
def imageResize(image):
    height = np.shape(image)[0]
    width = np.shape(image)[1]
    if(height> (8/10)*nx):
        ratio = ((8/10)*nx / height)
        image = cv.resize(image,(int(height * ratio),int(width*ratio)))
    if(width> (8/10) * ny):
        ratio = ((8/10)*nx / width)
        image = cv.resize(image,(int(height * ratio),int(width*ratio)))
    return(image)
        
airfoil_resized = imageResize(filled_image)

# Create a binary mask of the resized airfoil image
airfoil_mask = (airfoil_resized > 0).astype(np.uint8)

# Initial conditions
vx = np.zeros((ny, nx))  # Velocity in the x direction
vy = np.zeros((ny, nx))  # Velocity in the y direction
p = np.zeros((ny, nx))   # Pressure

# Set boundary conditions for velocity
vx[:, 0] = 1  # Left boundary inlet velocity
vx[:, -1] = 0  # Right boundary outlet velocity

# Apply the airfoil mask to the velocity fields (setting velocity to zero inside the airfoil)
vx[airfoil_mask == 1] = 0
vy[airfoil_mask == 1] = 0

# Create a mesh grid for visualization
X, Y = np.meshgrid(np.linspace(0, nx * ar, nx), np.linspace(0, ny, ny))

num_particles = 100
px = np.random.rand(num_particles) * nx  
py = np.random.rand(num_particles) * ny
inlet_size = ny // 10

def compute_velocity(vx, vy, p, dt):
    """Apply Navier-Stokes discretization to compute velocity."""
    dpdx, dpdy = np.gradient(p)
    vx -= dt * dpdx
    vy -= dt * dpdy
    return vx, vy

def divergence(vx, vy):
    dx = np.gradient(vx, axis=1)
    dy = np.gradient(vy, axis=0)
    return dx + dy

J = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / 4  # Jacobi stencil

def compute_pressure(p, vx, vy):
    """Apply pressure Poisson equation to update pressure."""
    rhs = -divergence(vx, vy)
    for i in range(100):
        p = convolve(p, J, mode='constant', cval=0) + rhs / 2
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
        vx4, vy4 = interpolate_velocity(px[i] + 0.5 * k3x, py[i] + 0.5 * k3y, vx, vy)
        k4x, k4y = vx4 * dt, vy4 * dt
        px2[i] += 1 / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
        py2[i] += 1 / 6 * (k1y + 2 * k2y + 2 * k3y + k4y)
    return px2, py2

# Main time-stepping loop
for t in range(nt):
    vx, vy = compute_velocity(vx, vy, p, dt)  # Compute velocities
    p = compute_pressure(p, vx, vy)           # Update pressure
    px, py = runge_kutta(px, py, vx, vy, dt)  # Update particle positions

    # Apply the airfoil mask again after velocity update to enforce solid boundary condition
    vx[airfoil_mask == 1] = 0
    vy[airfoil_mask == 1] = 0

    # Optionally, visualize intermediate results every N steps
    if t % 20 == 0:
        plt.clf()
        plt.imshow(np.sqrt(vx**2 + vy**2), cmap='viridis')  # Display velocity magnitude
        plt.title(f"Velocity Magnitude at t={t*dt:.2f}")
        plt.colorbar()
        plt.pause(0.0001)

plt.show()

"""
# Visualize the final airfoil mask on the velocity grid
plt.imshow(airfoil_mask, cmap='gray')
plt.title("Airfoil Mask on Velocity Grid")
plt.colorbar()
plt.show()   """
