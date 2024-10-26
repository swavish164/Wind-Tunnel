import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from scipy.ndimage import convolve

nx, ny = 400, 150   
dx, dy = 1.0, 1.0  
nt = 500           
dt = 0.01         
viscosity = 0.1   
ar = 5

image = cv.imread('f1CarSide.jpg')

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
blurred = cv.GaussianBlur(gray, (5, 5), 0)
edges = cv.Canny(blurred, threshold1=30, threshold2=100)
kernel = np.ones((5, 5), np.uint8) 
closed_edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)
contours, _ = cv.findContours(closed_edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

filled_image = np.zeros_like(gray)
cv.drawContours(filled_image, contours, -1, (255), thickness=cv.FILLED)

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

airfoil_mask = (airfoil_resized > 0).astype(np.uint8)

vx = np.zeros((ny, nx))  
vy = np.zeros((ny, nx)) 
p = np.zeros((ny, nx))   

vx[:, 0] = 1  
vx[:, -1] = 0  

vx[airfoil_mask == 1] = 0
vy[airfoil_mask == 1] = 0

X, Y = np.meshgrid(np.linspace(0, nx * ar, nx), np.linspace(0, ny, ny))

num_particles = 100
px = np.random.rand(num_particles) * nx  
py = np.random.rand(num_particles) * ny
inlet_size = ny // 10

def compute_velocity(vx, vy, p, dt):
    dpdx, dpdy = np.gradient(p)
    vx -= dt * dpdx
    vy -= dt * dpdy
    return vx, vy

def divergence(vx, vy):
    dx = np.gradient(vx, axis=1)
    dy = np.gradient(vy, axis=0)
    return dx + dy

J = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / 4  

def compute_pressure(p, vx, vy):
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

for t in range(nt):
    vx, vy = compute_velocity(vx, vy, p, dt)  
    p = compute_pressure(p, vx, vy)           
    px, py = runge_kutta(px, py, vx, vy, dt)  

    vx[airfoil_mask == 1] = 0
    vy[airfoil_mask == 1] = 0

    if t % 20 == 0:
        plt.clf()
        plt.imshow(np.sqrt(vx**2 + vy**2), cmap='viridis')  
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
