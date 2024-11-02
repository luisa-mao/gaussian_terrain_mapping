import cv2
import numpy as np
from model import Model
import torch
torch.autograd.set_detect_anomaly(True)
from torch import nn
from tqdm import tqdm
# from arguments import ModelParams, PipelineParams, OptimizationParams


# Load the map and make an empty canvas the same size as the map
# map_path = "/scratch/luisamao/all_terrain/aerial_maps/map1.1.png"
map_path = "canvas.png"
map = cv2.imread(map_path)/255

# resize map to 1/4 of the original size
# map = cv2.resize(map, (map.shape[1]//4, map.shape[0]//4))/255
canvas = np.zeros_like(map, dtype=np.float32)  # Use float for accumulation

n_samples = 5
# n_samples = 1
# sample n_samples points from the map
# get the xy coordinates of the sampled points, and the pixel values

xy_coords = np.random.randint(0, map.shape[0], (n_samples, 2))
pixel_values = map[xy_coords[:, 0], xy_coords[:, 1]]
yaws = np.random.uniform(0, 2*np.pi, n_samples)
opacities = np.random.uniform(0, 1, n_samples)

# make points a list of tuples (x,y,yaws,opacities, pixel_values)
points = list(zip(xy_coords, yaws, opacities, pixel_values))

# make map into a tensor
map = torch.tensor(map, dtype=torch.float32).cuda()

# training loop
gaussians = Model(points)
iterations = 10
iter_start = torch.cuda.Event(enable_timing = True)
iter_end = torch.cuda.Event(enable_timing = True)

first_iter = 1
progress_bar = tqdm(range(first_iter, iterations), desc="Training progress")

l1_loss_fn = nn.L1Loss()

gaussians.training_setup()


for iteration in range(first_iter, iterations + 1):
    # clear the canvas
    canvas.fill(0)
    canvas_tensor = torch.tensor(canvas, dtype=torch.float32).cuda()

    tqdm.write(f"Rendering for iteration {iteration}")
    gaussians.render(canvas_tensor)  # Modify canvas_tensor in place

    # save the canvas to a file
    canvas_copy = (canvas_tensor.detach().cpu().numpy() * 255).astype(np.uint8)
    cv2.imwrite(f"canvas_{iteration}.png", canvas_copy)

    # log that rendering is done
    tqdm.write(f"Rendering done for iteration {iteration}")

    # Ensure map is a tensor and moved to the same device
    # map_tensor = torch.tensor(map, dtype=torch.float32).cuda()

    # l1 and ssim loss
    canvas_tensor.requires_grad_(True)
    loss = l1_loss_fn(canvas_tensor, map)
    tqdm.write(f"Loss for iteration {iteration}: {loss.item()}")

    # Backward pass
    loss.backward()

    # Print gradient of self._xy
    # print(f"Gradient of self._xy before optimization step: {gaussians._xy.grad}")

    # Optimizer step
    gaussians.optimizer.step()
    gaussians.optimizer.zero_grad(set_to_none=True)

    # Update progress bar
    progress_bar.update()
# print the xy coordinates of the gaussians
print("xy", gaussians._xy)
# print the scaling of the gaussians
print("scaling", gaussians._scaling)
# print the yaw of the gaussians
print("rotation", gaussians._rotation)
# print the opacity of the gaussians
print("opacity", gaussians._opacity)
# print the pixel values of the gaussians
print("rgb", gaussians._rgb)

