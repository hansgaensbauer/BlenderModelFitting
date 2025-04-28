import bpy
import matplotlib.pyplot as plt
import numpy as np
import imageio

import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    RasterizationSettings, MeshRenderer, MeshRasterizer, SoftPhongShader,
    PerspectiveCameras, PointLights, TexturesVertex, look_at_view_transform
)

from tqdm import tqdm 

# Setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    print("GPU Detected")
else:
    device = torch.device("cpu")

bpy.ops.wm.open_mainfile(filepath="Blender Files/shapekey_demo.blend")

# Camera and lights (customize if needed)
R, T = look_at_view_transform(dist=3, elev=60, azim=20)
cameras = PerspectiveCameras(R=R, T=T, device=device)
lights = PointLights(device=device, location=[[3.0, 0.0, 3.0]])

# Basic renderer
image_size = 512
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=RasterizationSettings(image_size=image_size)
    ),
    shader=SoftPhongShader(device=device, cameras=cameras, lights=lights)
)

cube = bpy.context.scene.objects[0].data

#Render and show the target mesh
faces = np.array([[v for v in poly.vertices] for poly in cube.polygons], dtype=np.int32)
verts = np.array([v.co[:] for v in cube.vertices], dtype=np.float32)
verts_tensor = torch.tensor(verts[np.newaxis,:,:], dtype=torch.float32, device=device)  # shape (1, V, 3)
faces_tensor = torch.tensor(faces[np.newaxis,:,:], dtype=torch.int64, device=device)    # shape (1, F, 3)
color = torch.tensor([0.8, 0.8, 0.8], device=verts_tensor.device)  # light gray
vertex_colors = color[None, None, :].expand(verts_tensor.shape)    # (1, V, 3)
textures = TexturesVertex(verts_features=vertex_colors)
mesh = Meshes(verts=verts_tensor, faces=faces_tensor, textures=textures)
target_image = renderer(mesh)
# plt.imshow(target_image[0].cpu().numpy())  # Only show the first image in the batch
# plt.axis("off")
# plt.show()

# Apply a shape key externally
basis = cube.shape_keys.key_blocks["Basis"].data
basis_verts = torch.tensor(np.array([v.co[:] for v in basis], dtype=np.float32)[np.newaxis,:,:], requires_grad=True, device=device)
key1 = cube.shape_keys.key_blocks["Wider Top"].data
key1_verts = torch.tensor(np.array([v.co[:] for v in key1], dtype=np.float32)[np.newaxis,:,:], requires_grad=True, device=device)

#Compute the distorted mesh
keyval = torch.tensor([0.8], requires_grad=True, device=device)

# Render
image = renderer(mesh)

optimizer = torch.optim.Adam([keyval], lr=0.05)
verts = torch.zeros_like(basis_verts, requires_grad=True)

num_epochs = 100
epochs_per_save = 5
losses = []
gif_images = np.zeros((num_epochs//epochs_per_save, image_size, image_size, 3), dtype=np.uint8)
with tqdm(range(num_epochs)) as titer:
    for i in titer:
        optimizer.zero_grad()
        #Update the mesh
        verts = basis_verts + keyval * (key1_verts - basis_verts)
        mesh = Meshes(verts=verts, faces=faces_tensor, textures=textures)

        #Render
        cimage = renderer(mesh)
        #Save the image into a GIF to show the training proces
        if(not i % epochs_per_save):
            gif_images[i//epochs_per_save] = (cimage[...,:3]*255).byte().detach().cpu().numpy()

        loss = torch.sum((cimage - target_image)**2)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        titer.set_postfix(loss=loss.item(), keyval=keyval.item())

# Save GIF
default_duration = 50
lag = 1000
duration = [default_duration] * (len(gif_images)-1) + [lag]
imageio.mimsave('Examples/outputs/shapekey_training_process.gif', gif_images, duration=duration, loop=0)

lossfig = plt.figure(figsize=(4,4))
plt.plot(losses)
plt.xlabel("Iteration")
plt.ylabel("MSE Loss")
plt.title("Training Loss")
plt.tight_layout()
plt.savefig("Examples/outputs/shapekey_demo_training_loss.png", bbox_inches='tight')