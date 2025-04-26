import bpy
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

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

bpy.ops.wm.open_mainfile(filepath="shapekey_demo.blend")

# Camera and lights (customize if needed)
R, T = look_at_view_transform(dist=5, elev=90, azim=0)
cameras = PerspectiveCameras(R=R, T=T, device=device)
lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

# Basic renderer
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=RasterizationSettings(image_size=512)
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

optimizer = torch.optim.Adam([keyval], lr=0.1)
verts = torch.zeros_like(basis_verts, requires_grad=True)
with tqdm(range(100)) as titer:
    for i in titer:
        optimizer.zero_grad()
        #Update the mesh
        verts = basis_verts + keyval * (key1_verts - basis_verts)
        mesh = Meshes(verts=verts, faces=faces_tensor, textures=textures)
        #Render
        cimage = renderer(mesh)
        loss = torch.sum((cimage - target_image)**2)
        loss.backward()
        optimizer.step()
        titer.set_postfix(loss=loss.item(), keyval=keyval.item())