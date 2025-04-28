import bpy
import torch
import matplotlib.pyplot as plt
import numpy as np
import imageio

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    RasterizationSettings, MeshRenderer, MeshRasterizer, SoftPhongShader,
    PerspectiveCameras, PointLights, TexturesVertex, look_at_view_transform
)

from tqdm import tqdm 
from einops import einsum

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    print("GPU Detected")
else:
    device = torch.device("cpu")

bpy.ops.wm.open_mainfile(filepath="Blender Files/transform_driver_demo.blend")
# Camera and lights (customize if needed)
R, T = look_at_view_transform(dist=12, elev=45, azim=60)
cameras = PerspectiveCameras(R=R, T=T, device=device)
lights = PointLights(device=device, location=[[0.0, 3.0, 3.0]])

# Basic renderer
image_size = 512
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=RasterizationSettings(image_size=image_size)
    ),
    shader=SoftPhongShader(device=device, cameras=cameras, lights=lights)
)

cube = bpy.data.objects["Target"]

#Render and show the target mesh (no rotation or scale)
depsgraph = bpy.context.evaluated_depsgraph_get()

# Get the evaluated version of the object (including modifiers, constraints, etc.)
eval_cube = cube.evaluated_get(depsgraph).to_mesh()

faces = np.array([[v for v in poly.vertices] for poly in cube.data.polygons], dtype=np.int32)
verts = np.array([v.co[:] for v in cube.data.vertices], dtype=np.float32)
#transform by world matrix
verts_tx = np.array([cube.matrix_world @ v.co for v in eval_cube.vertices])
# verts_tx = np.array([v.co[:] for v in (cube.matrix_world @ cube.data.vertices[:])], dtype=np.float32)
verts_tensor = torch.tensor(verts[np.newaxis,:,:], dtype=torch.float32, device=device)  # shape (1, V, 3)
verts_tensor_tx = torch.tensor(verts_tx[np.newaxis,:,:], dtype=torch.float32, device=device)  # shape (1, V, 3)
faces_tensor = torch.tensor(faces[np.newaxis,:,:], dtype=torch.int64, device=device)    # shape (1, F, 3)
color = torch.tensor([0.8, 0.8, 0.8], device=verts_tensor.device)  # light gray
vertex_colors = color[None, None, :].expand(verts_tensor.shape)    # (1, V, 3)
textures = TexturesVertex(verts_features=vertex_colors)
mesh = Meshes(verts=verts_tensor_tx, faces=faces_tensor, textures=textures)
target_image = renderer(mesh)

plt.imshow(target_image[0].cpu().numpy())  # Only show the first image in the batch
plt.axis("off")
plt.show()

# 1. Extract all of the custom properties from the empty control object
control = bpy.data.objects["Control"]
custom_props = {}

if control is not None:
    for key, value in control.items():
        if not key.startswith("_"):
            custom_props[key] = value

#Assign each of the custom properties to custom variables
def initialize_params(custom_props):
    params_assign_string = "global opt_props\n"
    params_assign_string += "opt_props = []\n"
    for prop in custom_props:
        params_assign_string += f"global {prop}\n"
        # params_assign_string += f"{prop} = torch.tensor([{custom_props[prop]}], device=device, requires_grad=True)\n"
        params_assign_string += f"{prop} = torch.tensor([1.0], device=device, requires_grad=True)\n" #Mess up the initial parameter so we see something interesting
        params_assign_string += f"opt_props.append({prop})\n"
        
    print(params_assign_string)
    exec(params_assign_string)

initialize_params(custom_props)

# 2. Extract the list of drivers from the objects in the scene
obj = bpy.data.objects["Target"]
drivers = {}
#defaults
drivers["location"] =       [{"exp":"0","vars":{}},
                             {"exp":"0","vars":{}},
                             {"exp":"0","vars":{}}]

drivers["rotation_euler"] = [{"exp":"0","vars":{}},
                             {"exp":"0","vars":{}},
                             {"exp":"0","vars":{}}]

drivers["scale"] =          [{"exp":"1","vars":{}},
                             {"exp":"1","vars":{}},
                             {"exp":"1","vars":{}}]

if obj.animation_data and obj.animation_data.drivers:
    for driver in obj.animation_data.drivers:
        drivers[driver.data_path][driver.array_index]["exp"] = driver.driver.expression
        variables = {}
        for var in driver.driver.variables:
            variables[var.name] = var.targets[0].data_path
        drivers[driver.data_path][driver.array_index]["vars"] = variables
else:
    print("No drivers on this object.")

# 3. Evaluate each driver using the custom properties as dependencies


def get_transform_update_string(drivers):
    vstr = "def update_transform():\n"
    for driver in drivers:
        vstr += f"\t{driver} = torch.zeros(3, device=device)\n"
        for axis in range(3):
            vars = drivers[driver][axis]["vars"]
            for var in vars:
                vstr += f"\t{var} = {vars[var][2:-2]}\n"
            vstr
            exp = drivers[driver][axis]["exp"]
            vstr += f"\t{driver}[{axis}] = {exp}\n"
    vstr += "\treturn build_transform(location, rotation_euler, scale)\n"
    return vstr

# 4. Compute the transform matrix using the drivers
def build_transform(location, rotation, scale):

    cx, cy, cz = torch.cos(rotation)
    sxr, syr, szr = torch.sin(rotation)

    ones = torch.ones((), device=device)
    zeros = torch.zeros((), device=device)

    Rx = torch.stack([
        torch.stack([ones, zeros, zeros], dim=-1),
        torch.stack([zeros, cx, -sxr], dim=-1),
        torch.stack([zeros, sxr, cx], dim=-1)
    ], dim=0)

    Ry = torch.stack([
        torch.stack([cy, zeros, syr], dim=-1),
        torch.stack([zeros, ones, zeros], dim=-1),
        torch.stack([-syr, zeros, cy], dim=-1)
    ], dim=0)

    Rz = torch.stack([
        torch.stack([cz, -szr, zeros], dim=-1),
        torch.stack([szr, cz, zeros], dim=-1),
        torch.stack([zeros, zeros, ones], dim=-1)
    ], dim=0)

    R = Rz @ Ry @ Rx

    S = torch.diag(torch.stack([scale[0], scale[1], scale[2]]))
    RS = R @ S

    loc = location.view(3, 1)
    upper = torch.cat([RS, loc], dim=1)
    
    return upper.to(dtype=torch.float32)


exec(get_transform_update_string(drivers))
# def update_transform(): 
#         location = torch.zeros(3, device=device)
#         rotation_euler = torch.zeros(3, device=device)
#         rotation_euler[0] = 0.3*scale1
#         scale = torch.ones(3, device=device)
#         scale[0] = 2*scale1
#         return build_transform(location, rotation_euler, scale)
print(get_transform_update_string(drivers))

# location = torch.zeros(3, device=device)
# rotation_euler = torch.zeros(3, device=device)
# scale = torch.ones(3, device=device)

#Optimize
print(verts_tensor.shape)
optimizer = torch.optim.Adam(opt_props, lr=0.02)

num_epochs = 100
epochs_per_save = 5
gif_images = np.zeros((num_epochs//epochs_per_save, image_size, image_size, 3), dtype=np.uint8)
losses = []

with tqdm(range(num_epochs)) as titer:
    for i in titer:
        optimizer.zero_grad()
        matrix_world = update_transform()

        ones = torch.ones(verts_tensor.shape[0], verts_tensor.shape[1], 1, device=verts_tensor.device)
        nverts = einsum(torch.cat([verts_tensor, ones], dim=-1), matrix_world,'b i v, j v-> b i j')
        mesh = Meshes(verts=nverts, faces=faces_tensor, textures=textures)
        cimage = renderer(mesh)
        #Save the image into a GIF to show the training proces
        if(not i % epochs_per_save):
            gif_images[i//epochs_per_save] = (cimage[...,:3]*255).byte().detach().cpu().numpy()

        loss = torch.sum((cimage - target_image)**2)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        titer.set_postfix(loss=loss.item(), scale1=opt_props[0].item())
    
# Save GIF
default_duration = 50
lag = 1000
duration = [default_duration] * (len(gif_images)-1) + [lag]
imageio.mimsave('Examples/outputs/driver_training_process.gif', gif_images, duration=duration, loop=0)

lossfig = plt.figure(figsize=(4,4))
plt.plot(losses)
plt.xlabel("Iteration")
plt.ylabel("MSE Loss")
plt.title("Training Loss")
plt.tight_layout()
plt.savefig("Examples/outputs/driver_demo_training_loss.png", bbox_inches='tight')
