import bpy
import torch
import matplotlib.pyplot as plt
import numpy as np
import imageio
import kornia.filters as K

from pytorch3d.structures import Meshes, join_meshes_as_scene
from pytorch3d.renderer import (
    RasterizationSettings, MeshRenderer, MeshRasterizer, SoftPhongShader,
    SoftSilhouetteShader, PerspectiveCameras, PointLights, TexturesVertex, look_at_view_transform
)

from tqdm import tqdm 
from einops import einsum

# import lpips


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    print("GPU Detected")
else:
    device = torch.device("cpu")

bpy.ops.wm.open_mainfile(filepath="Blender Files/bolt.blend")
# bpy.ops.wm.open_mainfile(filepath="bolt_nopos.blend")
# Camera and lights (customize if needed)
R, T = look_at_view_transform(dist=7, elev=40, azim=00)
cameras = PerspectiveCameras(R=R, T=T, device=device)
lights = PointLights(device=device, location=[[0.0, 0.0, 3.0]])

sigma = 1e-4
raster_settings_silhouette = RasterizationSettings(
    image_size=512, 
    blur_radius=np.log(1. / 1e-4 - 1.)*sigma, 
    faces_per_pixel=50, 
)

# Basic renderer
image_size = 512
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings_silhouette
    ),
    shader=SoftSilhouetteShader()
)

visrenderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings_silhouette
    ),
    shader=SoftPhongShader(device=device, cameras=cameras, lights=lights)
)


target_collection = bpy.data.collections.get("Target")

meshes = []
textures = []
color = torch.tensor([0.5, 0.5, 0.5], device=device)  # light gray
# Loop through each mesh object
for obj in target_collection.objects:
    if obj.type == 'MESH':
        mesh = obj.data

        # Get vertices
        verts = np.array([obj.matrix_world @ v.co for v in mesh.vertices])
        verts_tensor = torch.tensor(verts[np.newaxis,:,:], dtype=torch.float32, device=device)  # shape (1, V, 3)
        faces = np.array([[v for v in poly.vertices] for poly in obj.data.polygons], dtype=np.int32)
        faces_tensor = torch.tensor(faces[np.newaxis,:,:], dtype=torch.int64, device=device)    # shape (1, F, 3)

        vertex_colors = color[None, None, :].expand(verts_tensor.shape)    # (1, V, 3)
        textures = TexturesVertex(verts_features=vertex_colors)

        mesh = Meshes(verts=verts_tensor, faces=faces_tensor, textures=textures)
        meshes.append(mesh)

scene = join_meshes_as_scene(meshes, True)

target_image = renderer(scene, cameras=cameras, lights=lights)[0,...,3]
# image_tensor shape: (N, C, H, W)
target_image = K.gaussian_blur2d(target_image.unsqueeze(0).unsqueeze(0), (101,101), (10,10))[0,0]
print(target_image.shape)
# timage = (target_image[...,:3]*255).byte().detach().cpu().numpy()
# imageio.imsave('location_training_process.png', timage[0])
plt.imshow(target_image.cpu().numpy())  # Only show the first image in the batch
plt.axis("off")
plt.show()

# 1. Extract all of the custom properties from the empty control object
control = bpy.data.objects["Control"]
custom_props = {}

#Randomize before optimization
control["l_x"] = -3.2
control["l_y"] = 0.0
control["body_length"] = 2.3
control["body_diameter"] = 0.78
control["head_diameter"] = 3.0
control["head_thickness"] = 0.5

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
        params_assign_string += f"{prop} = torch.tensor([{custom_props[prop]}], device=device, requires_grad=True)\n"
        # params_assign_string += f"{prop} = torch.tensor([1.0], device=device, requires_grad=True)\n" #Mess up the initial parameter so we see something interesting
        params_assign_string += f"opt_props.append({prop})\n"
        
    print(params_assign_string)
    exec(params_assign_string)

initialize_params(custom_props)

# 2. Extract the list of drivers from the objects in the scene
drivers_list = []
for obj in target_collection.objects:
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

    drivers_list.append(drivers)
    
# 3. Evaluate each driver using the custom properties as dependencies

def get_transform_update_string(drivers):
    vstr = "def update_transform():\n"
    vstr += "\ttransform_matrices = []\n\n"
    for drivers in drivers_list:
        for driver in drivers:
            vstr += f"\t{driver} = torch.zeros(3, device=device)\n"
            for axis in range(3):
                vars = drivers[driver][axis]["vars"]
                for var in vars:
                    vstr += f"\t{var} = {vars[var][2:-2]}\n"
                vstr
                exp = drivers[driver][axis]["exp"]
                vstr += f"\t{driver}[{axis}] = {exp}\n"
        vstr += "\ttransform_matrices.append(build_transform(location, rotation_euler, scale))\n\n"
    vstr += "\treturn transform_matrices\n"
    return vstr

print(get_transform_update_string(drivers))

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

#Optimize

#Initialize the mesh
verts_tensor_list = []
faces_tensor_list = []
textures_list = []

for obj in target_collection.objects:
    if obj.type == 'MESH':
        mesh = obj.data

        # Get vertices
        verts = np.array([v.co for v in mesh.vertices])
        verts_tensor_list.append(torch.tensor(verts[np.newaxis,:,:], dtype=torch.float32, device=device))  # shape (1, V, 3)
        faces = np.array([[v for v in poly.vertices] for poly in obj.data.polygons], dtype=np.int32)
        faces_tensor_list.append(torch.tensor(faces[np.newaxis,:,:], dtype=torch.int64, device=device))    # shape (1, F, 3)

        vertex_colors = color[None, None, :].expand(verts_tensor.shape)    # (1, V, 3)
        textures_list.append(TexturesVertex(verts_features=vertex_colors))

# print(verts_tensor.shape)
optimizer = torch.optim.Adam(opt_props, lr=0.03)
# optimizer = torch.optim.SGD(opt_props, lr=0.00001, momentum=0.0001)

num_epochs = 300
epochs_per_save = 10
gif_images = np.zeros((num_epochs//epochs_per_save, image_size, image_size, 3), dtype=np.uint8)

losses = []
with tqdm(range(num_epochs)) as titer:
    for i in titer:
        optimizer.zero_grad()
        transform_matrices = update_transform()
        meshes = []
        # Apply each of the transforms
        for j in range(len(transform_matrices)):
            matrix_world = transform_matrices[j]
            ones = torch.ones(verts_tensor_list[j].shape[0], verts_tensor_list[j].shape[1], 1, device=device)
            nverts = einsum(torch.cat([verts_tensor_list[j], ones], dim=-1), matrix_world,'b i v, j v-> b i j')
            meshes.append(Meshes(verts=nverts, faces=faces_tensor_list[j], textures=textures_list[j]))
        scene = join_meshes_as_scene(meshes, True)
        
        cimage = renderer(scene, cameras=cameras, lights=lights)[0,...,3]
        cimage = K.gaussian_blur2d(cimage.unsqueeze(0).unsqueeze(0), (101,101), (10,10))[0,0]
        #Save the image into a GIF to show the training proces
        if(not i % epochs_per_save):
            simage = visrenderer(scene).flip(1)
            gif_images[i//epochs_per_save] = (simage[...,:3]*255).byte().detach().cpu().numpy()

        loss = torch.sum((cimage - target_image)**2)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        titer.set_postfix(loss=loss.item(), length=opt_props[-1].item())
    
# Save GIF
default_duration = 50
lag = 1000
duration = [default_duration] * (len(gif_images)-1) + [lag]
imageio.mimsave('Examples/outputs/location_training_process_translate.gif', gif_images, duration=duration, loop=0)

lossfig = plt.figure(figsize=(4,4))
plt.plot(losses)
plt.xlabel("Iteration")
plt.ylabel("MSE Loss")
plt.title("Training Loss")
plt.tight_layout()
plt.savefig("Examples/outputs/bolt_demo_training_loss_translate.png", bbox_inches='tight')