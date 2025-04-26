import bpy
import torch

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    print("GPU Detected")
else:
    device = torch.device("cpu")

bpy.ops.wm.open_mainfile(filepath="transform_driver_demo.blend")

# 1. Extract all of the custom properties from the empty control object
control = bpy.data.objects["Control"]
custom_props = {}

if control is not None:
    for key, value in control.items():
        if not key.startswith("_"):
            custom_props[key] = value

#Assign each of the custom properties to custom variables
def initialize_params(custom_props):
    params_assign_string = ""
    for prop in custom_props:
        params_assign_string += f"global {prop}\n"
        params_assign_string += f"{prop} = torch.tensor([{custom_props[prop]}], device=device, requires_grad=True)\n"
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
location = torch.tensor([0,0,0], dtype=float)
rotation_euler = torch.tensor([0,0,0], dtype=float)
scale = torch.tensor([0,0,0], dtype=float)

def get_transform_update_string(drivers):
    vstr = "def update_transform():\n"
    for driver in drivers:
        vstr += f"\tglobal {driver}\n"
        for axis in range(3):
            vars = drivers[driver][axis]["vars"]
            for var in vars:
                vstr += f"\t{var} = {vars[var][2:-2]}\n"
            exp = drivers[driver][axis]["exp"]
            vstr += f"\t{driver}[{axis}] = {exp}\n"
        # vstr += f"\t{driver} = torch.tensor({driver})\n"
    vstr += "\treturn build_transform(location, rotation_euler, scale)\n"
    return vstr

print(get_transform_update_string(drivers))

# 4. Compute the transform matrix using the drivers
def build_transform(location, rotation, scale):
    lx, ly, lz = location
    sx, sy, sz = scale
    cx, cy, cz = torch.cos(rotation)
    sxr, syr, szr = torch.sin(rotation)

    Rx = torch.tensor([
        [1, 0, 0],
        [0, cx, -sxr],
        [0, sxr, cx]
    ])

    Ry = torch.tensor([
        [cy, 0, syr],
        [0, 1, 0],
        [-syr, 0, cy]
    ])

    Rz = torch.tensor([
        [cz, -szr, 0],
        [szr, cz, 0],
        [0, 0, 1]
    ])
    R = Rz @ Ry @ Rx
    S = torch.diag(torch.tensor([sx, sy, sz]))
    RS = R @ S

    M = torch.eye(4)
    M[:3, :3] = RS
    M[:3, 3] = torch.tensor([lx, ly, lz])

    return M

exec(get_transform_update_string(drivers))
print(update_transform())