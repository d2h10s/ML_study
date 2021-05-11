from tensorboardX import SummaryWriter
import numpy as np

if __name__ == '__main__':
    writer = SummaryWriter()

    funcs = {"sin":np.sin, "cos":np.cos, "tan":np.tan}

    for angle in range(-360, 360):
        angle_rad = angle * np.pi / 180
        for name, fun in funcs.items():
            val = fun(angle_rad)
            writer.add_scalar(name, val, angle)
    writer.close()