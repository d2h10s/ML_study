import torch
 
#  Returns a bool indicating if CUDA is currently available.
print(torch.cuda.is_available())
#  True
 
#  Returns the index of a currently selected device.
print(torch.cuda.current_device())
#  0
 
#  Returns the number of GPUs available.
print(torch.cuda.device_count())
#  1
 
#  Gets the name of a device.
print(torch.cuda.get_device_name(0))
#  'GeForce GTX 1060'
 
#  Context-manager that changes the selected device.
#  device (torch.device or int) – device index to select. 
print(torch.cuda.device(0))