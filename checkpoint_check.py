import torch

# Load the checkpoint file
checkpoint_path = "/home/tawheed/parseq/outputs/parseq/2025-03-23_05-59-14/checkpoints/last.ckpt"
checkpoint = torch.load(checkpoint_path, map_location="cpu")


print(checkpoint.keys())
print(checkpoint['callbacks'])
print("\n\n")
print(checkpoint['global_step'])
