import numpy as np

def generate_patches(image_size, patch_sizes):
    patches = []
    height = image_size
    top1 = left1 = k = 0
    
    while k < len(patch_sizes):
        row = []
        patch_size = patch_sizes[k]
        patch_count = int(height/patch_size)
        for i in range(patch_count):
            if i == 0 or i == patch_count - 1:
                for j in range(patch_count):
                    top = top1 + i * patch_size
                    left = left1 + j * patch_size
                    patch = (top, left, top + patch_size, left + patch_size)
                    row.append(patch)
 
            else: 
                top = top1 + i * patch_size
                left = left1
                patch = (top, left, top + patch_size, left + patch_size)
                row.append(patch)
                top = top1 + i * patch_size
                left = left1 + (patch_count - 1) * patch_size
                patch = (top, left, top + patch_size, left + patch_size)
                row.append(patch)                        
        top1 = left1 = top1 + patch_size
        height = height - patch_size * 2
        patches.extend(row)
        if k == len(patch_sizes)-1 and height != 0:
            k = k-1
        k = k + 1  
    return patches

# image_size = (384, 384)
# patch_sizes = [64, 32, 32, 16, 16, 8, 8, 4]
# s=0
# patches = generate_patches(image_size, patch_sizes)
# print(len(patches))
# # Print the generated patches
# for i, patch_row in enumerate(patches):
#     print(f"Patch Size: {patch_sizes[i]}") if i < len(patch_sizes) else print(f"Patch Size: {patch_sizes[-1]}")
#     s = s + len(patch_row)
#     # for patch in patch_row:
#     #     print(patch)
#     print(s)