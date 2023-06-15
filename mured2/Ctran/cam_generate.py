import torch
from torch.autograd import Function
import cv2
import numpy as np

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activation_maps = None
        self.gradient_maps = None
        self.register_hooks()

    def register_hooks(self):
        def forward_hook(module, input, output):
            self.activation_maps = output

        def backward_hook(module, grad_input, grad_output):
            self.gradient_maps = grad_output[0]

        target_layer_module = dict([*self.model.named_modules()])[self.target_layer]
        target_layer_module.register_forward_hook(forward_hook)
        target_layer_module.register_backward_hook(backward_hook)

    def generate_heatmap(self, input_image, target_class):
        self.model.zero_grad()
        output = self.model(input_image)
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        gradient_maps = self.gradient_maps.detach().cpu().numpy()[0]
        activation_maps = self.activation_maps.detach().cpu().numpy()[0]
        weights = np.mean(gradient_maps, axis=(1, 2))
        cam = np.sum(weights * activation_maps, axis=0)
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input_image.shape[2:])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam

def cam_generate(model, image, label):
    transform = ...  # Define the necessary transformation to preprocess the image
    input_image = transform(image).unsqueeze(0)
    cam = GradCAM(model, target_layer='your_target_layer')
    heatmap = cam.generate_heatmap(input_image, target_class=label)
    return heatmap