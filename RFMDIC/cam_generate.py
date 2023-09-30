import torch
from captum.attr import ShapleyValueSampling
import numpy as np

class ShapCAMGenerator:
    def __init__(self, model, device):
        self.model = model                               # Initializing the model
        self.device = device                             # Initializing the device
        self.sampler = ShapleyValueSampling(model)       # Initializing sampler instance
    
    def generate_shap_cam(self, image, target_class):
        image = image.to(self.device)                    # Passing the image to the device
        
        self.model.eval()                                # Evaluating the model on the image
        self.sampler.model = self.model                  # Set the model for the sampler
        print("done")
        attributions = self.sampler.attribute(image, target=target_class)  # Pass the target_class to attribute method\
        print("done")
        shap_cam = torch.sum(attributions, dim=1, keepdim=True)  # Sum over channels
        
        shap_cam = shap_cam.cpu().detach().numpy()[0]            # Convert shap cam tensor to numpy array
        shap_cam = np.maximum(0, shap_cam)                       # Apply ReLU to get only positive attributions. Set negative contributions to 0
        shap_cam /= np.max(shap_cam)                             # Normalize to [0, 1]
        
        return shap_cam


# Function to generate heatmaps for a chunk of images
def generate_heatmaps(image_chunk, model1, model2, shap_cam_generator1, shap_cam_generator2, device, a, b):
    
    heatmaps1 = []
    heatmaps2 = []
    for image in image_chunk:
        image = image.unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs1 = model1(image)
            outputs2 = model2(image)
            weighted_outputs = a * torch.sigmoid(outputs1) + b * torch.sigmoid(outputs2)
            print(weighted_outputs)
            target_class = torch.argmax(weighted_outputs).unsqueeze(0)
            print(target_class)
            
        # Generate heatmap for model1
        heatmap1 = shap_cam_generator1.generate_shap_cam(image, target_class)
        heatmaps1.append(heatmap1)
        
        # Generate heatmap for model2
        heatmap2 = shap_cam_generator2.generate_shap_cam(image, target_class)
        heatmaps2.append(heatmap2)
        
    return heatmaps1, heatmaps2


# Function to generate heatmaps for a chunk of images
def generate_heatmap(image_chunk, model1, model2, shap_cam_generator1, device, a, b):
    
    heatmaps1 = []
    for image in image_chunk:
        image = image.unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs1 = model1(image)
            outputs2 = model2(image)
            weighted_outputs = a * torch.sigmoid(outputs1) + b * torch.sigmoid(outputs2)
            print(weighted_outputs)
            target_class = torch.argmax(weighted_outputs).unsqueeze(0)
            print(target_class)
            
        # Generate heatmap for model1
        heatmap1 = shap_cam_generator1.generate_shap_cam(image, target_class)
        heatmaps1.append(heatmap1)
        
        # Generate heatmap for model2
        #heatmap2 = shap_cam_generator2.generate_shap_cam(image, target_class)
        #heatmaps2.append(heatmap2)
        
    return heatmaps1