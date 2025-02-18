# MIT License
# Copyright (c) 2025 Massachusetts Institute of Technology
# See the LICENSE file for full license details.

import torch
import torch.nn as nn
from torchvision.models import VGG19_Weights
from torchvision.models.vgg import vgg19


class VGGTextureDescriptor(nn.Module):
    """Texture descriptor evaluation based on a pretrained VGG19 network.
    """
    def __init__(self):
        super().__init__()

        # Record intermediate results from the feature extraction network to compute the texture
        # descriptor
        self.features = []

        # Set up the feature extraction network
        self._setup_model()

        # Image statistics for normalizing an input texture
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1))

    def _setup_model(self):
        """Initialize the texture feature extraction model.
        """
        # Get a pretrained VGG19 network and set it to evaluation state
        model: nn.Sequential = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        model.eval()

        # Disable network training
        for param in model.parameters():
            param.requires_grad_(False)

        # Change the max pooling to average pooling
        for i, module in enumerate(model):
            if isinstance(module, nn.MaxPool2d):
                model[i] = nn.AvgPool2d(kernel_size=2)

        # The forward hook function for capturing output at a certain network layer
        def forward_hook(module, input, output):
            self.features.append(output)

        # Register the forward hook function
        for i in (4, 9, 18, 27):
            model[i].register_forward_hook(forward_hook)

        self.model = model

    def _texture_descriptor(self, img):
        """Compute the texture descriptor of an input image (B, C, H, W).
        """
        # Normalize the input image
        img = (img - self.mean) / self.std

        # Run the VGG feature extraction network
        self.features.clear()
        self.features.append(self.model(img))

        def gram_matrix(img_feature):
            mat = img_feature.flatten(-2)
            gram = torch.matmul(mat, mat.transpose(-2, -1)) / mat.shape[-1]
            return gram.flatten(1)

        # Compute the Gram matrices using recorded features
        # The feature descriptor has a shape of (B, F), where F is feature length
        return torch.cat([gram_matrix(img_feature) for img_feature in self.features], dim=1)

    def forward(self, img):
        """Compute the texture descriptor of an input image at multiple scales.
        """
        # Compute the texture descriptor at native resolution
        return self._texture_descriptor(img.contiguous())
