import torch

from .base_networks import *
from .spatial_transformer import SpatialTransform, Dense3DSpatialTransformer
from monai.networks.blocks import Warp
import os
os.environ['VXM_BACKEND']="pytorch"
os.environ['NEURITE_BACKEND']="pytorch"
import voxelmorph as vxm
from monai.networks.nets import LocalNet, GlobalNet

nf_enc = [16, 32, 32, 32]
nf_dec = [32, 32, 32, 32, 32, 16, 16]
nb_features = [
    nf_enc,
    nf_dec
]
# vxm_model = vxm.networks.VxmDense(inshape=(128, 128, 128), nb_unet_features=nb_features)

def norm(ddf):
    return (2.0 * (ddf - torch.min(ddf))/(torch.max(ddf) - torch.min(ddf))) - 1.0

def max_norm(ddf):
    return(ddf / torch.max(torch.abs(ddf)))


class RecursiveCascadeNetwork(nn.Module):
    def __init__(self, n_cascades, im_size=(512, 512)):
        super(RecursiveCascadeNetwork, self).__init__()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.stems = []
        # See note in base_networks.py about the assumption in the image shape

        self.stems.append(GlobalNet(
            image_size=im_size,
            spatial_dims=3,
            in_channels=2,
            num_channel_initial=4,
            depth=5
        ).to(device))

        # self.stems.append(VTNAffineStem(dim=len(im_size), im_size=im_size[0]))


        for i in range(n_cascades):
            self.stems.append(LocalNet(
                                        spatial_dims=3,
                                        in_channels=2,
                                        out_channels=3,
                                        num_channel_initial=32,
                                        extract_levels=[3],
                                        out_activation=None,
                                        out_kernel_initializer="zeros"))
            # self.stems.append(VTN(dim=len(im_size), flow_multiplier=1.0 / n_cascades))

        # Parallelize across all available GPUs
        if torch.cuda.device_count() > 1:
            self.stems = [nn.DataParallel(model) for model in self.stems]

        for model in self.stems:
            model.to(device)

        # self.reconstruction = SpatialTransform(im_size)
        self.reconstruction = Warp()
        # self.reconstruction = Dense3DSpatialTransformer(device=device)
        self.reconstruction = nn.DataParallel(self.reconstruction)
        self.reconstruction.to(device)

    # def forward(self, fixed, moving):
    #     flows = []
    #     stem_results = []
    #     # Affine registration
    #     flow = self.stems[0](fixed, moving)
    #     stem_results.append(self.reconstruction(moving, flow))
    #     flows.append(flow)
    #     for model in self.stems[1:]: # cascades
    #         # registration between the fixed and the warped from last cascade
    #         flow = model(fixed, stem_results[-1])
    #         stem_results.append(self.reconstruction(stem_results[-1], flow))
    #         flows.append(flow)
    #
    #     return stem_results, flows

    # def forward(self, fixed, moving):
    #     flows = []
    #     stem_results = []
    #     # Affine registration
    #     flow = self.stems[0](fixed, moving)
    #     stem_results.append(self.reconstruction(moving, flow))
    #     flows.append(flow)
    #
    #     for model in self.stems[1:]: # cascades
    #         # registration between the fixed and the warped from last cascade
    #         flow = model(fixed, stem_results[-1])
    #         stem_results.append(self.reconstruction(stem_results[-1], flow))
    #         flows.append(flow)
    #
    #
    #     ddf = flows[0]
    #     for flow in flows[1:]:
    #         ddf = self.reconstruction(ddf, flow) + flow
    #
    #     m2f = self.reconstruction(moving, ddf)
    #
    #     return stem_results, flows, ddf, m2f

    def forward(self, fixed, moving):
        flows = []
        stem_results = []
        # Affine registration
        x = torch.cat((moving, fixed), dim=1)
        flow = self.stems[0](x)

        # flow = self.stems[0](fixed, moving)

        stem_results.append(self.reconstruction(moving, flow))
        flows.append(flow)
        ddf = flows[0]

        for model in self.stems[1:]: # cascades
            # registration between the fixed and the warped from last cascade
            y = torch.cat((stem_results[-1], fixed), dim=1)
            flow = model(y)
            # flow = flow * (1 / 5)    # here, 5 is n_cascades
            # flow = model(fixed, stem_results[-1])

            ddf = self.reconstruction(ddf, flow) + flow

            # ddf = max_norm(ddf)
            # ddf = torch.clamp(ddf, -1, 1)

            stem_results.append(self.reconstruction(moving, ddf))
            flows.append(flow)

        return stem_results, flows, ddf








