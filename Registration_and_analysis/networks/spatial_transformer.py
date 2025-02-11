from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialTransform(nn.Module):
    """
        This implementation was taken from:
        https://github.com/voxelmorph/voxelmorph/blob/master/voxelmorph/torch/layers.py
    """
    def __init__(self, size):
        super(SpatialTransform, self).__init__()

        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        new_locs = self.grid + flow

        shape = flow.shape[2:]

        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        # new_locs = torch.clamp(new_locs, -1, 1)

        return F.grid_sample(src, new_locs, mode='bilinear', align_corners=False)


class Dense3DSpatialTransformer(nn.Module):
    def __init__(self, device, padding=False):

        super(Dense3DSpatialTransformer, self).__init__()
        self.device = device
        self.padding = padding

    def forward(self, x):

        if len(x) > 2:
            raise Exception('Spatial Transformer must be called on a list of length 2. '
                            'First argument is the image, second is the flow field.')
        if len(x[1].shape) != 5 or x[1].shape[1] != 3:
            raise Exception('Flow field must be one 5D tensor with 3 channels. '
                            'Got: ' + str(list(x[1].shape)))
        return self._transform(x[0], x[1][:, 1, :, :, :], x[1][:, 0, :, :, :], x[1][:, 2, :, :, :])

    def _transform(self, I, dx, dy, dz):

        dx = torch.squeeze(dx, dim=1)
        dy = torch.squeeze(dy, dim=1)
        dz = torch.squeeze(dz, dim=1)

        batch_size = dx.shape[0]
        height = dx.shape[1]
        width = dx.shape[2]
        depth = dx.shape[3]

        # Convert dx and dy to absolute locations
        x_mesh, y_mesh, z_mesh = self._meshgrid(height, width, depth)
        x_mesh = torch.unsqueeze(x_mesh, 0)
        y_mesh = torch.unsqueeze(y_mesh, 0)
        z_mesh = torch.unsqueeze(z_mesh, 0)

        x_mesh = x_mesh.repeat(batch_size, 1, 1, 1)
        y_mesh = y_mesh.repeat(batch_size, 1, 1, 1)
        z_mesh = z_mesh.repeat(batch_size, 1, 1, 1)
        x_new = dx + x_mesh.to(dx.device)
        y_new = dy + y_mesh.to(dy.device)
        z_new = dz + z_mesh.to(dz.device)

        return self._interpolate(I, x_new, y_new, z_new)

    def _repeat(self, x, n_repeats):
        rep = torch.transpose(
            torch.unsqueeze(torch.ones(n_repeats), dim=1), 0, 1)
        rep = rep.int()
        x = torch.matmul(torch.reshape(x, (-1, 1)), rep)
        return torch.reshape(x, [-1])

    def _meshgrid(self, height, width, depth):
        x_t = torch.matmul(torch.ones([height, 1]),
                        torch.transpose(torch.unsqueeze(torch.linspace(0.0,
                                                                width-1.0, width), 1), 0, 1))
        y_t = torch.matmul(torch.unsqueeze(torch.linspace(0.0,
                                                   height-1.0, height), 1),
                        torch.ones([1, width]))

        x_t = torch.repeat_interleave(torch.unsqueeze(x_t, 2), depth, dim=2)
        y_t = torch.repeat_interleave(torch.unsqueeze(y_t, 2), depth, dim=2)

        z_t = torch.linspace(0.0, depth-1.0, depth)
        z_t = torch.unsqueeze(torch.unsqueeze(z_t, 0), 0)
        z_t = torch.repeat_interleave(torch.repeat_interleave(z_t, height, dim=0), width, dim=1)

        return x_t, y_t, z_t

    def _interpolate(self, im, x, y, z):
        if self.padding:
            im = F.pad(im, [1, 1, 1, 1, 1, 1], mode='constant', value=0) # im [B, C, H, W, D]

        num_batch = im.shape[0]
        height = im.shape[2]
        width = im.shape[3]
        depth = im.shape[4]
        channels = im.shape[1]

        out_height = x.shape[1]
        out_width = x.shape[2]
        out_depth = x.shape[3]

        x = torch.reshape(x, [-1])  #[batch_size * 128 * 128 * 128]
        y = torch.reshape(y, [-1])
        z = torch.reshape(z, [-1])

        padding_constant = 1.0 if self.padding else 0.0
        x = x.float() + padding_constant
        y = y.float() + padding_constant
        z = z.float() + padding_constant

        max_x = width - 1.0
        max_y = height - 1.0
        max_z = depth - 1.0

        x0 = torch.floor(x).int()
        x1 = x0 + 1
        y0 = torch.floor(y).int()
        y1 = y0 + 1
        z0 = torch.floor(z).int()
        z1 = z0 + 1

        x0 = torch.clamp(x0, 0, max_x)
        x1 = torch.clamp(x1, 0, max_x)
        y0 = torch.clamp(y0, 0, max_y)
        y1 = torch.clamp(y1, 0, max_y)
        z0 = torch.clamp(z0, 0, max_z)
        z1 = torch.clamp(z1, 0, max_z)

        dim3 = depth
        dim2 = depth*width
        dim1 = depth*width*height
        base = self._repeat(torch.arange(0, num_batch, dtype=torch.int32)*dim1,
                            out_height*out_width*out_depth)

        base_y0 = base.to(y0.device) + y0*dim2
        base_y1 = base.to(y1.device) + y1*dim2

        idx_a = base_y0 + x0*dim3 + z0
        idx_b = base_y1 + x0*dim3 + z0
        idx_c = base_y0 + x1*dim3 + z0
        idx_d = base_y1 + x1*dim3 + z0
        idx_e = base_y0 + x0*dim3 + z1
        idx_f = base_y1 + x0*dim3 + z1
        idx_g = base_y0 + x1*dim3 + z1
        idx_h = base_y1 + x1*dim3 + z1

        # Using indices to lookup pixels in the flat image and restore channels dim
        im_flat = torch.reshape(im.permute(0, 2, 3, 4, 1), [-1, channels]).float()

        Ia = torch.gather(im_flat, 0, torch.repeat_interleave(torch.unsqueeze(idx_a, dim=-1), channels, dim=-1).long().to(im_flat.device))
        Ib = torch.gather(im_flat, 0, torch.repeat_interleave(torch.unsqueeze(idx_b, dim=-1), channels, dim=-1).long().to(im_flat.device))
        Ic = torch.gather(im_flat, 0, torch.repeat_interleave(torch.unsqueeze(idx_c, dim=-1), channels, dim=-1).long().to(im_flat.device))
        Id = torch.gather(im_flat, 0, torch.repeat_interleave(torch.unsqueeze(idx_d, dim=-1), channels, dim=-1).long().to(im_flat.device))
        Ie = torch.gather(im_flat, 0, torch.repeat_interleave(torch.unsqueeze(idx_e, dim=-1), channels, dim=-1).long().to(im_flat.device))
        If = torch.gather(im_flat, 0, torch.repeat_interleave(torch.unsqueeze(idx_f, dim=-1), channels, dim=-1).long().to(im_flat.device))
        Ig = torch.gather(im_flat, 0, torch.repeat_interleave(torch.unsqueeze(idx_g, dim=-1), channels, dim=-1).long().to(im_flat.device))
        Ih = torch.gather(im_flat, 0, torch.repeat_interleave(torch.unsqueeze(idx_h, dim=-1), channels, dim=-1).long().to(im_flat.device))

        # And finally calculate interpolated values
        x1_f = x1.float()
        y1_f = y1.float()
        z1_f = z1.float()

        dx = x1_f - x
        dy = y1_f - y
        dz = z1_f - z

        wa = torch.unsqueeze((dz * dx * dy), 1).to(Ia.device)
        wb = torch.unsqueeze((dz * dx * (1 - dy)), 1).to(Ib.device)
        wc = torch.unsqueeze((dz * (1 - dx) * dy), 1).to(Ic.device)
        wd = torch.unsqueeze((dz * (1 - dx) * (1 - dy)), 1).to(Id.device)
        we = torch.unsqueeze(((1 - dz) * dx * dy), 1).to(Ie.device)
        wf = torch.unsqueeze(((1 - dz) * dx * (1 - dy)), 1).to(If.device)
        wg = torch.unsqueeze(((1 - dz) * (1 - dx) * dy), 1).to(Ig.device)
        wh = torch.unsqueeze(((1 - dz) * (1 - dx) * (1 - dy)), 1).to(Ih.device)

        output = wa*Ia + wb*Ib + wc*Ic + wd*Id + we*Ie + wf*If + wg*Ig + wh*Ih
        output = torch.reshape(output, [-1, out_height, out_width, out_depth, channels])

        return output.permute(0, 4, 1, 2, 3)
