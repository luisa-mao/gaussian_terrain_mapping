import torch
from torch import nn
import numpy as np
import cv2
from utils import get_covariance
from tqdm import tqdm

class Model:
    def __init__(self, points):
        # points is a list of tuples (x, y, yaw, opacity, rgb)
        # make all of these trainable tensors
        xy_tensor = torch.tensor([[float(x), float(y)] for (x, y), _, _, _ in points])
        yaws_tensor = torch.tensor([yaw for _, yaw, _, _ in points])
        opacities_tensor = torch.tensor([opacity for _, _, opacity, _ in points])
        self._xy = nn.Parameter(xy_tensor.requires_grad_(True))

        # idk what these are
        # self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        # self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        # self._scaling = nn.Parameter(scales.requires_grad_(True))

        self._rotation = nn.Parameter(yaws_tensor.requires_grad_(True))
        self._opacity = nn.Parameter(opacities_tensor.requires_grad_(True))
        self._rgb = nn.Parameter(torch.tensor([rgb for _, _, _, rgb in points]).requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor([[1.0, 1.0] for _ in points]).requires_grad_(True))


        # other things
        self.max_radii2D = torch.empty(0)
        self.xy_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.opacity_activation = torch.sigmoid
        # self.inverse_opacity_activation = inverse_sigmoid

        self.covariance_activation = get_covariance



    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self._rotation
    
    @property
    def get_xy(self):
        return self._xy
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    

    # things here can be fixed
    def training_setup(self):
        self.percent_dense = 0.01
        self.xy_gradient_accum = torch.zeros((self.get_xy.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xy.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xy], 'lr': 1.0, "name": "xy"},
            # {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            # {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._rgb], 'lr': 0.005, "name": "rgb"},
            {'params': [self._opacity], 'lr': 0.025, "name": "opacity"},
            {'params': [self._scaling], 'lr': 0.005, "name": "scaling"},
            {'params': [self._rotation], 'lr': 0.001, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        # if self.optimizer_type == "default":
        #     self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        # elif self.optimizer_type == "sparse_adam":
        #     try:
        #         self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
        #     except:
        #         # A special version of the rasterizer is required to enable sparse adam
        #         self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        # self.exposure_optimizer = torch.optim.Adam([self._exposure])

        # self.xy_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
        #                                             lr_final=training_args.position_lr_final*self.spatial_lr_scale,
        #                                             lr_delay_mult=training_args.position_lr_delay_mult,
        #                                             max_steps=training_args.position_lr_max_steps)
        
        # self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
        #                                                 lr_delay_steps=training_args.exposure_lr_delay_steps,
        #                                                 lr_delay_mult=training_args.exposure_lr_delay_mult,
        #                                                 max_steps=training_args.iterations)

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors
    def densification_postfix(self, new_xy, new_rgb, new_opacities, new_scaling, new_rotation, new_tmp_radii):
        d = {"xy": new_xy,
        # "f_dc": new_features_dc,
        # "f_rest": new_features_rest,
        "rgb": new_rgb,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xy = optimizable_tensors["xy"]
        # self._features_dc = optimizable_tensors["f_dc"]
        # self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._rgb = optimizable_tensors["rgb"]

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xy_gradient_accum = torch.zeros((self.get_xy.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xy.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xy.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xy.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        # selected_pts_mask = torch.logical_and(selected_pts_mask,
                                            #   torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        # Construct the 2D rotation matrices
        cos_yaws = torch.cos(yaws)
        sin_yaws = torch.sin(yaws)

        rots = torch.stack([
            torch.stack([cos_yaws, -sin_yaws], dim=-1),
            torch.stack([sin_yaws, cos_yaws], dim=-1)
        ], dim=-2)
        new_xy = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xy[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        # new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        # new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_rgb = self._rgb[selected_pts_mask].repeat(N,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        self.densification_postfix(new_xy, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        # look into the scaling? world space scaling?
        # selected_pts_mask = torch.logical_and(selected_pts_mask,
                                            #   torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xy = self._xy[selected_pts_mask]
        # new_features_dc = self._features_dc[selected_pts_mask]
        # new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_rgb = self._rgb[selected_pts_mask]

        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        self.densification_postfix(new_xy, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii)

    
    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        grads = self.xy_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.tmp_radii = radii
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            prune_mask = torch.logical_or(prune_mask, big_points_vs)
        self.prune_points(prune_mask)
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xy_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def render(self, canvas):
        h,w = canvas.shape[:2]
        n = self.get_xy.shape[0]
        covariances = self.covariance_activation(self.get_scaling, self.get_rotation)
        inverted_covariances = torch.inverse(covariances)
        # draw the gaussians on the canvas
        for i in range(w):
            for j in range(h):
                r, g, b = 0, 0, 0
                for k in range(n):
                    # get the pdf of the gaussian at this point
                    x, y = self.get_xy[k]
                    dx = i - x
                    dy = j - y
                    inv_cov = inverted_covariances[k]
                    # calculate the pdf
                    pdf = torch.exp(-0.5 * (dx * inv_cov[0, 0] * dx + dy * inv_cov[1, 1] * dy + 2 * inv_cov[0, 1] * dx * dy))
                    # multiply by the opacity
                    pdf = torch.mul(pdf, self.get_opacity[k])
                    # multiply by the rgb
                    r += pdf * self._rgb[k, 0]
                    g += pdf * self._rgb[k, 1]
                    b += pdf * self._rgb[k, 2]
                canvas[j, i, 0] = canvas[j, i, 0] + r
                canvas[j, i, 1] = canvas[j, i, 1] + g
                canvas[j, i, 2] = canvas[j, i, 2] + b 
        return canvas