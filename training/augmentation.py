import random
import torch
import torch.nn.functional as F

class Augmentator:
    def __init__(self):
        self.p_color = 0.7
        self.p_geometric = 0.6
        self.p_noise = 0.4

    def augment_batch(self, batch):
        """Apply data augmentation strategies"""
        augmented_batch = {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}

        # Vision augmentations
        if 'frames' in batch:
            augmented_batch['frames'] = self.augment_frames(batch['frames'])

        # Action augmentations
        if 'action' in batch:
            augmented_batch['action'] = self.augment_actions(batch['action'])

        return augmented_batch

    def augment_frames(self, frames):
        """
        Frame augmentation strategies that preserve left/right directions
        Args:
            frames: Tensor of shape (B, C, T, H, W) or (B, C, H, W)
        Returns:
            Augmented frames with same shape
        """

        convert = frames.dtype == torch.uint8
        if convert:
            frames = frames.float() / 255.0
        # Apply augmentations with certain probabilities
        frames = self.color_augmentation(frames, p=self.p_color)
        frames = self.geometric_augmentation(frames, p=self.p_geometric)
        # frames = self.noise_augmentation(frames, p=self.p_noise)
        if convert:
            frames = (frames * 255).byte()
        return frames

    def geometric_augmentation(self, frames, p):
        """Geometric augmentations that preserve left/right"""
        if random.random() > p:
            return frames

        original_shape = frames.shape
        if len(frames.shape) == 5:  # (B, C, T, H, W)
            B, C, T, H, W = frames.shape
            frames = frames.reshape(B * T, C, H, W)

        # Random crop and resize (maintains aspect ratio awareness)
        if random.random() < 0.6:
            crop_ratio = random.uniform(0.85, 0.95)
            H, W = frames.shape[-2:]
            new_H, new_W = int(H * crop_ratio), int(W * crop_ratio)

            top = random.randint(0, H - new_H)
            left = random.randint(0, W - new_W)

            frames = frames[:, :, top:top+new_H, left:left+new_W]
            frames = F.interpolate(frames, size=(H, W), mode='bilinear', align_corners=False)

        # Vertical translation
        if random.random() < 0.4:
            max_shift = int(frames.shape[-2] * 0.1)  # 10% max shift
            shift = random.randint(-max_shift, max_shift)
            if shift != 0:
                frames = torch.roll(frames, shift, dims=-2)
                # Fill rolled areas with edge pixels
                if shift > 0:
                    # Shifted down: fill top with first valid row
                    frames[..., :shift, :] = frames[..., shift:shift+1, :].expand_as(frames[..., :shift, :])
                else:
                    # Shifted up: fill bottom with last valid row
                    frames[..., shift:, :] = frames[..., shift-1:shift, :].expand_as(frames[..., shift:, :])

        # Horizontal translation (left/right shift)
        if random.random() < 0.4:
            max_shift = int(frames.shape[-1] * 0.1)  # 10% max shift
            shift = random.randint(-max_shift, max_shift)
            if shift != 0:
                frames = torch.roll(frames, shift, dims=-1)
                # Fill rolled areas with edge pixels
                if shift > 0:
                    # Shifted right: fill left with first valid column
                    frames[..., :shift] = frames[..., shift:shift+1].expand_as(frames[..., :shift])
                else:
                    # Shifted left: fill right with last valid column
                    frames[..., shift:] = frames[..., shift-1:shift].expand_as(frames[..., shift:])

        # Scale (zoom in/out)
        if random.random() < 0.3:
            scale_factor = random.uniform(0.9, 1.1)
            H, W = frames.shape[-2:]
            frames = F.interpolate(frames, scale_factor=scale_factor, mode='bilinear', align_corners=False)
            frames = F.interpolate(frames, size=(H, W), mode='bilinear', align_corners=False)

        # Restore original shape
        if len(original_shape) == 5:
            frames = frames.reshape(B, C, T, H, W)

        return frames

    def noise_augmentation(self, frames, p):
        """Add various types of noise"""
        if random.random() > p:
            return frames

        # Gaussian noise
        if random.random() < 0.6:
            noise_std = random.uniform(0.01, 0.05)
            noise = torch.randn_like(frames) * noise_std
            frames = torch.clamp(frames + noise, 0, 1)

        return frames


    def color_augmentation(self, frames, p):
        """Color-based augmentations"""
        if random.random() > p:
            return frames

        original_shape = frames.shape
        # Flatten temporal dimension if present
        if len(frames.shape) == 5:  # (B, C, T, H, W)
            B, C, T, H, W = frames.shape
            frames = frames.reshape(B * T, C, H, W)

        # Brightness adjustment
        if random.random() < 0.5:
            brightness_factor = random.uniform(0.8, 1.2)
            frames = torch.clamp(frames * brightness_factor, 0, 1)

        # Contrast adjustment
        if random.random() < 0.5:
            contrast_factor = random.uniform(0.8, 1.2)
            mean = frames.mean(dim=(2, 3), keepdim=True)
            frames = torch.clamp((frames - mean) * contrast_factor + mean, 0, 1)

        # Saturation adjustment (for RGB)
        if frames.shape[1] == 3 and random.random() < 0.5:
            saturation_factor = random.uniform(0.8, 1.2)
            gray = frames.mean(dim=1, keepdim=True)
            frames = torch.clamp((frames - gray) * saturation_factor + gray, 0, 1)

        # Restore original shape
        if len(original_shape) == 5:
            frames = frames.reshape(B, C, T, H, W)

        return frames


    def augment_actions(self, actions):
        """
        Action augmentation strategies
        Args:
            actions: Tensor of shape (B, action_dim) or (B, seq_len, action_dim)
        Returns:
            Augmented actions with same shape
        """
        # Apply augmentations with certain probabilities
        actions = self.noise_augmentation_actions(actions, p=self.p_noise)
        # actions = self.magnitude_augmentation(actions, p=0.5)
        # actions = self.directional_augmentation(actions, p=0.3)

        return actions

    def noise_augmentation_actions(self, actions, p=0.6):
        """Add various types of noise to actions"""
        if random.random() > p:
            return actions

        # Gaussian noise (your original approach, but with adaptive scaling)
        if random.random() < 0.7:
            # Adaptive noise based on action magnitude
            action_std = actions.std(dim=-1, keepdim=True) + 1e-8
            noise_scale = random.uniform(0.005, 0.02)
            noise = torch.randn_like(actions) * action_std * noise_scale
            actions = actions + noise

        return actions

    def magnitude_augmentation(self, actions, p=0.5):
        """Augment action magnitudes while preserving directions"""
        if random.random() > p:
            return actions

        # Global scaling
        if random.random() < 0.6:
            scale_factor = random.uniform(0.8, 1.2)
            actions = actions * scale_factor

        return actions

    def directional_augmentation(self, actions, p=0.3):
        """Augment action directions while being careful with spatial semantics"""
        if random.random() > p:
            return actions

        # Small directional perturbations
        if random.random() < 0.6:
            # Add small random directional noise
            direction_noise = torch.randn_like(actions) * 0.05
            actions = actions + direction_noise

        # Coordinate system perturbations (small rotations for 2D/3D actions)
        if actions.shape[-1] >= 2 and random.random() < 0.3:
            # Small rotation for 2D actions (x, y components)
            if actions.shape[-1] >= 2:
                angle = random.uniform(-0.1, 0.1)  # Small rotation angle
                angle_t = torch.tensor(angle, device=actions.device, dtype=actions.dtype)
                cos_a, sin_a = torch.cos(angle_t), torch.sin(angle_t)

                # Apply 2D rotation to first two dimensions
                x = actions[..., 0]
                y = actions[..., 1]

                actions[..., 0] = cos_a * x - sin_a * y
                actions[..., 1] = sin_a * x + cos_a * y

        return actions
