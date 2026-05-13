import sys
import inspect
import random
from pathlib import Path
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.nn.parameter import is_lazy
from torch.utils.data import DataLoader
from torchvision.io import decode_image
import numpy as np
from transformers import set_seed, BatchFeature
from PIL import Image

DIR_PATH = Path(__file__).parent.resolve()

class DotDict(dict):
    """
    A dictionary wrapper that allows access to dictionary items using dot notation.

    Example:
        config = DotDict({'model': {'lr': 0.01, 'batch_size': 32}})
        print(config.model.lr)  # 0.01
        config.model.epochs = 100
        print(config.model.epochs)  # 100
    """

    def __init__(self, d=None):
        if d is None:
            d = {}
        super().__init__()
        for key, value in d.items():
            self[key] = self._convert(value)

    def _convert(self, value):
        """Recursively convert nested dicts to DotDict instances."""
        if isinstance(value, dict) and not isinstance(value, DotDict):
            return DotDict(value)
        elif isinstance(value, list):
            return [self._convert(item) for item in value]
        elif isinstance(value, tuple):
            return tuple(self._convert(item) for item in value)
        return value

    def __getattr__(self, key):
        """Allow access via dot notation (config.key)."""
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        """Allow setting via dot notation (config.key = value)."""
        self[key] = self._convert(value)

    def __delattr__(self, key):
        """Allow deletion via dot notation (del config.key)."""
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")

    def __setitem__(self, key, value):
        """Override to convert nested dicts when setting items."""
        super().__setitem__(key, self._convert(value))

    def update(self, other):
        """Override update to convert nested dicts."""
        for key, value in other.items():
            self[key] = value

def batch_of_dicts_to_outer_dict(batch, padding_side='right'):
    if (not isinstance(batch, list)) or len(batch) == 0:
        return batch

    outer = {}
    first_dictionary = batch[0]

    if not (isinstance(first_dictionary, dict) or isinstance(first_dictionary, BatchFeature)):
        return batch

    for k in first_dictionary.keys():
        outer[k] = [dictionary[k] for dictionary in batch]
        if torch.is_tensor(outer[k][0]):
            outer[k] = pad_sequence(outer[k], batch_first=True, padding_side=padding_side, padding_value=0)
    return outer

def decode_image_sequence(images):
    decoded_images = torch.stack([ decode_image(torch.tensor(np.frombuffer(img, dtype=np.uint8))) for img in images ])
    return decoded_images

def seed_all(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    set_seed(seed)

def test_gpu_availability(cfg=None):
    print(f"Using torch {torch.__version__}", file=sys.stdout)
    print(f"Cuda available: {torch.cuda.is_available()}", file=sys.stdout)
    print('__CUDNN VERSION:', torch.backends.cudnn.version(), file=sys.stdout)
    print('Available devices ', torch.cuda.device_count(), file=sys.stdout)
    print('Current cuda device ', torch.cuda.current_device(), file=sys.stdout)
    print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    if cfg and cfg.training.device == "gpu" and torch.cuda.is_available():
        print("Using GPU", file=sys.stdout)
        device = torch.device("cuda")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        print("Using CPU", file=sys.stdout)
        device = torch.device("cpu")

    return device

def get_dtype(dt):
    dtypes = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return dtypes.get(dt, torch.bfloat16)

def prepare_inputs_for_inference(input_dict):
    for k in ["labels"]:
        if k in input_dict:
            del input_dict[k]
    return input_dict

def printable_params(cls):
    def print_trainable_parameters(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad and not is_lazy(p))
        total = sum(p.numel() for p in self.parameters() if not is_lazy(p))
        print(f"trainable: {trainable:,} || all params: {total:,} || trainable%: {trainable/(total+1e-8)*100:.4f}")
    cls.print_trainable_parameters = print_trainable_parameters
    return cls

def preprocess_image_for_display(image, target_size=(64, 64)):
    """
    Preprocess image for display as thumbnail in the plot.

    Args:
        image: Image tensor, PIL Image, or numpy array
        target_size: Tuple of (width, height) for thumbnail

    Returns:
        PIL Image ready for display
    """
    # Convert tensor to numpy if needed
    if torch.is_tensor(image):
        if image.dim() == 4:  # Batch dimension
            image = image.squeeze(0)
        if image.dim() == 3 and image.shape[0] in [1, 3]:  # Channel first
            image = image.permute(1, 2, 0)
        image = image.cpu().numpy()

    # Normalize if values are in [0, 1] or [-1, 1]
    if image.dtype == np.float32 or image.dtype == np.float64:
        if image.min() >= -1 and image.max() <= 1:
            image = (image + 1) / 2  # Convert from [-1, 1] to [0, 1]
        if image.max() <= 1:
            image = (image * 255).astype(np.uint8)

    # Convert to PIL Image
    if isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[2] == 1:  # Grayscale with channel dim
            image = image.squeeze(2)
        image = Image.fromarray(image)

    # Resize to target size
    image = image.resize(target_size, Image.Resampling.LANCZOS)

    return image

class ResetAwareDataLoader(DataLoader):
    def __init__(self, *args, cleanup_frequency=1000, **kwargs):
        super().__init__(*args, **kwargs)
        self.cleanup_frequency = cleanup_frequency
        self.batch_count = 0

    def __iter__(self):
        # Reset dataset if it has the method
        if hasattr(self.dataset, 'reset_episodes'):
            self.dataset.reset_episodes()

        self.batch_count = 0  # Reset batch counter for new epoch
        return self._iter_with_cleanup()

    def _iter_with_cleanup(self):
        for batch in super().__iter__():
            self.batch_count += 1

            # Periodic cleanup of exhausted episodes
            if (self.batch_count % self.cleanup_frequency == 0 and
                hasattr(self.dataset, 'cleanup_exhausted_episodes')):
                self.dataset.cleanup_exhausted_episodes()

            yield batch

        # Reset dataset if it has the method
        if hasattr(self.dataset, 'reset_episodes'):
            self.dataset.reset_episodes()


def check_tensor(tensor, name):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"❌ {name}: NaN/Inf detected!")
        print(f"   Shape: {tensor.shape}")
        print(f"   Range: [{tensor.min():.6f}, {tensor.max():.6f}]")
        print(f"   Mean: {tensor.mean():.6f}, Std: {tensor.std():.6f}")
        import pdb; pdb.set_trace() # pylint: disable=multiple-statements
    else:
        print(f"✅ {name}: OK - Range: [{tensor.min():.3f}, {tensor.max():.3f}], Mean: {tensor.mean():.3f}")

def debug_batch_data(batch):
    # Check input ranges
    pixel_values = batch.get('pixel_values')
    actions = batch.get('action')

    print(f"Pixel values: shape={pixel_values.shape}, range=[{pixel_values.min():.3f}, {pixel_values.max():.3f}]")
    print(f"Actions: shape={actions.shape}, range=[{actions.min():.3f}, {actions.max():.3f}]")

    # Check for unusual patterns
    if pixel_values.std() < 0.01:
        print("❌ Pixel values have very low variance - possible preprocessing issue")

    if actions.std() < 0.001:
        print("❌ Actions have very low variance - possible data issue")

    # Check for extreme values
    if pixel_values.max() > 10 or pixel_values.min() < -10:
        print("❌ Pixel values outside expected range")

def has_parameter(func, param_name):
    """Check if function has a parameter by name."""
    sig = inspect.signature(func)
    return param_name in sig.parameters

BASE_IMAGE_PROCESSOR_FAST_DOCSTRING = r"""

    Args:
        do_resize (`bool`, *optional*, defaults to `self.do_resize`):
            Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by the
            `do_resize` parameter in the `preprocess` method.
        size (`dict`, *optional*, defaults to `self.size`):
            Size of the output image after resizing. Can be overridden by the `size` parameter in the `preprocess`
            method.
        default_to_square (`bool`, *optional*, defaults to `self.default_to_square`):
            Whether to default to a square image when resizing, if size is an int.
        resample (`PILImageResampling`, *optional*, defaults to `self.resample`):
            Resampling filter to use if resizing the image. Only has an effect if `do_resize` is set to `True`. Can be
            overridden by the `resample` parameter in the `preprocess` method.
        do_center_crop (`bool`, *optional*, defaults to `self.do_center_crop`):
            Whether to center crop the image to the specified `crop_size`. Can be overridden by `do_center_crop` in the
            `preprocess` method.
        crop_size (`Dict[str, int]` *optional*, defaults to `self.crop_size`):
            Size of the output image after applying `center_crop`. Can be overridden by `crop_size` in the `preprocess`
            method.
        do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the
            `do_rescale` parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `self.rescale_factor`):
            Scale factor to use if rescaling the image. Only has an effect if `do_rescale` is set to `True`. Can be
            overridden by the `rescale_factor` parameter in the `preprocess` method.
        do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
            Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
            method. Can be overridden by the `do_normalize` parameter in the `preprocess` method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method. Can be
            overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
            Can be overridden by the `image_std` parameter in the `preprocess` method.
        do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
            Whether to convert the image to RGB.
        return_tensors (`str` or `TensorType`, *optional*, defaults to `self.return_tensors`):
            Returns stacked tensors if set to `pt, otherwise returns a list of tensors.
        data_format (`ChannelDimension` or `str`, *optional*, defaults to `self.data_format`):
            Only `ChannelDimension.FIRST` is supported. Added for compatibility with slow processors.
        input_data_format (`ChannelDimension` or `str`, *optional*, defaults to `self.input_data_format`):
            The channel dimension format for the input image. If unset, the channel dimension format is inferred
            from the input image. Can be one of:
            - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
            - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
        device (`torch.device`, *optional*, defaults to `self.device`):
            The device to process the images on. If unset, the device is inferred from the input images."""

BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS = r"""
    Preprocess an image or batch of images.

    Args:
        images (`ImageInput`):
            Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
            passing in images with pixel values between 0 and 1, set `do_rescale=False`.
        do_resize (`bool`, *optional*, defaults to `self.do_resize`):
            Whether to resize the image.
        size (`Dict[str, int]`, *optional*, defaults to `self.size`):
            Describes the maximum input dimensions to the model.
        resample (`PILImageResampling` or `InterpolationMode`, *optional*, defaults to `self.resample`):
            Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only
            has an effect if `do_resize` is set to `True`.
        do_center_crop (`bool`, *optional*, defaults to `self.do_center_crop`):
            Whether to center crop the image.
        crop_size (`Dict[str, int]`, *optional*, defaults to `self.crop_size`):
            Size of the output image after applying `center_crop`.
        do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
            Whether to rescale the image.
        rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
            Rescale factor to rescale the image by if `do_rescale` is set to `True`.
        do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
            Whether to normalize the image.
        image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
            Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.
        image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
            Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to
            `True`.
        do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
            Whether to convert the image to RGB.
        return_tensors (`str` or `TensorType`, *optional*, defaults to `self.return_tensors`):
            Returns stacked tensors if set to `pt, otherwise returns a list of tensors.
        data_format (`ChannelDimension` or `str`, *optional*, defaults to `self.data_format`):
            Only `ChannelDimension.FIRST` is supported. Added for compatibility with slow processors.
        input_data_format (`ChannelDimension` or `str`, *optional*, defaults to `self.input_data_format`):
            The channel dimension format for the input image. If unset, the channel dimension format is inferred
            from the input image. Can be one of:
            - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
            - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
        device (`torch.device`, *optional*, defaults to `self.device`):
            The device to process the images on. If unset, the device is inferred from the input images."""
