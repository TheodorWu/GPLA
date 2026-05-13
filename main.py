import sys, os
from pathlib import Path
submodule_path = Path(__file__).parent / "openvla_mini"
sys.path.insert(0, str(submodule_path))
# needed only to avoid error for unhandled case of PRISMATIC_DATA_ROOT not being set
os.environ["PRISMATIC_DATA_ROOT"] = "/tmp/data"
os.environ["USE_TORCH"] = "1"
import hydra
from omegaconf import DictConfig #, OmegaConf

import transformers
if hasattr(transformers, 'video_utils'):
    transformers.image_utils.VideoInput = transformers.video_utils.VideoInput
    transformers.image_utils.make_batched_videos = transformers.video_utils.make_batched_videos

from utils import seed_all, BASE_IMAGE_PROCESSOR_FAST_DOCSTRING, BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS#, DIR_PATH
transformers.image_processing_utils_fast.BASE_IMAGE_PROCESSOR_FAST_DOCSTRING = BASE_IMAGE_PROCESSOR_FAST_DOCSTRING
transformers.image_processing_utils_fast.BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS = BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS


from model.initializer import Initializer


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):

    initializer = Initializer(cfg=cfg)
    initializer.init_wandb()

    # seed after randomly generating run name
    seed_all(cfg.training.get("seed", 42))

    strategy = initializer.init_strategy()

    strategy.train()
    strategy.test()

    strategy.cleanup()

if __name__=="__main__":
    main() # pylint: disable=no-value-for-parameter
