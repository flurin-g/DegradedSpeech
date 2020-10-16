import os
from shutil import copyfile, copytree

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import soundfile as sf


def write_file(edited_file: np.ndarray, dst: str, sample_rate: int):
    sf.write(dst, edited_file, sample_rate)


def create_copy_and_apply(func: callable):
    def copy_and_apply(src, dst, *, follow_symlinks=True):
        if os.path.isdir(dst):
            dst = os.path.join(dst, os.path.basename(src))
        elif src.endswith('.txt'):
            copyfile(src, dst)
        else:
            edited_file, sample_rate = func(src)
            write_file(edited_file, dst, sample_rate)
        return dst

    return copy_and_apply


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    copytree(cfg.source_directory, cfg.target_directory)


if __name__ == "__main__":
    main()
