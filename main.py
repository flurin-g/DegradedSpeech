import os
import pathlib
from shutil import copyfile, copytree

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import soundfile as sf

from add_noise import AddNoise
from freq_filter import FreqFilter
from reverberate import create_ir_dataframe, Reverberate

CWD = pathlib.Path(__file__).parent.absolute()


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

    source_directory = f'{CWD}/{cfg.source_directory}'

    if cfg.task == "add_noise":
        fx = AddNoise(cfg, CWD)
        target_dir = f'{CWD}/{cfg.target_directory}/noise'
    elif cfg.task == "reverberate":
        fx = Reverberate(cfg, CWD)
        target_dir = f'{CWD}/{cfg.target_directory}/reverb'
    elif cfg.task == "freq_filter":
        fx = FreqFilter(cfg, CWD)
        target_dir = f'{CWD}/{cfg.target_directory}/freq_filter'
    elif cfg.task == "create_ir_meta":
        df_ir = create_ir_dataframe(meta_path=cfg.ir_meta_path,
                                    ir_path=cfg.impulse_response_path,
                                    cwd=CWD)
        print(df_ir.head())
        return

    apply_fx = create_copy_and_apply(fx)

    copytree(source_directory, target_dir, copy_function=apply_fx, dirs_exist_ok=True)


if __name__ == "__main__":
    main()
