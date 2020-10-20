import os
import pathlib
from shutil import copyfile, copytree

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import soundfile as sf

from add_noise import AddNoise
from dtln_de_noise import DtlnDeNoise
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


def get_task(cfg):
    target_base = f'{CWD}/{cfg.target_directory}'
    targets = {
        "add_noise": (f'{target_base}/noise', AddNoise),
        "reverberate": (f'{target_base}/reverb', Reverberate),
        "freq_filter": (f'{target_base}/freq_filter', FreqFilter),
    }

    if cfg.task == "de_noise":
        task = (f'{target_base}/de_noised-{cfg.de_noise}', DtlnDeNoise)
        source_directory = targets[cfg.de_noise][0]
    else:
        task = targets[cfg.task]
        source_directory = f'{CWD}/{cfg.source_directory}'
    target_directory = task[0]
    fx = task[1]

    return fx, source_directory, target_directory


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    if cfg.task == "create_ir_meta":
        df_ir = create_ir_dataframe(meta_path=cfg.ir_meta_path,
                                    ir_path=cfg.impulse_response_path,
                                    cwd=CWD)
        print(df_ir.head())
    else:
        fx, source_directory, target_directory = get_task(cfg)

        apply_fx = create_copy_and_apply(fx(cfg, CWD))

        copytree(source_directory, target_directory, copy_function=apply_fx, dirs_exist_ok=True)


if __name__ == "__main__":
    main()
