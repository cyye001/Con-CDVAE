import os
from pathlib import Path
from typing import Optional

import dotenv
from omegaconf import DictConfig, OmegaConf

def get_env(env_name: str, default: Optional[str] = None) -> str:
    """
    Safely read an environment variable.
    Raises errors if it is not defined or it is empty.

    :param env_name: the name of the environment variable
    :param default: the default (optional) value for the environment variable

    :return: the value of the environment variable
    """
    a = os.environ
    if env_name not in os.environ:
        if default is None:
            raise KeyError(
                f"{env_name} not defined and no default value is present!")
        return default

    env_value: str = os.environ[env_name]
    if not env_value:
        if default is None:
            raise ValueError(
                f"{env_name} has yet to be configured and no default value is present!"
            )
        return default

    return env_value


def load_envs(env_file: Optional[str] = None) -> None:
    """
    Load all the environment variables defined in the `env_file`.
    This is equivalent to `. env_file` in bash.

    It is possible to define all the system specific variables in the `env_file`.

    :param env_file: the file that defines the environment variables to use. If None
                     it searches for a `.env` file in the project.
    """
    dotenv.load_dotenv(dotenv_path=env_file, override=True)


def param_statistics(model):
    # Total params
    total_params = sum(p.numel() for p in model.parameters())
    print("Total params:", total_params)

    # Trainable params
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable params:", trainable_params)

    # Space
    total_params_size = sum(p.numel() * p.element_size() for p in model.parameters())
    print("Used Space(MB):", round(total_params_size/1024/1024,2))

STATS_KEY: str = "stats"


# Load environment variables
load_envs()

# Set the cwd to the project root
PROJECT_ROOT: Path = Path(get_env("PROJECT_ROOT"))
assert (
    PROJECT_ROOT.exists()
), "You must configure the PROJECT_ROOT environment variable in a .env file!"

os.chdir(PROJECT_ROOT)