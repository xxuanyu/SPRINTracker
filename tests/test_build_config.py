import sys

from configs.util import load_super_config
from models.motip import build
from utils.misc import yaml_to_dict


config_path = sys.argv[1]
config = yaml_to_dict(config_path)
config = load_super_config(config, config["SUPER_CONFIG_PATH"])
model, _ = build(config)
print(type(model.trajectory_modeling).__name__, sum(p.numel() for p in model.parameters()))
