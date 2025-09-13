# spinup/__init__.py — TF-optional, PyTorch-first

# PyTorch algos (no TF required)
from spinup.algos.pytorch.ddpg.ddpg import ddpg as ddpg_pytorch
from spinup.algos.pytorch.ppo.ppo import ppo as ppo_pytorch
from spinup.algos.pytorch.sac.sac import sac as sac_pytorch
from spinup.algos.pytorch.td3.td3 import td3 as td3_pytorch
from spinup.algos.pytorch.trpo.trpo import trpo as trpo_pytorch
from spinup.algos.pytorch.vpg.vpg import vpg as vpg_pytorch

# Optional TF1 algos (guard everything; no top-level tf import)
try:
    from spinup.algos.tf1.ddpg.ddpg import ddpg as ddpg_tf1
except Exception:
    ddpg_tf1 = None
try:
    from spinup.algos.tf1.sac.sac import sac as sac_tf1
except Exception:
    sac_tf1 = None
try:
    from spinup.algos.tf1.td3.td3 import td3 as td3_tf1
except Exception:
    td3_tf1 = None
try:
    from spinup.algos.tf1.trpo.trpo import trpo as trpo_tf1
except Exception:
    trpo_tf1 = None
try:
    from spinup.algos.tf1.ppo.ppo import ppo as ppo_tf1
except Exception:
    ppo_tf1 = None
try:
    from spinup.algos.tf1.vpg.vpg import vpg as vpg_tf1
except Exception:
    vpg_tf1 = None

from spinup.utils.logx import Logger, EpochLogger
from spinup.version import __version__
