# Unitree RL

Train and deploy locomotion policies for Unitree robots across three stages:
- Training in IsaacGym
- Sim2Sim evaluation in MuJoCo
- Real-world deployment (networked controller)

Well Supported robots: `g1_dof29` (full-body with dexterous hands) and `g1_dof12` (full-body without hands).


## Environment setup

Install the RL library dependency (rsl_rl v3.1.1) from source:
```
git clone https://github.com/leggedrobotics/rsl_rl
cd rsl_rl
git checkout v3.1.1
pip install -e .
```


## Training (IsaacGym)

General form:
```
python roboverse_learn/rl/unitree_rl/main.py \
  --task <your_task> \
  --sim isaacgym \
  --num_envs 8192 \
  --robot <your_robot>
```

Examples:
- G1 humanoid walking (IsaacSim):
```
python roboverse_learn/rl/unitree_rl/main.py --task walk_g1_dof29 --sim isaacsim --num_envs 8192 --robot g1_dof29
```
- G1Dof12 walking (IsaacGym):
```
python roboverse_learn/rl/unitree_rl/main.py --task walk_g1_dof12 --sim isaacgym --num_envs 8192 --robot g1_dof12 --run_name g1dof12_walk
```

Outputs and checkpoints are saved to:
```
outputs/unitree_rl/<robot>_<task>/<datetime>/
```
Each checkpoint is named `model_<iter>.pt`.

## Evaluation / Play

You can evaluate trained policies in both MuJoCo, Isaacsim (sim2sim) and IsaacGym. In evaluation, `main.py` also exports the jit version policy to the directory `outputs/unitree_rl/<robot>_<task>/<datetime>/exported/model_exported_jit.pt`, which can be further used for real-world deployment.

IsaacGym evaluation:
```
python roboverse_learn/rl/unitree_rl/main.py \
  --task walk_g1_dof29 \
  --sim isaacgym \
  --num_envs 1 \
  --robot g1_dof29 \
  --load_run <datetime_from_outputs> \
  --checkpoint <iter> \
  --eval
```

MuJoCo evaluation (e.g., DOF12 with public policy):
```
python roboverse_learn/rl/unitree_rl/main.py \
  --checkpoint <iter> \
  --task dof12_walking \
  --sim mujoco \
  --robot g1_dof12 \
  --load_run <datetime_from_outputs> \
  --eval
```


## Real-World deployment

First please install the `unitree_sdk2_python` package:
```
cd third_party
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
pip install -e .
```

Real-world deployment entry point:
```
python roboverse_learn/rl/unitree_rl/deploy/deploy_real.py <network_interface> <robot_yaml>
```
Example:
```
python roboverse_learn/rl/unitree_rl/deploy/deploy_real.py eno1 g1_dof29_dex3.yaml
```
where you should modify the corresponding `yaml` file in `roboverse_learn/rl/unitree_rl/deploy/configs`, setting the `policy_path` to the exported jit policy.
This will initialize the real controller and stream commands to the robot. Ensure your networking and safety interlocks are correctly configured.

## Command-line arguments

The most relevant flags (see `helper/utils.py`):
- `--task` (str): Task name. CamelCase or snake_case accepted. Examples: `humanoid_walking`, `dof12_walking`.
- `--robot` (str): Robot identifier. Common: `g1_dof29_dex3`, `g1_dof12`.
- `--num_envs` (int): Number of parallel environments.
- `--sim` (str): Simulator. Supported: `isaacgym` (training), `mujoco` (evaluation).
- `--run_name` (str): Required run tag for training logs/checkpoints.
- `--learning_iterations` (int): Number of learning iterations (default 15000).
- `--resume` (flag): Resume training from a checkpoint in the specified run.
- `--load_run` (str): Run directory (datetime or name) under `outputs/...` for loading.
- `--checkpoint` (int): Which checkpoint to load. `-1` loads the latest.
- `--headless` (flag): Headless rendering (IsaacGym).
- `--use_wandb` (flag) and `--wandb` (str): Enable Weights & Biases and set project name.
- `--jit_load` (bool): Load a TorchScript-exported policy.
- `--reindex_actions` (bool): Reindex actions from alphabetical order to default order to match simulator/robot joint order if needed.

Notes:
- Checkpoints: `outputs/unitree_rl/<robot>_<task>/<run_name or datetime>/model_<iter>.pt`
- Exported JIT model (when used): `outputs/unitree_rl/<robot>_<task>/<run_name or datetime>/exported/model_exported_jit.pt`
