# Unitree RL Lib

## Training
```python
python3 roboverse_learn/unitree_rl/train.py --task "humanoid_walking" --sim "isaacgym" --num_envs 2 --robot "h1_wrist"
```

## Play
```python
python3 roboverse_learn/unitree_rl/play.py --task legged_walking --sim isaacgym --robot go2 --load_run 2025_0806_021440  --checkpoint 0
```

## Setup the rsl_rl_lib

```
git clone https://github.com/leggedrobotics/rsl_rl
cd rsl_rl
git checkout v1.0.2
pip install -e .
```

## Use the [public policy](https://github.com/unitreerobotics/unitree_rl_gym/tree/main/deploy/pre_train/g1) of Unitree to check the effectiveness of metasim

1. Copy the policy to "outputs/unitree_rl/g1_dof12_dof12_walking/pretrain/model_0.pt"
2. For Mujoco Evaluation
```bash
mamba activate metasim

python ./roboverse_learn/unitree_rl/play.py --robot "g1_dof12" --load_run pretrain --checkpoint 0  --task dof12_walking --jit_load true --reindex_actions true --sim mujoco
```
3. For Isaacgym Evaluation
```bash
mamba activate metasim_isaacgym

python ./roboverse_learn/unitree_rl/play.py --robot "g1_dof12" --load_run pretrain --checkpoint 0  --task dof12_walking --jit_load true --reindex_actions true --sim isaacgym
```
