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
