# Train
```python
python3 roboverse_learn/unitree_rl/train.py --task dof12_walking --sim isaacgym --num_envs 4096 --robot 'g1_dof12'
```

# Play
mkdir -p outputs/unitree_rl/g1_dof12_dof12_walking/2025_0901_052923/; cp roboverse_learn/unitree_rl/model_2500.pt outputs/unitree_rl/g1_dof12_dof12_walking/2025_0901_052923/; python ./roboverse_learn/unitree_rl/play.py --robot "g1_dof12" --load_run 2025_0901_052923 --checkpoint 2500 --task dof12_walking --sim mujoco
