from metasim.task.base import BaseTaskEnv

class RslRLWrapper:
    def __init__(self, env: BaseTaskEnv):
        self.env = env


class SB3Wrapper:
    def __init__(self, env: BaseTaskEnv):
        self.env = env


class RlLibWrapper:
    def __init__(self, env: BaseTaskEnv):
        self.env = env


class Runner:
    def __init__(self):
        pass

    def learn():
        pass

def make_runner(env:BaseTaskEnv=None,
                train_cfg=None,
                lib:str='rsl') -> Runner:
    if lib == 'rsl':
        use_wandb = args.use_wandb
        if use_wandb:
            wandb.init(project=args.wandb, name=args.run_name)

        log_dir = get_log_dir(args, scenario, args.log_dir)
        task_wrapper = get_class(args.task, suffix="Task")
        task_env = task_wrapper(scenario)
        # dump snapshot of training config
        task_path = f"roboverse_learn/unitree_rl/tasks/{task_config.task_name}.py"
        if not os.path.exists(task_path):
            log.error(f"Task path {task_path} does not exist, please check your task name in config carefully")
            return
        shutil.copy2(task_path, log_dir)

        try:
            ppo_runner = OnPolicyRunner(
                env=task_env,
                train_cfg=task_env.train_cfg,
                device=task_env.device,
                log_dir=log_dir,
                # wandb=use_wandb,
                args=args,
            )
        except Exception as e:
            ppo_runner = OnPolicyRunner(
                env=task_env,
                train_cfg=task_env.train_cfg,
                device=task_env.device,
                log_dir=log_dir,
                # wandb=use_wandb,
                # args=args,
            )
        if args.load_run:
            resume_dir = get_log_dir(args, scenario, args.load_run)
            ppo_runner.load(resume_dir + f"/model_{args.checkpoint}.pt")
        ppo_runner.learn(num_learning_iterations=task_config.ppo_cfg.runner.max_iterations, init_at_random_ep_len=True)
