import logging
import os
import pickle
import pprint

import tyro
from brax.io import model

import wandb
from jaxgcrl.utils.config import Config
from jaxgcrl.utils.env import MetricsRecorder, create_env


def main(config: Config):
    """Train an agent and save checkpoints.

    Creates the following directory structure:
        ./runs/
            run_{exp_name}_s_{seed}/
                args.pkl      # run and agent configuration
                ckpt/         # step_{env_steps}.pkl after every evaluation, and final
    """
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s|  %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    info = {**vars(config.run), **vars(config.agent)}
    info["agent"] = type(config.agent).__name__
    logging.info("Arguments:\n%s", pprint.pformat(info))

    exp_name = config.run.exp_name
    if exp_name == "run":
        agent_name = info["agent"].lower()
        if agent_name == "dcp" and info.get("use_direct_goal_direction", False):
            agent_name = "direct_dcp"
        exp_name = f"{agent_name}_{config.run.env}"
        config = config.replace(run=config.run.replace(exp_name=exp_name))

    wandb.init(
        project=config.run.wandb_project_name,
        group=config.run.wandb_group,
        name=f"{exp_name}_s{config.run.seed}",
        config=info,
        mode=config.run.wandb_mode if config.run.log_wandb else "disabled",
    )

    env = create_env(env_name=config.run.env, backend=config.run.backend)
    if config.run.eval_env:
        eval_env = create_env(env_name=config.run.eval_env, backend=config.run.backend)
    else:
        eval_env = env

    run_dir = f"./runs/run_{exp_name}_s_{config.run.seed}"
    ckpt_dir = run_dir + "/ckpt"
    os.makedirs(ckpt_dir, exist_ok=True)
    if config.run.checkpoint_logdir is None:
        config = config.replace(run=config.run.replace(checkpoint_logdir=ckpt_dir))
    with open(run_dir + "/args.pkl", "wb") as f:
        pickle.dump(vars(config), f)

    metrics_to_collect = [
        "eval/episode_success",
        "eval/episode_success_any",
        "eval/episode_dist",
        "eval/episode_reward",
        "training/actor_loss",
        "training/alpha_loss",
        "training/log_alpha",
        "training/entropy",
        "training/critic_loss",
        "training/categorical_accuracy",
        "training/logits_pos",
        "training/logits_neg",
        "training/logsumexp",
        "training/latent_distance",
        "training/sps",
        "training/buffer_current_size",
    ]

    metrics_recorder = MetricsRecorder(
        config.run.total_env_steps,
        metrics_to_collect,
        run_dir,
        exp_name,
        mode=config.run.wandb_mode,
    )

    _, params, _ = config.agent.train_fn(
        train_env=env,
        eval_env=eval_env,
        config=config.run,
        progress_fn=metrics_recorder.progress,
    )
    model.save_params(ckpt_dir + "/final", params)


def cli():
    tyro.cli(
        main,
        config=(
            tyro.conf.OmitArgPrefixes,
            tyro.conf.OmitSubcommandPrefixes,
            tyro.conf.ConsolidateSubcommandArgs,
        ),
    )


if __name__ == "__main__":
    cli()
