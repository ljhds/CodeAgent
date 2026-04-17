import hydra

from verl.trainer.main_ppo import run_ppo


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    """Additive entrypoint for coder1 AgentLoop migration."""
    run_ppo(config)


if __name__ == "__main__":
    main()
