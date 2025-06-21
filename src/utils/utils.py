import wandb 

def initialize_wandb(api_key, project_name, exp_name, config, group):
    wandb.login(key=api_key)
    return wandb.init(
        project=project_name,
        name=exp_name,
        config=config,
        group=group
    )