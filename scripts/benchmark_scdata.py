import hydra
from omegaconf import DictConfig
import conceptlab as clab
import wandb
import omegaconf
from conceptlab.utils import helpers
import numpy as np

@hydra.main(config_path="../fm_config/", config_name="general.yaml")
def main(cfg: DictConfig):

    wandb.config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )

    wandb_name = cfg.wandb.experiment + "_" + helpers.timestamp()
    wandb_entity = cfg.wandb.get("entity", None)

    run = wandb.init(project=cfg.wandb.project, name=wandb_name, entity=wandb_entity)

    dataset = hydra.utils.instantiate(cfg.data)
    model = hydra.utils.instantiate(cfg.model)
    
    adata, adata_train, adata_test, adata_inter =  dataset.get_anndatas()
    
    model.train(adata_train.copy())
    adata_preds = model.predict_intervention(adata_inter.copy(), hold_out_label = dataset.hold_out_label, concepts_to_flip = dataset.concepts_to_flip)

    if isinstance(adata_train.X,np.ndarray):
        x_baseline = adata_train.X
        x_target = adata_test.X
    else:
        x_baseline = adata_train.X.toarray()
        x_target = adata_test.X.toarray()

    mmd_score = clab.evaluation.interventions.evaluate_intervention_mmd_with_target(
        x_train = x_baseline,
        x_ivn = adata_preds.X,
        x_target = x_target,
        labels_train = adata_train.obs[dataset.label_variable].values
        )
    
    de_score = clab.evaluation.interventions.evaluate_intervention_DE_with_target(
        x_train = x_baseline,
        x_ivn = adata_preds.X,
        x_target = x_target,
        genes_list = adata_train.var.index
    ) 
    
    print(mmd_score)
    print(de_score)
    for k, v in mmd_score.items():
        wandb.log({k: v})
    for k, v in de_score.items():
        wandb.log({k: v})
    wandb.finish()


if __name__ == "__main__":
    main()


