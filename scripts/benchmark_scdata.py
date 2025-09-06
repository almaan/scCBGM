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

    if cfg.model.obsm_key == "X_pca":
        x_baseline_rec = adata_train.X
        x_target_rec = adata_test.X
        x_ivn_rec = adata_train.uns["pc_transform"].inverse_transform(adata_preds.obsm["X_pca"])

    
    mmd_score = clab.evaluation.interventions.evaluate_intervention_mmd_with_target(
        x_train = adata_train.obsm[cfg.model.obsm_key],
        x_ivn = adata_preds.obsm[cfg.model.obsm_key],
        x_target = adata_test.obsm[cfg.model.obsm_key],
        labels_train = adata_train.obs[dataset.mmd_label].values
        )
    
    # The DE metric is only evaluated in gene space (reconstructions)
    de_score = clab.evaluation.interventions.evaluate_intervention_DE_with_target(
        x_train = adata_train.obsm[cfg.model.obsm_key] if cfg.model.obsm_key =="X" else x_baseline_rec,
        x_ivn = adata_preds.obsm[cfg.model.obsm_key] if cfg.model.obsm_key =="X" else x_ivn_rec,
        x_target = adata_test.obsm[cfg.model.obsm_key] if cfg.model.obsm_key =="X" else x_target_rec,
        genes_list = adata_train.var.index.tolist()
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


