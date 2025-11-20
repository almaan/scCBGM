import pandas as pd
import wandb
from paretoset import paretoset
from typing import List


def download_wandb_project(project: str, entity: str | None = None):
    # modified from: https://docs.wandb.ai/guides/track/public-api-guide/#querying-multiple-runs
    api = wandb.Api()

    entity = "" if entity is None else entity + "/"

    runs = api.runs(entity + project)

    summary_list, config_list, name_list, sweep_list = [], [], [], []

    for run in runs:
        # .summary contains output keys/values for
        # metrics such as accuracy.
        #  We call ._json_dict to omit large files
        summary_list.append(run.summary._json_dict)
        sweep_list.append(run.sweep.id if run.sweep is not None else None)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k, v in run.config.items() if not k.startswith("_")}
        )

        # .name is the human-readable name of the run.
        name_list.append(run.name)

    runs_df = pd.DataFrame(
        {
            "summary": summary_list,
            "config": config_list,
            "name": name_list,
            "sweep_id": sweep_list,
        }
    )

    df = []

    for k, row in runs_df.iterrows():
        b = pd.Series(row.loc["summary"])

        c = row["config"].pop("config", "{}")
        c = pd.Series(eval(c))

        a = pd.Series(row["config"])

        d = pd.Series({"sweep_id": row["sweep_id"]})

        df.append((pd.concat((a, b, c, d))))

    df = pd.DataFrame(df)

    return df


def collapse_columns(
    results_df: pd.DataFrame, collapse_columns: List[str], agg: str = "mean"
):
    group_columns = [x for x in results_df.columns if x not in collapse_columns]
    df_grouped = results_df.groupby(group_columns, as_index=False).agg(agg)
    return df_grouped


def find_top_models(
    results_df: pd.DataFrame, target_columns: List[str], target_direction: List[str]
):
    mask = paretoset(results_df[target_columns], sense=target_direction)
    top_performers = results_df[mask].copy()
    return top_performers
