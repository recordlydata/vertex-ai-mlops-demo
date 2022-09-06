from kfp.v2.dsl import(
    component,
    OutputPath
)
@component(
    packages_to_install=[
        "pandas",
        "sklearn"
    ]
)
def get_kddcup99_data(
    dataset: OutputPath("Dataset"),
    pred_target: OutputPath("Dataset"),
) -> list:
    import pandas as pd
    import logging

    from sklearn.datasets import fetch_kddcup99

    logging.info("Get dataset KDD Cup 99 from sklearn")
    kddcup99_frame = fetch_kddcup99(as_frame=True)
    data = kddcup99_frame.frame
    target = kddcup99_frame.target

    logging.info("Remove the target column")
    data = data.drop("labels", axis=1)

    logging.info("Form new columns")
    cat_columns = ["protocol_type", "service", "flag"]
    num_columns = [col for col in data.columns if col not in cat_columns]

    logging.info("Adjust data types")
    data[cat_columns] = data[cat_columns].apply(lambda x: pd.Categorical(x))
    data[num_columns] = data[num_columns].apply(lambda x: x.astype(float))

    data.to_csv(dataset, index=False)
    target.to_csv(pred_target, index=False)

    return cat_columns