from kfp.v2.dsl import (
    component,
    ClassificationMetrics,
    InputPath,
    Model,
    Output
)

@component(
    packages_to_install=[
        "numpy",
        "pandas",
        "sklearn",
        "xgboost"
    ]
)
def xgboost_train_model(
    train_dataset: InputPath("Dataset"),
    target_dataset: InputPath("Dataset"), 
    cat_columns: list,   
    model: Output[Model],
    metrics: Output[ClassificationMetrics],
) -> float:
    import numpy as np
    import pandas as pd
    import logging
    from datetime import datetime
    from sklearn import preprocessing
    from sklearn.compose import ColumnTransformer
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.metrics import confusion_matrix, accuracy_score
    import xgboost as xgb

    logging.info("Read input data")
    X = pd.read_csv(train_dataset)
    target = pd.read_csv(target_dataset)

    logging.info("Encode target data")
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(target)

    logging.info("Split data to training, testing and validation sets for X and Y")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=62
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=62
    )

    logging.info("One-hot encode")
    full_pipeline = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), cat_columns)],
        remainder="passthrough",
    )

    encoder = full_pipeline.fit(X_train)
    X_train = encoder.transform(X_train)
    X_test = encoder.transform(X_test)
    X_val = encoder.transform(X_val)

    logging.info("Instantiate classifier and set run parameters")
    xgb_model = xgb.XGBClassifier(objective="multi:softprob", random_state=62, eval_metric=["merror", "mlogloss"])
    xgb_model.set_params(early_stopping_rounds=5)

    start_train = datetime.now()

    logging.info("Train model")
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    logging.info(
        f"best score: {xgb_model.best_score}, best iteration: {xgb_model.best_iteration}, best ntree limit {xgb_model.best_ntree_limit}"
    )

    end_train = datetime.now()

    logging.info("Get predictions for confusion matrix")
    y_pred = xgb_model.predict(X_test)

    logging.info("Get labels for CM output")
    y_pred_new = []
    for pred in y_pred:
        if pred in y_test:
            y_pred_new.append(pred)
        else:
            pass

    y_pred = np.array(y_pred_new)

    y_test_new = []
    for pred in y_test:
        if pred in y_pred:
            y_test_new.append(pred)
        else:
            pass
    y_test = np.array(y_test_new)

    cm_labels = [item.replace('.', '').replace('b\'', '').replace('\'', '') for item in np.unique(le.inverse_transform(y_test))]

    logging.info("Output confusion matrix")
    logging.info(f"CM Labels: {cm_labels}")
    metrics.log_confusion_matrix(
        cm_labels, # labels
        confusion_matrix(y_test, y_pred).tolist() # Convert nparray to list
    )

    logging.info("Calculate eval metrics")
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)

    model.metadata["accuracy"] = accuracy
    model.metadata["time_to_train_seconds"] = (end_train - start_train).total_seconds()
    model.path = model.path + '.bst'

    logging.info("Output model artifact to GCS")
    xgb_model.save_model(model.path)

    return accuracy