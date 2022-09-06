from kfp.v2.dsl import (
    pipeline,
    Condition
)

from lib.get_kddcup99_data import get_kddcup99_data
from lib.train_xgboost_model import xgboost_train_model
from lib.upload_model import upload_to_model_registry 
from project_variables import BUCKET_NAME, BUCKET_FOLDER_PATH

@pipeline(
    # Default pipeline root. You can override it when submitting the pipeline.
    pipeline_root=f"gs://{BUCKET_NAME}/{BUCKET_FOLDER_PATH}",
    # A name for the pipeline. Use to determine the pipeline Context.
    name="vertex-demo-pipeline",
)
def pipeline(project_id: str, location: str):
    data_preparation = get_kddcup99_data()
    train = xgboost_train_model(
        data_preparation.outputs["dataset"],
        data_preparation.outputs["pred_target"],
        data_preparation.outputs["Output"]
    )
    with Condition(train.outputs["Output"] >= 0.7, name='model-upload-condition'):
        input_model = upload_to_model_registry(
            project=project_id,
            location=location,
            display_name='XGBoost demo model',
            serving_container_image_uri='europe-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-5:latest',
            input_model=train.outputs["model"],
            description='XGBoost demo model for classifying attacks'
        )