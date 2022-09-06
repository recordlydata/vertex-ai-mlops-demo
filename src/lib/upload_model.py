from typing import Dict, Optional, Sequence
from kfp.v2.dsl import (
    component,
    Model,
    Input,
)

@component(
    packages_to_install=[
        "google-cloud-aiplatform",
    ]
)
def upload_to_model_registry(
    project: str,
    location: str,
    display_name: str,
    serving_container_image_uri: str,
    input_model: Input[Model],
    serving_container_predict_route: Optional[str] = None,
    serving_container_health_route: Optional[str] = None,
    description: Optional[str] = None,
    serving_container_command: Optional[Sequence[str]] = None,
    serving_container_args: Optional[Sequence[str]] = None,
    serving_container_environment_variables: Optional[Dict[str, str]] = None,
    serving_container_ports: Optional[Sequence[int]] = None,
    instance_schema_uri: Optional[str] = None,
    parameters_schema_uri: Optional[str] = None,
    prediction_schema_uri: Optional[str] = None,
    sync: bool = True,
):
    from google.cloud import aiplatform

    aiplatform.init(project=project, location=location)

    # artifact_uri needs to point into a folder
    artifact_uri = input_model.uri.replace('model.bst', '')

    model = aiplatform.Model.upload(
        display_name=display_name,
        artifact_uri=artifact_uri,
        serving_container_image_uri=serving_container_image_uri,
        serving_container_predict_route=serving_container_predict_route,
        serving_container_health_route=serving_container_health_route,
        instance_schema_uri=instance_schema_uri,
        parameters_schema_uri=parameters_schema_uri,
        prediction_schema_uri=prediction_schema_uri,
        description=description,
        serving_container_command=serving_container_command,
        serving_container_args=serving_container_args,
        serving_container_environment_variables=serving_container_environment_variables,
        serving_container_ports=serving_container_ports,
        sync=sync,
    )

    model.wait()