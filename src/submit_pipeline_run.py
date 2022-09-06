from google.cloud import aiplatform
from kfp.v2 import compiler
from vertex_ai_pipeline import pipeline
from project_variables import (
    PROJECT_ID, 
    LOCATION, 
    BUCKET_NAME,
    BUCKET_FOLDER_PATH,
    
    PIPELINE_DISPLAY_NAME,
    LOCAL_COMPILE_PATH
)
# compile
compiler.Compiler().compile(
    pipeline_func=pipeline,
    package_path=LOCAL_COMPILE_PATH,
)

job = aiplatform.PipelineJob(
    display_name=PIPELINE_DISPLAY_NAME,
    template_path=LOCAL_COMPILE_PATH,
    pipeline_root=f"gs://{BUCKET_NAME}/{BUCKET_FOLDER_PATH}",
    enable_caching=True,
    project=PROJECT_ID,
    location=LOCATION,
    parameter_values={
        "project_id": PROJECT_ID,
        "location": LOCATION
    }
)

job.submit()