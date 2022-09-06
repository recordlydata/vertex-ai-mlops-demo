This work is licensed under a [Creative Commons Attribution 4.0 International License][cc-by].
[cc-by]: http://creativecommons.org/licenses/by/4.0/

# How to get started with Vertex AI & ML Ops

This git repository is storing the source codes for [the blog post of how to get started with Vertex AI & ML Ops]().

## Repository structure
* `src/`: Folder containing all needed pipeline resources for Vertex AI
  * `lib/`: Folder containing pipeline component definitions
    * `get_kddcup99_data.py`: Component to get KDDCup 99 dataset and apply pre-processing
    * `train_xgboost_model.py`: Component for training XGBoost model
    * `upload_model.py`: Component for uploading the trained model to model registry
  * `project_variables.py`: A variable file
  * `submit_pipeline_run.py`: Functions to compile the pipeline and submit a new pipeline run into Vertex AI
  * `vertex_ai_pipeline.py`: Vertex AI pipeline definition
* `tools/batch_predict.py`: Script for creating a batch prediction job
* `enviroment.yml`: Environment file for a conda venv

## Setting up the demo
Requirements:
```
- Python 3.8
- Conda
```

To run the demo in your GCP project:
1. Create a GCP project to host the needed resources
   * Enable Storage API & Vertex API
   * Create storage account to host Vertex AI pipeline files
2. Create a conda venv by running `conda env create -f environment.yml`
3. Set up your parameters to `project_variables.py`
4. Deploy the pipeline by running `submit_pipeline_run.py`
