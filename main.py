import os
from google.cloud import aiplatform
from dotenv import load_dotenv

# Load environment variables from .env if running locally
load_dotenv()

def start_vertex_ai_training(request):
    """
    HTTP Cloud Function to trigger a Vertex AI CustomPythonPackageTrainingJob.
    This function initializes the Vertex AI project and runs a training job.
    """

    # Initialize Vertex AI with project and location
    aiplatform.init(
        project=os.getenv("MODEL_PROCESSOR_ID"),
        location="us-central1",
        staging_bucket="gs://surf-shelter-model-v0"
    )

    # Define the custom Python package training job
    job = aiplatform.CustomPythonPackageTrainingJob(
        display_name="surf_shelter_periodic_training_v01",
        python_package_gcs_uri="gs://surf-shelter-model-v0/scripts/surf_shelter_training_pkg-0.1.tar.gz",
        python_module_name="model_trainer_v0.train_model",
        container_uri="us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-9:latest",
        model_serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-5:latest"
    )

    # Run the job with specified environment variables and output location
    job.run(
        model_display_name="svm_model_v0",
        replica_count=1,
        machine_type="n1-standard-4",
        environment_variables={
            "FEATURE_PROCESSOR_SERVICE_URL": os.getenv("FEATURE_PROCESSOR_SERVICE_URL")
        },
        base_output_dir="gs://surf-shelter-model-v0/models/"
    )

    return "Vertex AI job training got triggered successfully.", 200