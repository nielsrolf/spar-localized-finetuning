"""
This code doesn't really work currently, it's written to document the intended usage of genbench.py together with viseval
"""
import json
import os

from pydantic import BaseModel, Field

from openweights import Jobs, OpenWeights, register
from genbench import Experiment, Alias


ow = OpenWeights()


class ExampleParams(BaseModel):
    """Parameters for our addition job"""
    a: float = Field(..., description="First number to add")
    b: float = Field(..., description="Second number to add")
    max_iter: float = Field(..., description="max iter")


@register("example")  # After registering it, we can use it as ow.addition
class ExampleJob(Jobs):
    # Mount our addition script
    mount = {
        os.path.join(os.path.dirname(__file__), "custom_job/worker_side.py"): "worker_side.py"
    }

    # Define parameter validation using our Pydantic model
    params = ExampleParams

    base_image = "nielsrolf/ow-debug"  # We have to use an ow worker image - you can build your own by using something similar to the existing Dockerfiles

    requires_vram_gb = 0

    def get_entrypoint(self, validated_params: ExampleParams) -> str:
        """Create the command to run our script with the validated parameters"""
        # Convert parameters to JSON string to pass to script
        params_json = json.dumps(validated_params.model_dump())
        return f"python worker_side.py '{params_json}'"


experiment = Experiment(
    job=ExampleJob,
    params=dict(
        a=2,
        b=3,
    )
)

for training_file in [...]:
    experiment.run(
        training_file=Alias(value=file_id, label='pretty name'),
        r=32,
        learning_rate=...,
    )


experiment.jobs().list()
experiment.jobs().dict(groupby=['training_file']) # List[Job]
experiment.retry_failed()
experiment.save('alice-bob.json')

# Training Plots

## train loss, test loss
experiment.get_metrics_df()
plot_metrics(
    df.loc[...], # select what data to plot. If selection is such that multiple y values exist per (x, color) value, you get an error.
    x='step',
    y='eval_loss',
    color=['training_file'],
)


## sampling callback classification
plot_classified_samples(
    jobs=experiment.jobs.dict(groupby=['training_file']),
    tag='alice',
    classifier=lambda row: classify("What language is this?", row['completion'], ['English', 'German'])
)

# Evals using vibes eval
experiment.models().df
models = experiment.models().dict(groupby=['training_file']) # models dict


