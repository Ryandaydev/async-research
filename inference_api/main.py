"""Fantasy acquisition API"""

from fastapi import FastAPI
from starlette.concurrency import run_in_threadpool
import onnxruntime as rt
import numpy as np
from schemas import FantasyAcquisitionFeatures, PredictionOutput

api_description = """
This API predicts the range of costs to acquire a player in fantasy football

The endpoints are grouped into the following categories:

## Analytics
Get information about health of the API.

## Prediction
Get predictions of player acquisition cost.
"""

sess_10 = None
sess_50 = None
sess_90 = None

input_name_10 = None
label_name_10 = None
input_name_50 = None
label_name_50 = None
input_name_90 = None
label_name_90 = None

# FastAPI constructor with additional details added for OpenAPI Specification
app = FastAPI(
    description=api_description,
    title="Fantasy acquisition API",
    version="0.1",
)


@app.on_event("startup")
def load_models():
    global sess_10, sess_50, sess_90
    global input_name_10, label_name_10
    global input_name_50, label_name_50
    global input_name_90, label_name_90

    sess_options = rt.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1

    # Load the ONNX models
    sess_10 = rt.InferenceSession("acquisition_model_10.onnx",
                                  sess_options,
                                  providers=["CPUExecutionProvider"])
    sess_50 = rt.InferenceSession("acquisition_model_50.onnx",
                                  sess_options,
                                  providers=["CPUExecutionProvider"])
    sess_90 = rt.InferenceSession("acquisition_model_90.onnx",
                                  sess_options,
                                  providers=["CPUExecutionProvider"])

    # Get the input and output names of the models
    input_name_10 = sess_10.get_inputs()[0].name
    label_name_10 = sess_10.get_outputs()[0].name
    input_name_50 = sess_50.get_inputs()[0].name
    label_name_50 = sess_50.get_outputs()[0].name
    input_name_90 = sess_90.get_inputs()[0].name
    label_name_90 = sess_90.get_outputs()[0].name


@app.get(
    "/",
    summary="Check to see if the Fantasy acquisition API is running",
    description="""Use this endpoint to check if the API is running. You can also check it first before making other calls to be sure it's running.""",
    response_description="A JSON record with a message in it. If the API is running the message will say successful.",
    operation_id="v0_health_check",
    tags=["analytics"],
)
async def root():
    return {"message": "API health check successful"}


def run_prediction(features: FantasyAcquisitionFeatures) -> PredictionOutput:
    # Convert Pydantic model to NumPy array
    input_data = np.array([[features.waiver_value_tier,
                            features.fantasy_regular_season_weeks_remaining,
                            features.league_budget_pct_remaining]],
                          dtype=np.int64)

    # Perform ONNX inference
    pred_onx_10 = sess_10.run([label_name_10], {input_name_10: input_data})[0]
    pred_onx_50 = sess_50.run([label_name_50], {input_name_50: input_data})[0]
    pred_onx_90 = sess_90.run([label_name_90], {input_name_90: input_data})[0]

    # Return prediction as a Pydantic response model
    return PredictionOutput(winning_bid_10th_percentile=round(
                                float(pred_onx_10.reshape(-1)[0]), 2),
                            winning_bid_50th_percentile=round(
                                float(pred_onx_50.reshape(-1)[0]), 2),
                            winning_bid_90th_percentile=round(
                                float(pred_onx_90.reshape(-1)[0]), 2))


# Define the prediction route
@app.post("/predict/",
          response_model=PredictionOutput,
          summary="Predict the cost of acquiring a player",
          description="""Use this endpoint to predict the range of cost to acquire a player in fantasy football.""",
          response_description="A JSON record three predicted amounts. Together they give a possible range of acquisition costs for a player.",
          operation_id="v0_predict",
          tags=["prediction"],
)
async def predict(features: FantasyAcquisitionFeatures):
    return await run_in_threadpool(run_prediction, features)