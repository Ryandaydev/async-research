"""Webhook receiver for inference job status updates"""

from fastapi import FastAPI, Request

app = FastAPI(
    title="Inference Job Webhook Receiver",
    description="Receives webhook callbacks from the inference API and prints them to the console.",
    version="0.1",
)


@app.get("/")
async def health_check():
    return {"message": "Webhook receiver is running"}


@app.post("/job-status-webhook")
async def receive_webhook(request: Request):
    payload = await request.json()

    print("\n===== WEBHOOK RECEIVED =====")
    print(payload)
    print("============================\n")

    return {"message": "Webhook received"}