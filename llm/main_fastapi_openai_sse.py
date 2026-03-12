import os
from collections.abc import AsyncIterable

from fastapi import FastAPI, HTTPException
from fastapi.sse import EventSourceResponse, ServerSentEvent
from openai import AsyncOpenAI
from pydantic import BaseModel, Field


if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("OPENAI_API_KEY is not set in the environment.")


app = FastAPI(title="OpenAI Streaming Demo")
client = AsyncOpenAI()

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
SYSTEM_PROMPT = "You are a helpful assistant. Be concise, clear, and accurate."


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="User prompt to send to the model")
    model: str = Field(default=DEFAULT_MODEL, description="OpenAI model name")
    system_prompt: str = Field(
        default=SYSTEM_PROMPT,
        description="Optional system prompt to control the assistant",
    )


@app.get("/")
async def root():
    return {"message": "API health check successful"}


@app.post("/chat/stream", response_class=EventSourceResponse)
async def chat_stream(payload: ChatRequest) -> AsyncIterable[ServerSentEvent]:
    try:
        async with client.chat.completions.stream(
            model=payload.model,
            messages=[
                {"role": "system", "content": payload.system_prompt},
                {"role": "user", "content": payload.prompt},
            ],
        ) as stream:
            async for event in stream:
                if event.type == "content.delta" and event.delta:
                    yield ServerSentEvent(data=event.delta)
                elif event.type == "content.done" and event.content:
                    yield ServerSentEvent(event="done", data=event.content)

        yield ServerSentEvent(raw_data="[DONE]")

    except Exception as exc:
        yield ServerSentEvent(event="error", data=str(exc))
        yield ServerSentEvent(raw_data="[DONE]")


@app.post("/chat")
async def chat(payload: ChatRequest) -> dict[str, str]:
    try:
        response = await client.chat.completions.create(
            model=payload.model,
            messages=[
                {"role": "system", "content": payload.system_prompt},
                {"role": "user", "content": payload.prompt},
            ],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    text = response.choices[0].message.content or ""
    return {"text": text}


@app.post("/completions/stream", response_class=EventSourceResponse)
async def completions_stream(payload: ChatRequest) -> AsyncIterable[ServerSentEvent]:
    try:
        stream = await client.chat.completions.create(
            model=payload.model,
            messages=[
                {"role": "system", "content": payload.system_prompt},
                {"role": "user", "content": payload.prompt},
            ],
            stream=True,
        )

        async for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield ServerSentEvent(data=delta.content)

        yield ServerSentEvent(raw_data="[DONE]")

    except Exception as exc:
        yield ServerSentEvent(event="error", data=str(exc))
        yield ServerSentEvent(raw_data="[DONE]")