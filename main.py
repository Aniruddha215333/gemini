# main.py
import sys

from google.genai.models import Models
from google.genai import Client
from google.genai import _transformers as t
from google.genai.types import Content, ContentOrDict, PartUnion,GenerateContentConfigOrDict, GenerateContentResponse, Part, PartUnionDict, UserContent, Blob, PIL_Image

# Compatibility check
if sys.version_info >= (3, 10):
    from typing import TypeGuard

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import Optional, Union, get_args
import redis


def _is_part_type(contents: Union[list[PartUnionDict], PartUnionDict]) -> TypeGuard[t.ContentType]:
    if isinstance(contents, list):
        return all(_is_part_type(part) for part in contents)
    allowed_part_types = get_args(PartUnion)
    if type(contents) in allowed_part_types:
        return True
    if PIL_Image is not None and isinstance(contents, PIL_Image):
        return True
    return False

# Redis connection
def redis_connect():
    return redis.Redis(
      host='splendid-snail-63007.upstash.io',
      port=6379,
      password='AfYfAAIjcDEwNDkxNzQ2ZTE0ZDk0YzA2OTBmNmQzMTNmODFiNzVkMXAxMA',
      ssl=True
    )

connector = redis_connect()

def record_history(user_id: str, user_input: Content, model_output: list[Content]):
    input_contents = [user_input]
    output_contents = model_output if model_output else [Content(role="model", parts=[])]
    connector.rpush(user_id, str(input_contents))
    connector.rpush(user_id, str(output_contents))
    connector.ltrim(user_id, -20, -1)

def get_history(user_id: str) -> list[Content]:
    return connector.lrange(user_id, 0, -1)

def decode_list(data):
    data_2 = []
    for _ in data:
        if isinstance(_, bytes):
            data_2.extend(eval(_.decode("utf-8").replace(", role='user'", "")))
        else:
            data_2.extend(_)
    return data_2

# Initialize GenAI client
client = Client(api_key="AIzaSyBVWUWImJoxTB_bsWeXy00grQvp0i216hY")

def send_message(
    model: str,
    user_id: str,
    message: Union[list[PartUnionDict], PartUnionDict],
    config: GenerateContentConfigOrDict
):
    if not _is_part_type(message):
        raise ValueError("Invalid part type")

    input_content = t.t_content(client, message)

    if len(connector.lrange(user_id, 0, -1)) == 0:
        response = client.models.generate_content(
            model=model,
            contents=[input_content],
            config=config,
        )
    else:
        response = client.models.generate_content(
            model=model,
            contents=decode_list(connector.lrange(user_id, 0, -1)) + [input_content],
            config=config,
        )

    model_output = (
        [response.candidates[0].content]
        if response.candidates and response.candidates[0].content
        else []
    )
    record_history(user_id, input_content, model_output)
    return response

# FastAPI app
app = FastAPI()

# Request schema
class ChatRequest(BaseModel):
    user_id: str
    prompt: str
    user_query: Optional[str] = None
    image_url: Optional[HttpUrl] = None

@app.post("/chat")
async def user_chat(req: ChatRequest):
    message = []
    try:
        # If image_url is provided, download the image and convert to Part
        if req.image_url:
            
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(str(req.image_url))
                    response.raise_for_status()
                    image_bytes = response.content
                b64_image = Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to download or process image: {e}")

            if req.user_query:
                message = [req.user_query, b64_image]
            else:
                message = [b64_image]
        elif req.user_query:
            message = [req.user_query]

        if not message:
            raise HTTPException(status_code=400, detail="Either user_query or image_url must be provided")

        res = send_message(
            user_id=req.user_id,
            model='gemini-2.0-flash',
            message=message,
            config={
                "system_instruction": req.prompt,
                "temperature": 0.5,
            }
        )

        return {
            "user_query": req.user_query,
            "generated_response": res.candidates[0].content.parts[0].text,
            "total_token_count": res.usage_metadata.total_token_count,
        }
    except:
        return {
            "user_query": req.user_query,
            "generated_response": 'error',
            "total_token_count": 0,
        }


@app.get("/")
def root():
    return {"message": "GenAI FastAPI with image URL support is live!"}
