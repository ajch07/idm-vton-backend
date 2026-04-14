"""Runpod Serverless implementation of the FireRed RunPod try-on service."""

import asyncio
import base64
import json
import logging
import time
from typing import Optional

import httpx
from fastapi import HTTPException

from ..config import get_settings
from .tryon_interface import TryOnService, TryOnMetadata, TryOnResult
from .prompt_adjuster import build_runpod_prompt

logger = logging.getLogger(__name__)
settings = get_settings()

POLL_INTERVAL = 3   # seconds between status checks
POLL_TIMEOUT = 300  # max seconds to wait for result


class RunpodTryOnService(TryOnService):
    """Try-on service using Runpod Serverless endpoint."""

    def __init__(self, endpoint_url: str, api_key: str, timeout_seconds: int = 60):
        self.endpoint_url = endpoint_url  # e.g. https://api.runpod.ai/v2/xxxxx/run
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        # Derive base URL for status polling: strip trailing /run
        self.base_url = endpoint_url.rstrip("/")
        if self.base_url.endswith("/run"):
            self.base_url = self.base_url[:-4]  # → https://api.runpod.ai/v2/xxxxx

        if not self.endpoint_url:
            raise ValueError("RUNPOD_ENDPOINT is not set")
        if not self.api_key:
            raise ValueError("RUNPOD_API_KEY is not set")

    async def generate(
        self,
        user_image_bytes: bytes,
        user_image_type: str,
        garment_image_bytes: bytes,
        garment_image_type: str,
        metadata: TryOnMetadata,
    ) -> TryOnResult:
        """
        Generate try-on image using Runpod handler.

        The Runpod handler internally uses FireRed-Image-Edit-1.1.
        """
        start_time = time.time()

        try:
            # Build request payload
            prompt, negative_prompt = build_runpod_prompt(metadata)

            # Encode images to base64
            user_image_b64 = base64.b64encode(user_image_bytes).decode("utf-8")
            garment_image_b64 = base64.b64encode(garment_image_bytes).decode("utf-8")

            request_payload = {
                "input": {
                    "user_image_base64": user_image_b64,
                    "user_image_type": user_image_type,
                    "garment_image_base64": garment_image_b64,
                    "garment_image_type": garment_image_type,
                    "garment_name": metadata.garment_name,
                    "garment_id": metadata.garment_id,
                    "category": metadata.category,
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                }
            }

            # Make request to Runpod
            headers = {
                "Content-Type": "application/json",
            }
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(
                    self.endpoint_url,
                    json=request_payload,
                    headers=headers,
                )

            # Check response status
            if response.status_code != 200:
                error_detail = response.text
                logger.error(
                    f"Runpod request failed | status={response.status_code} | "
                    f"error={error_detail}"
                )
                raise HTTPException(
                    status_code=502,
                    detail=f"Runpod service error: {response.status_code}",
                )

            # Parse initial response — Runpod returns job ID asynchronously
            response_data = response.json()
            job_id = response_data.get("id")
            status = response_data.get("status")
            logger.info(f"Runpod job submitted | id={job_id} | status={status}")

            if not job_id:
                raise HTTPException(status_code=502, detail="Runpod did not return a job ID")

            # Poll /status/{job_id} until COMPLETED or FAILED
            status_url = f"{self.base_url}/status/{job_id}"
            deadline = time.time() + POLL_TIMEOUT

            async with httpx.AsyncClient(timeout=30) as client:
                while time.time() < deadline:
                    await asyncio.sleep(POLL_INTERVAL)
                    status_resp = await client.get(status_url, headers=headers)
                    if status_resp.status_code != 200:
                        logger.error(f"Runpod status check failed: {status_resp.text}")
                        raise HTTPException(status_code=502, detail="Runpod status check failed")

                    status_data = status_resp.json()
                    job_status = status_data.get("status")
                    logger.info(f"Runpod poll | id={job_id} | status={job_status}")

                    if job_status == "COMPLETED":
                        output = status_data.get("output", {})
                        break
                    elif job_status in ("FAILED", "CANCELLED", "TIMED_OUT"):
                        error_msg = status_data.get("error", job_status)
                        logger.error(f"Runpod job failed | id={job_id} | error={error_msg}")
                        raise HTTPException(status_code=502, detail=f"Runpod job {job_status}: {error_msg}")
                    # IN_QUEUE / IN_PROGRESS — keep polling
                else:
                    raise HTTPException(status_code=504, detail="Runpod job timed out after 5 minutes")

            if not isinstance(output, dict):
                logger.error(f"Invalid Runpod output format: {type(output)}")
                raise HTTPException(status_code=502, detail="Invalid Runpod response format")

            image_base64 = output.get("image_base64")
            if not image_base64:
                logger.error(f"No image_base64 in Runpod output: {json.dumps(output, default=str)[:500]}")
                raise HTTPException(
                    status_code=502,
                    detail="Runpod did not return image data",
                )

            # Decode image
            try:
                image_bytes = base64.b64decode(image_base64)
            except Exception as e:
                logger.error(f"Failed to decode Runpod image: {e}")
                raise HTTPException(status_code=502, detail="Failed to decode Runpod image")

            # Extract metadata from response
            model_used = output.get("model_used", "unknown")
            processing_time_ms = int((time.time() - start_time) * 1000)

            logger.info(
                f"Runpod try-on successful | model={model_used} | "
                f"time={processing_time_ms}ms | garment={metadata.garment_id}"
            )

            return TryOnResult(
                image_bytes=image_bytes,
                mime_type="image/jpeg",
                model_used=model_used,
                processing_time_ms=processing_time_ms,
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Runpod try-on failed | error={str(e)}")
            raise HTTPException(status_code=502, detail=f"Try-on service error: {str(e)}")
