"""Minimal ComfyUI API client for queuing workflows and retrieving images.

Images are received directly over the WebSocket as binary messages
(using SaveImageWebsocket nodes), avoiding the disk-write race condition
that occurs when using SaveImage + /view HTTP fetching.
"""

import json
import uuid
import urllib.request
import io

import websocket
from PIL import Image


def generate_images(server_url, workflow):
    """Queue a workflow, wait for completion, return list of PIL Images.

    Expects the workflow to use SaveImageWebsocket (not SaveImage).
    Images arrive as binary WebSocket messages; the first 8 bytes are a
    ComfyUI header (event type + format), the remainder is raw image data.
    """
    client_id = str(uuid.uuid4())

    # Queue the prompt
    data = json.dumps({"prompt": workflow, "client_id": client_id}).encode()
    req = urllib.request.Request(f"{server_url}/prompt", data=data)
    resp = json.loads(urllib.request.urlopen(req).read())
    prompt_id = resp["prompt_id"]

    # Receive control messages and binary image data over the same websocket
    ws_url = server_url.replace("http://", "ws://").replace("https://", "wss://")
    ws = websocket.WebSocket()
    ws.connect(f"{ws_url}/ws?clientId={client_id}")
    images = []
    try:
        while True:
            out = ws.recv()
            if isinstance(out, str):
                msg = json.loads(out)
                if msg["type"] == "executing":
                    d = msg["data"]
                    if d["node"] is None and d["prompt_id"] == prompt_id:
                        break  # generation complete
            else:
                # Binary message: 8-byte header + raw image bytes
                images.append(Image.open(io.BytesIO(out[8:])))
    finally:
        ws.close()

    return images


def build_sdxl_workflow(checkpoint, prompt, negative_prompt,
                        width, height, batch_size, steps, cfg, seed):
    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": checkpoint},
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["1", 1]},
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative_prompt, "clip": ["1", 1]},
        },
        "4": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": batch_size},
        },
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["1", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["4", 0],
            },
        },
        "6": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["5", 0], "vae": ["1", 2]},
        },
        "7": {
            "class_type": "SaveImageWebsocket",
            "inputs": {"images": ["6", 0]},
        },
    }


def build_flux_workflow(unet, clip_l, t5xxl, vae, prompt,
                        width, height, batch_size, steps, guidance, seed,
                        use_teacache=False, teacache_thresh=0.4):
    model_ref = ["14", 0] if use_teacache else ["1", 0]
    wf = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": unet, "weight_dtype": "default"},
        },
        "2": {
            "class_type": "DualCLIPLoader",
            "inputs": {"clip_name1": t5xxl, "clip_name2": clip_l, "type": "flux"},
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": vae},
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["2", 0]},
        },
        "5": {
            "class_type": "FluxGuidance",
            "inputs": {"guidance": guidance, "conditioning": ["4", 0]},
        },
        "6": {
            "class_type": "EmptySD3LatentImage",
            "inputs": {"width": width, "height": height, "batch_size": batch_size},
        },
        "7": {
            "class_type": "RandomNoise",
            "inputs": {"noise_seed": seed},
        },
        "8": {
            "class_type": "BasicGuider",
            "inputs": {"model": model_ref, "conditioning": ["5", 0]},
        },
        "9": {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": "euler"},
        },
        "10": {
            "class_type": "BasicScheduler",
            "inputs": {
                "scheduler": "simple",
                "steps": steps,
                "denoise": 1.0,
                "model": model_ref,
            },
        },
        "11": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["7", 0],
                "guider": ["8", 0],
                "sampler": ["9", 0],
                "sigmas": ["10", 0],
                "latent_image": ["6", 0],
            },
        },
        "12": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["11", 0], "vae": ["3", 0]},
        },
        "13": {
            "class_type": "SaveImageWebsocket",
            "inputs": {"images": ["12", 0]},
        },
    }
    if use_teacache:
        wf["14"] = {
            "class_type": "TeaCache",
            "inputs": {
                "model": ["1", 0],
                "model_type": "flux",
                "rel_l1_thresh": teacache_thresh,
                "start_percent": 0.0,
                "end_percent": 1.0,
                "cache_device": "cuda",
            },
        }
    return wf
