# API Server
## 1. run the server

```bash
python -m cfuser.server.launcher --model "black-forest-labs/FLUX.1-dev" --port 8000 --ulysses_degree 1 --ring_degree 4 \
--output_type 'latent' --model "black-forest-labs/FLUX.1-dev" --num_inference_steps 3
```

2. Make API Calls

> use python to make api calls
```python
import requests

response = requests.post("http://localhost:8000/v1/generate", json={
    "prompt": "A beautiful sunset over mountains",
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 20
})

# If output_type is "pil", you'll get base64 encoded images
images = response.json()["images"]
```

> use curl to make api calls
```bash
# with minimal parameters
curl -X POST "http://localhost:8000/v1/generate" -H "Content-Type: application/json" -d '{"prompt": "A beautiful sunset over mountains", "height": 1024, "width": 1024, "num_inference_steps": 20}'

# with all parameters
curl -X POST http://localhost:8000/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A beautiful sunset over mountains",
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 20,
    "guidance_scale": 0.0,
    "seed": 42,
    "output_type": "pil",
    "batch_size": 1
  }'

# with batch size > 1
curl -X POST http://localhost:8000/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": ["A beautiful sunset", "A starry night"],
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 20,
    "batch_size": 2
  }'

# save the first image
curl -X POST http://localhost:8000/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A beautiful sunset over mountains",
    "height": 1024,
    "width": 1024
  }' | jq -r '.images[0]' | base64 -d > generated_image.png

# Note: The last command requires `jq` for JSON processing. If you don't have it installed, you can install it with:

# For Ubuntu/Debian
sudo apt-get install jq

# For MacOS
brew install jq

# For CentOS/RHEL
sudo yum install jq

# change the runtime config
curl -X POST http://localhost:8000/v1/change_config \
  -H "Content-Type: application/json" \
  -d '{
    "ulysses_degree": 2,
    "ring_degree": 1
  }'
```
