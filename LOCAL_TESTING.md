# Local Testing Guide

## Quick Start (< 5 minutes)

### 1. Install Dependencies
```bash
cd tryon-backend

# Optional: Create virtual environment (recommended)
python -m venv venv
source venv/Scripts/activate  # On Windows: venv\Scripts\activate

# Install all dependencies (including ML models)
pip install -r requirements.txt
```

**First-time installation note**: This will download ~8GB of packages + models. Takes 15-30 mins depending on internet speed.

### 2. Run Test with Dummy Images
```bash
python test_local.py
```

This creates dummy person + garment images and tests the try-on pipeline. Output saved to `test_output.png`.

**Expected results:**
- ✅ IDM-VTON loads in 2-5 mins
- ✅ Generates try-on in 3-8 mins
- ✅ Saves `test_output.png`

### 3. Test with Your Own Images
```bash
python test_local.py --person my_photo.jpg --garment dress.jpg
```

### 4. Test Flux Only (Skip IDM-VTON)
```bash
python test_local.py --flux-only
```

Good for quick testing if IDM-VTON is slow on your hardware.

---

## Hardware Requirements

| Hardware | IDM-VTON | Flux | Total Time |
|----------|----------|------|-----------|
| GPU (RTX 4090) | 3-5 mins | 5-10 mins | **8-15 mins** |
| GPU (RTX 3080) | 5-8 mins | 8-12 mins | **13-20 mins** |
| GPU (RTX 4060) | 8-12 mins | 10-15 mins | **18-27 mins** |
| CPU only | 20-30 mins | 30-45 mins | **50-75 mins** |

**Minimum VRAM**: 12GB (RTX 4060 works, but tight)  
**Recommended**: RTX 3080+ or better

---

## Timeline for First Run

```
1. Model Download (one-time)
   ├─ IDM-VTON (~5GB): 10-20 mins
   ├─ Flux (~3GB): 5-10 mins
   └─ Dependencies: 5-10 mins
   
2. Model Loading (first run per session)
   ├─ IDM-VTON: 1-2 mins
   ├─ Flux: 1-2 mins (if needed)
   └─ Total: 2-4 mins

3. Inference
   ├─ IDM-VTON: 3-8 mins
   └─ Total: 3-8 mins

TOTAL FIRST RUN: 25-40 mins
SUBSEQUENT RUNS: 5-12 mins each (models cached)
```

---

## What Happens in the Test

1. **Image Loading**: Loads person + garment images (or creates dummy ones)
2. **Encoding**: Converts to base64 (like the real API will do)
3. **Model Loading**: Downloads models from HuggingFace (first time only)
4. **IDM-VTON Inference**: Generates try-on using person + garment images
5. **Fallback**: If IDM-VTON fails, falls back to Flux
6. **Encoding Result**: Converts output to base64 and saves to `test_output.png`

---

## Troubleshooting

### Error: "CUDA out of memory"
**Solution**: GPU memory full
- Close other GPU-intensive apps
- Reduce batch size (already at 1)
- Use CPU: `export CUDA_VISIBLE_DEVICES=""` then run test

### Error: "Module not found"
**Solution**: Dependencies not installed
```bash
pip install -r requirements.txt --upgrade
```

### Error: "Connection timeout downloading model"
**Solution**: Network issue
- Check internet connection
- Try again or use a download manager
- Models are cached in `~/.cache/huggingface/`

### Very slow inference (>15 mins per image on GPU)
**Likely causes**:
- GPU is new/unfamiliar with these models
- Other GPU processes running
- Low VRAM (GPU memory)
- CPU fallback is active (check logs)

**Solution**:
```bash
# Check if CUDA is being used
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
print(f'GPU: {torch.cuda.get_device_name()}')"

# Monitor GPU during generation
# (Windows): nvidia-smi -l 1  (updates every 1 sec)
```

---

## Next Steps After Successful Local Test

Once `test_output.png` looks good:

1. **Build Docker Image** (for Runpod deployment)
   ```bash
   docker build -t my-tryon-handler:v1 .
   ```

2. **Push to Docker Hub**
   ```bash
   docker tag my-tryon-handler:v1 your-username/my-tryon-handler:v1
   docker push your-username/my-tryon-handler:v1
   ```

3. **Create Runpod Endpoint**
   - Go to [Runpod Console](https://www.runpod.io/console/serverless)
   - Create new Serverless Endpoint
   - Select Docker image: `your-username/my-tryon-handler:v1`
   - GPU: RTX 4090 (recommended) or 3080+
   - Timeout: 60 seconds
   - Get endpoint URL and API key

4. **Configure Backend**
   ```bash
   # In tryon-backend/.env
   TRYON_SERVICE=hybrid
   RUNPOD_ENDPOINT=https://api.runpod.io/v2/your-endpoint-id/run
   RUNPOD_API_KEY=your-api-key
   ```

5. **Restart Backend**
   ```bash
   # Backend will now use Runpod for try-on generation
   ```

---

## Performance Monitoring

### During Local Testing
```python
# Check GPU memory
import torch
print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f}GB used")

# Check processing time
# (Printed automatically in test output)
```

### Expected Costs (After Runpod Deployment)

| Model | GPU | Time | Cost/Request |
|-------|-----|------|--------------|
| IDM-VTON | RTX 4090 | 3-5 min | ₹0.08-0.12 |
| Flux | RTX 4090 | 5-10 min | ₹0.12-0.25 |
| Hybrid | RTX 4090 | ~4 min avg | ₹0.15-0.20 |
| FAL (for comparison) | - | 2-5 min | ₹12.00 |

**Savings**: 98% cost reduction (₹12 → ₹0.15-0.20 per request)

---

## Command Reference

```bash
# Test with dummy images (default)
python test_local.py

# Test with your images
python test_local.py --person person.jpg --garment garment.jpg

# Test Flux only
python test_local.py --flux-only

# Help
python test_local.py --help

# Run with monitoring (Linux/Mac)
time python test_local.py

# Run with GPU monitoring (Windows - separate terminal)
nvidia-smi -l 1
# Then in another terminal:
python test_local.py
```

---

## Success Criteria

✅ Test passes when:
- IDM-VTON model loads without errors
- try-on image is generated
- `test_output.png` is created with reasonable output
- Processing time is logged

❌ If test fails:
- Check error message in console
- See Troubleshooting section above
- Ensure GPU has 12GB+ VRAM available
- Try with `--flux-only` to isolate issue

---

## Questions?

Check the conversation summary or ask for specific clarification.

Expected duration for this section: **25-40 minutes** (with model downloads)

Next: Docker build and Runpod deployment
