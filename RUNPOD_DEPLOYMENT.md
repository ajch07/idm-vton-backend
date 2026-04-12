# Runpod Deployment Guide

## Step 1: Install Docker

### Windows
1. Download [Docker Desktop](https://www.docker.com/products/docker-desktop/)
2. Install and restart your computer
3. Open PowerShell/CMD and verify:
```bash
docker --version
```

---

## Step 2: Create Docker Hub Account

1. Go to [https://hub.docker.com/](https://hub.docker.com/)
2. Sign up (free account)
3. Create a username and remember it

---

## Step 3: Build Docker Image

```bash
cd c:\Users\Anubhav choudhary\OneDrive\Desktop\try-cloth-tool\tryon-backend

# Build the image
docker build -t my-tryon-handler:v1 .
```

**What this does:**
- Downloads PyTorch + CUDA base image (~6GB)
- Installs all Python dependencies
- Copies your app code
- Creates a ready-to-run image

**First time**: 10-20 mins  
**Subsequent builds**: 2-5 mins (cached)

---

## Step 4: Push to Docker Hub

```bash
# Login to Docker Hub
docker login
# (Enter your username and password)

# Tag the image with your username
docker tag my-tryon-handler:v1 your-docker-username/my-tryon-handler:v1

# Push to Docker Hub
docker push your-docker-username/my-tryon-handler:v1
```

**Example:**
```bash
docker tag my-tryon-handler:v1 john123/my-tryon-handler:v1
docker push john123/my-tryon-handler:v1
```

This uploads the image (~5-10GB). Takes 10-30 mins depending on internet speed.

---

## Step 5: Create Runpod Account

1. Go to [https://www.runpod.io/](https://www.runpod.io/)
2. Sign up (free, verify email)
3. Add payment method (credit/debit card)
4. Go to [Runpod Console](https://www.runpod.io/console/)

---

## Step 6: Create Runpod Serverless Endpoint

1. Click **"Serverless" → "Create New Endpoint"**
2. Fill in details:

   | Field | Value |
   |-------|-------|
   | **Endpoint Name** | `my-tryon-endpoint` |
   | **Select Model** | Find your Docker Hub image: `your-docker-username/my-tryon-handler:v1` |
   | **GPU** | RTX 4090 (fastest, ~₹0.25/min) or RTX 3080 (~₹0.15/min) |
   | **Min Workers** | 0 |
   | **Max Workers** | 3 |
   | **Idle Timeout** | 5 minutes |
   | **Timeout** | 60 seconds |

3. Click **"Deploy"**
4. Wait 2-5 mins for deployment
5. Copy your **Endpoint ID** (looks like: `abc123def456`)

---

## Step 7: Get Runpod API Key

1. In Runpod Console, click **"Settings"**
2. Scroll down to **"API Keys"**
3. Click **"Generate API Key"**
4. Copy and save it somewhere safe

---

## Step 8: Configure Backend

```bash
# Edit your .env file
# Add these lines:

TRYON_SERVICE=hybrid
RUNPOD_ENDPOINT=https://api.runpod.io/v2/YOUR_ENDPOINT_ID/run
RUNPOD_API_KEY=YOUR_API_KEY
```

**Example:**
```bash
TRYON_SERVICE=hybrid
RUNPOD_ENDPOINT=https://api.runpod.io/v2/abc123def456/run
RUNPOD_API_KEY=sk-12345abcde67890fghijk
```

---

## Step 9: Test the Endpoint

```bash
# Go to backend folder
cd ../..  # Go to tryon-backend

# Test with curl (Windows PowerShell)
$body = @{
    user_image = "base64_of_user_image_here"
    garment_image = "base64_of_garment_image_here"
    garment_id = "test_123"
    garment_name = "Test Dress"
} | ConvertTo-Json

Invoke-WebRequest -Uri "https://api.runpod.io/v2/YOUR_ENDPOINT_ID/run" `
  -Method POST `
  -Headers @{"Content-Type"="application/json"; "Authorization"="Bearer YOUR_API_KEY"} `
  -Body $body
```

Or test from your frontend by making a request to `/api/try-on` endpoint.

---

## Step 10: Monitor Costs

### Expected Costs

| GPU | Time/Image | Cost/Image |
|-----|-----------|-----------|
| RTX 4090 | 4-6 mins | **₹0.15-0.20** |
| RTX 3080 | 6-8 mins | **₹0.10-0.15** |
| FAL (for comparison) | 2-5 mins | **₹12** |

**Savings**: ~98% cost reduction! 🎉

### Monitor Usage

1. Go to [Runpod Console](https://www.runpod.io/console/)
2. Click **"Serverless → Logs"**
3. View requests, costs, and performance

---

## Troubleshooting

### Docker build fails
```bash
# Clear Docker cache and rebuild
docker system prune -a
docker build -t my-tryon-handler:v1 .
```

### Push to Docker Hub fails
```bash
# Check Docker login
docker whoami

# Re-login if needed
docker logout
docker login
```

### Runpod endpoint not responding
1. Check **Runpod Console → Logs** for errors
2. Verify endpoint is **"Active"** (green status)
3. Check **Timeout** setting (should be 60 seconds)

### Slow inference on Runpod
1. Models need to download on first request (~10 mins)
2. Subsequent requests are faster (5-10 mins)
3. If still slow, check error logs in Runpod console

---

## Commands Reference

```bash
# Docker commands
docker build -t my-tryon-handler:v1 .
docker images  # List images
docker ps  # Running containers
docker logs CONTAINER_ID  # View logs

# Push to Docker Hub
docker login
docker tag my-tryon-handler:v1 user/my-tryon-handler:v1
docker push user/my-tryon-handler:v1

# Test locally before pushing
docker run -it --gpus all my-tryon-handler:v1
```

---

## Next Steps

After deployment:

1. ✅ Frontend can call `/api/try-on` endpoint
2. ✅ Backend routes to Runpod (hybrid mode with FAL fallback)
3. ✅ Images generated on Runpod, results returned to user
4. ✅ Pay only for GPU time (₹0.15-0.20 per image)

---

## Questions?

- Docker issues? Check [Docker docs](https://docs.docker.com/)
- Runpod issues? Check [Runpod docs](https://docs.runpod.io/)
- Backend integration? Check backend logs

**Good luck! 🚀**
