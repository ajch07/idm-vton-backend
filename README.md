# Virtual Try-On Backend (FastAPI)

## Setup
```bash
cd tryon-backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Configure
Copy `.env.example` to `.env` and edit values:
```bash
cp .env.example .env
```

Set your FAL API key, database URL, JWT secret, and OAuth/payment keys. `.env` is loaded automatically on startup.

Environment variables:
- `FAL_API_KEY` (required)
- `FAL_MODEL` (optional, default `fal-ai/nano-banana`)
- `FAL_ENDPOINT` (optional, override full REST endpoint)
- `FAL_PROMPT_FIELD` (optional, default `prompt`)
- `FAL_NEGATIVE_FIELD` (optional, default `negative_prompt`)
- `FAL_IMAGE_FIELD` (optional, default `image_urls`)
- `FAL_USER_FIELD` + `FAL_GARMENT_FIELD` (optional, set both to use separate fields)
- `FAL_EXTRA_JSON` (optional, JSON object string merged into payload)
- `DATABASE_URL` (required, PostgreSQL)
- `JWT_SECRET` (required)
- `GOOGLE_CLIENT_ID` (required for Google login)
- `RAZORPAY_KEY_ID` + `RAZORPAY_KEY_SECRET` (required for payments)
- `ADMIN_EMAILS` (optional, comma-separated emails promoted to admin on signup)
- `CREDITS_SIGNUP` / `CREDITS_PER_TRYON` / `CREDITS_PER_PURCHASE` (optional)
- `SUPABASE_URL` (required for media uploads)
- `SUPABASE_SERVICE_ROLE_KEY` (required for media uploads)
- `SUPABASE_STORAGE_BUCKET` (optional, default `product-media`)

## Run
```bash
uvicorn main:app --reload --port 3001
```

## Endpoint
`POST /api/try-on`

Multipart fields:
- `userImage` (file)
- `garmentImage` (file)
- `garmentId` (string)
- `garmentName` (string)
- `prompt` (string, optional)
- `negativePrompt` (string, optional)

Health check: `GET /health`

## Auth + Admin
Auth:
- `POST /api/auth/register`
- `POST /api/auth/login`
- `POST /api/auth/google`
- `GET /api/auth/me`

Products:
- `GET /api/products`
- `GET /api/products/{slug}`
- `GET /api/admin/products` (admin, includes archived)
- `POST /api/admin/products` (admin)
- `PUT /api/admin/products/{id-or-slug}` (admin)
- `DELETE /api/admin/products/{id-or-slug}` (admin)

Media (admin):
- `GET /api/admin/products/{product_id}/media`
- `POST /api/admin/media/upload` (multipart form)
- `DELETE /api/admin/products/{product_id}/media/{media_id}`

Payments:
- `POST /api/payments/razorpay/order`
- `POST /api/payments/razorpay/verify`

Admin dashboard:
- `GET /api/admin/metrics`
- `GET /api/admin/users`
- `GET /api/admin/orders`
- `GET /api/admin/activity`
- `POST /api/admin/credits/grant`
