# Controllable Style Transfer App

A personal project for controllable artistic style transfer, where users upload a **content image** and a **style image**, then tune transfer intensity for:

- Color
- Texture
- Brushstroke

---

## What It Is

This is a monorepo with:

- `apps/frontend`: React + Vite UI for uploading images, tuning controls, generating output, and downloading results.
- `apps/backend`: Fastify + Transformers.js inference service for style transfer.

The product is intentionally simple and no-auth, built for quick experimentation.

---

## How It Works

1. User uploads content + style images in the frontend.
2. User provides (optional) creative intent text.
3. Backend parses intent into control weights.
4. Backend runs style transfer using the selected weights.
5. Frontend receives and displays output image, with download support.

Current control dimensions:

- `color_transfer`
- `texture_transfer`
- `brushstroke_transfer`

---

## Project Structure

```text
style-transfer/
├─ apps/
│  ├─ frontend/   # React + Vite UI
│  └─ backend/    # Fastify API + style transfer pipeline
├─ package.json   # workspace scripts
└─ README.md
```

---

## Steps to Setup

### Prerequisites

- Node.js 18+ (Node 20 recommended)
- npm 9+

### 1. Install dependencies

```bash
npm install
```

### 2. Configure frontend env

Create/edit:

- `apps/frontend/.env`

Example:

```bash
VITE_BACKEND_URL=http://localhost:3000
```

If `VITE_BACKEND_URL` is empty, frontend uses relative `/api` paths (works with local proxy in dev).

### 3. Run backend

```bash
npm run dev:backend
```

Backend default: `http://localhost:3000`
Health check: `GET /api/health`

### 4. Run frontend

```bash
npm run dev:frontend
```

Frontend default: `http://localhost:5173`

---

## Scripts

- `npm run dev:frontend` — run frontend dev server
- `npm run dev:backend` — run backend dev server
- `npm run landing` — alias for frontend dev server
- `npm run build` — build frontend and backend

---

## Demo

Demo video:

---

## Known Notes

- This project is currently in a local-first/dev-friendly stage.
- Backend inference is server-side (not pure client-side inference).
- Performance depends on server hardware for style transfer.

---

## Attribution

### Font

- Font: `Hiro Misake`
- License: `Freeware, Non-Commercial`
- Source: https://www.fontspace.com/hiro-misake-font-f160364
