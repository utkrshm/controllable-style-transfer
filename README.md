# Style Transfer Monorepo

## Apps

- `apps/frontend`: React + Vite landing/studio UI
- `apps/backend`: Fastify AI backend (WebGPU + Hugging Face Transformers)

## Run landing page

```bash
npm run landing
```

## Run backend

```bash
npm run dev:backend
```

## Deploy Frontend to GitHub Pages

The workflow file [deploy-frontend-pages.yml](/mnt/c/Users/Utkarsh/Temp/style-transfer/.github/workflows/deploy-frontend-pages.yml) deploys `apps/frontend` to:

- `https://<username>.github.io/<repo-name>/`

### 1. Set Backend URL for frontend

Set repository secret:

- `Settings` -> `Secrets and variables` -> `Actions` -> `New repository secret`
- Name: `BACKEND_URL`
- Value example: `https://your-backend.onrender.com`

This gets injected into frontend build as `VITE_BACKEND_URL`.

### 2. Configure backend CORS

Set backend environment variable:

- `CORS_ORIGIN=https://<username>.github.io`

If needed, pass comma-separated origins (for example staging + prod).

### 3. Local frontend env

Copy or edit:

- `apps/frontend/.env`
- `apps/frontend/.env.example`

`VITE_BACKEND_URL` should point to backend URL. For local dev you can keep:

- `http://localhost:3000`

## Font Attribution

- Font: `Hiro Misake`
- License: `Freeware, Non-Commercial`
- Source: https://www.fontspace.com/hiro-misake-font-f160364
