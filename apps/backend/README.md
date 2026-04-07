# Style Transfer AI Backend (Phase 1)

This implements Phase 1 backend capabilities from `style-transfer-onion-plan.md`, with a practical API surface for noob designers:

- prompt parsing into editable style controls
- image normalization and capability introspection
- generation config planning
- style transfer endpoint using Hugging Face Transformers.js on WebGPU

## Tech choices

- Fastify + TypeScript API server
- `@huggingface/transformers` with `device: "webgpu"`
- `image-to-image` ONNX stylization model via HF Hub
- style-image color/tone matching and structure-preserving blend for controllability

## Endpoints

- `GET /api/capabilities`
- `POST /api/parse-intent`
- `POST /api/normalize-image` (multipart)
- `POST /api/generation-config`
- `POST /api/style-transfer` (multipart)

`/api/style-transfer` form fields:

- `contentImage` (file, required)
- `styleImage` (file, required)
- `prompt` (string, optional)
- `qualityPreset` (`preview` or `export`, optional)
- `controls` (JSON string, optional overrides)

## Run

```bash
npm install
npm run dev
```

## Notes

- The server is stateless and does not persist history.
- WebGPU availability depends on runtime/browser/driver support.
- Style models are configurable with env vars:
  - `STYLE_MODEL_BALANCED`
  - `STYLE_MODEL_PAINTERLY`
  - `STYLE_MODEL_COLOR_POP`
