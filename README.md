# Style Transfer

A client-side neural style transfer web app. Upload a **content** image and a **style** image, pick a stylization strength, and get a stylized result — all rendered in your browser via WebGPU.

No backend. No account. No upload. The model runs on your machine.

## How It Works

- **Model**: Google's [Magenta `arbitrary-image-stylization-v1-256`](https://tfhub.dev/google/tfjs-model/magenta/arbitrary-image-stylization-v1-256/prediction/1), a pretrained feed-forward neural network.
- **Runtime**: [TensorFlow.js](https://www.tensorflow.org/js) with the WebGPU backend (falls back to WebGL if WebGPU is unavailable).
- **Pipeline**:
  1. Style prediction network → produces a style embedding from the style image.
  2. Identity embedding produced from the content image.
  3. Strength slider interpolates between them: `strength * style + (1 - strength) * identity`.
  4. Transfer network consumes the content image + blended embedding → stylized output.

## Project Structure

```text
style-transfer/
├─ apps/
│  └─ frontend/          # React + Vite UI — the whole app
│     └─ src/
│        ├─ App.tsx      # Model loading, inference, UI
│        ├─ styles.css   # Design system
│        ├─ samples/     # Bundled sample images
│        └─ main.tsx
├─ package.json          # Workspace scripts
└─ README.md
```

`apps/backend/` from an earlier architecture is preserved on disk but is not wired into any build or dev script.

## Setup

**Prerequisites**: Node.js 20+, npm 9+. A modern browser; Chrome/Edge 113+ for WebGPU.

```bash
npm install
```

## Run

```bash
npm run dev
```

Opens on `http://localhost:5173`.

On first visit the app downloads ~15–20 MB of model weights — subsequent loads are cached by the browser.

## Build

```bash
npm run build
```

Outputs a static bundle to `apps/frontend/dist/`. Serve it with any static host (e.g. `npm run preview`).

## Browser Support

| Backend | Browsers | Performance |
|---|---|---|
| WebGPU | Chrome 113+, Edge 113+, Safari 18+ | Best |
| WebGL | Everything modern | Fallback — still usable |

The app picks WebGPU first and silently falls back to WebGL if initialization fails. The active backend is shown in the header badge.

## License

Personal project. Model weights are Google's under their TFHub terms.
