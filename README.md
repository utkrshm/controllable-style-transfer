# Controllable Style Transfer

A fully client-side neural style transfer app with independent controls for **color**, **texture**, and **brushstroke** scale. No backend, no account, no uploads — inference runs entirely in your browser via WebGPU.

## How it works

Uses Google Magenta's [arbitrary-image-stylization-v1-256](https://github.com/reiinakano/arbitrary-image-stylization-tfjs) model via TensorFlow.js. Inference runs on **WebGPU** with automatic **WebGL** fallback.

The three style controls map onto the Magenta pipeline as follows:

| Control | Range | What it does |
|---|---|---|
| **Color** | 0–100% | Luminance-preserving chroma mix (YCbCr). At 0% the output keeps the content image's original palette. Instant — no network re-run. |
| **Texture** | 0–100% | Blends the style embedding with an identity embedding. At 0% the output ≈ original content. |
| **Brushstroke** | Fine → Coarse | Resizes the style image before feeding it to the style network (192–512 px). Smaller = finer strokes. |

**Pipeline**:
1. Style prediction net → style embedding from the style image (resized by Brushstroke value).
2. Identity embedding from the content image.
3. Texture slider interpolates: `texture * styleEmbed + (1 - texture) * identityEmbed`.
4. Transfer net → raw stylized output.
5. Color post-process: YCbCr chroma of stylized vs content image interpolated by Color slider.

## Project structure

```
apps/
  frontend/         React + Vite + TypeScript
    src/
      App.tsx       Inference pipeline + UI
      styles.css    Design system (teal accent, Inter font)
      samples/      6 built-in sample SVGs (3 content, 3 style)
      main.tsx
    index.html
    vite.config.ts
package.json        Workspace root (frontend only)
```

## Requirements

- Node 20+, npm 9+
- Chrome 113+ / Edge 113+ for WebGPU. WebGL fallback works in all modern browsers.
- ~15–20 MB model download on first load (cached by the browser after that).

## Install & run

```bash
npm install
npm run dev       # → http://localhost:5173
```

## Build

```bash
npm run build     # output → apps/frontend/dist/
npm run preview   # serve the built output locally
```

## Browser support

| Backend | Browsers | Performance |
|---|---|---|
| WebGPU | Chrome 113+, Edge 113+, Safari 18+ | Best |
| WebGL | All modern browsers | Fallback — still usable |

The active backend is shown in the header badge.

## License

Personal project. Model weights are Google's under their TFHub terms.
