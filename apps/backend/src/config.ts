import type { QualityPreset } from "./types.js";

export const MAX_UPLOAD_SIZE_MB = 10;
export const MAX_UPLOAD_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024;
export const MAX_RESOLUTION = 2048;
export const MODEL_INPUT_STRIDE = 8;

export const RESOLUTION_PRESETS: Record<QualityPreset, { longestSide: number; maxIterations: number }> = {
  preview: { longestSide: 512, maxIterations: 35 },
  export: { longestSide: 1024, maxIterations: 60 },
};

// These ONNX model-zoo models are fixed-style stylizers.
// We still use the user's style image for color/tone transfer after stylization.
export const STYLE_MODEL_MAP = {
  // Use Transformers.js-compatible image-to-image models by default.
  balanced: process.env.STYLE_MODEL_BALANCED ?? "Xenova/swin2SR-classical-sr-x2-64",
  painterly: process.env.STYLE_MODEL_PAINTERLY ?? "Xenova/swin2SR-compressed-sr-x4-48",
  color_pop: process.env.STYLE_MODEL_COLOR_POP ?? "Xenova/swin2SR-realworld-sr-x4-64-bsrgan-psnr",
} as const;
