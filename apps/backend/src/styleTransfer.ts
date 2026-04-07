import sharp from "sharp";
import { MODEL_INPUT_STRIDE, RESOLUTION_PRESETS } from "./config.js";
import { meanStd, normalizeImage, rawRgb } from "./image.js";
import type { ParsedIntent, QualityPreset, StyleControls } from "./types.js";

function clamp01(v: number): number {
  return Math.max(0, Math.min(1, v));
}

function clamp255(v: number): number {
  return Math.max(0, Math.min(255, Math.round(v)));
}

function luma(r: number, g: number, b: number): number {
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

function chromaMagnitude(r: number, g: number, b: number): number {
  const y = luma(r, g, b);
  const dr = r - y;
  const dg = g - y;
  const db = b - y;
  return Math.sqrt(dr * dr + dg * dg + db * db);
}

function blendChannel(a: number, b: number, alpha: number): number {
  return clamp255(a * (1 - alpha) + b * alpha);
}

async function applyColorTransfer(basePng: Buffer, stylePng: Buffer, intensity: number): Promise<Buffer> {
  const meta = await sharp(basePng).metadata();
  if (!meta.width || !meta.height) return basePng;

  const width = meta.width;
  const height = meta.height;
  const base = await rawRgb(basePng, width, height);
  const style = await rawRgb(stylePng, width, height);
  const baseStats = meanStd(base);
  const styleStats = meanStd(style);
  const t = clamp01(intensity);
  const out = new Uint8Array(base.length);

  for (let i = 0; i < base.length; i += 3) {
    const originalLuma = luma(base[i], base[i + 1], base[i + 2]);
    const matched: number[] = [0, 0, 0];

    for (let c = 0; c < 3; c += 1) {
      const normalized = (base[i + c] - baseStats.mean[c]) / baseStats.std[c];
      matched[c] = normalized * styleStats.std[c] + styleStats.mean[c];
    }

    // Keep scene brightness mostly stable while still allowing style chroma to come through.
    const matchedLuma = Math.max(1e-6, luma(matched[0], matched[1], matched[2]));
    const lumScale = originalLuma / matchedLuma;
    const lumaPreserveMix = 0.78 - t * 0.48; // high t -> more style color freedom
    const baseChroma = chromaMagnitude(base[i], base[i + 1], base[i + 2]);
    const matchedChroma = chromaMagnitude(matched[0], matched[1], matched[2]);
    const chromaGain = clamp01(0.65 + (matchedChroma / Math.max(1e-6, baseChroma + 1e-6)) * 0.35);

    for (let c = 0; c < 3; c += 1) {
      const lumPreserved = matched[c] * lumScale;
      const balanced = lumPreserved * lumaPreserveMix + matched[c] * (1 - lumaPreserveMix);
      const chromaBoosted = originalLuma + (balanced - originalLuma) * chromaGain;
      out[i + c] = blendChannel(base[i + c], chromaBoosted, t);
    }
  }

  return sharp(Buffer.from(out), { raw: { width, height, channels: 3 } }).png().toBuffer();
}

async function applyTextureTransfer(basePng: Buffer, stylePng: Buffer, intensity: number): Promise<Buffer> {
  const t = clamp01(intensity);
  if (t <= 0.01) return basePng;

  const baseMeta = await sharp(basePng).metadata();
  if (!baseMeta.width || !baseMeta.height) return basePng;
  const width = baseMeta.width;
  const height = baseMeta.height;

  const blurSigma = 2.0;
  const baseBlurPng = await sharp(basePng).blur(blurSigma).png().toBuffer();
  const styleBlurPng = await sharp(stylePng).blur(blurSigma).png().toBuffer();

  const base = await rawRgb(basePng, width, height);
  const style = await rawRgb(stylePng, width, height);
  const baseBlur = await rawRgb(baseBlurPng, width, height);
  const styleBlur = await rawRgb(styleBlurPng, width, height);

  const detailBase = new Float32Array(base.length);
  const detailStyle = new Float32Array(base.length);
  const sumSqBase = [0, 0, 0];
  const sumSqStyle = [0, 0, 0];
  const n = base.length / 3;

  for (let i = 0; i < base.length; i += 3) {
    for (let c = 0; c < 3; c += 1) {
      const db = base[i + c] - baseBlur[i + c];
      const ds = style[i + c] - styleBlur[i + c];
      detailBase[i + c] = db;
      detailStyle[i + c] = ds;
      sumSqBase[c] += db * db;
      sumSqStyle[c] += ds * ds;
    }
  }

  const stdBase = sumSqBase.map((x) => Math.sqrt(Math.max(1e-6, x / n)));
  const stdStyle = sumSqStyle.map((x) => Math.sqrt(Math.max(1e-6, x / n)));
  const gains = stdBase.map((b, c) => {
    const rawGain = stdStyle[c] / b;
    const clamped = Math.max(0.55, Math.min(1.9, rawGain));
    return 1 + (clamped - 1) * t;
  });

  const out = new Uint8Array(base.length);
  for (let i = 0; i < base.length; i += 3) {
    for (let c = 0; c < 3; c += 1) {
      const adjusted = baseBlur[i + c] + detailBase[i + c] * gains[c];
      out[i + c] = clamp255(adjusted);
    }
  }

  return sharp(Buffer.from(out), { raw: { width, height, channels: 3 } }).png().toBuffer();
}

async function applyBrushstrokeEffect(basePng: Buffer, intensity: number): Promise<Buffer> {
  const t = clamp01(intensity);
  if (t <= 0.01) return basePng;

  const meta = await sharp(basePng).metadata();
  if (!meta.width || !meta.height) return basePng;

  const width = meta.width;
  const height = meta.height;
  const downScale = 1 - t * 0.55;
  const w = Math.max(64, Math.floor(width * downScale));
  const h = Math.max(64, Math.floor(height * downScale));

  const painted = await sharp(basePng)
    .resize(w, h, { kernel: "lanczos3", fit: "fill" })
    .resize(width, height, { kernel: "nearest", fit: "fill" })
    .sharpen(1.2, 1, 2)
    .png()
    .toBuffer();

  const original = await rawRgb(basePng, width, height);
  const stylized = await rawRgb(painted, width, height);
  const out = new Uint8Array(original.length);

  for (let i = 0; i < original.length; i += 3) {
    out[i] = blendChannel(original[i], stylized[i], t);
    out[i + 1] = blendChannel(original[i + 1], stylized[i + 1], t);
    out[i + 2] = blendChannel(original[i + 2], stylized[i + 2], t);
  }

  return sharp(Buffer.from(out), { raw: { width, height, channels: 3 } }).png().toBuffer();
}

function clampCoord(v: number, max: number): number {
  if (v < 0) return 0;
  if (v > max) return max;
  return v;
}

async function applyStyleDrivenBrushstroke(basePng: Buffer, stylePng: Buffer, intensity: number): Promise<Buffer> {
  const t = clamp01(intensity);
  if (t <= 0.01) return basePng;

  const meta = await sharp(basePng).metadata();
  if (!meta.width || !meta.height) return basePng;
  const width = meta.width;
  const height = meta.height;

  const base = await rawRgb(basePng, width, height);
  const style = await rawRgb(stylePng, width, height);
  const styleBlurPng = await sharp(stylePng).blur(2.4).png().toBuffer();
  const styleBlur = await rawRgb(styleBlurPng, width, height);

  const styleLuma = new Float32Array(width * height);
  const styleDetail = new Float32Array(width * height);

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const i = (y * width + x) * 3;
      const p = y * width + x;
      const l = luma(style[i], style[i + 1], style[i + 2]);
      const lb = luma(styleBlur[i], styleBlur[i + 1], styleBlur[i + 2]);
      styleLuma[p] = l;
      styleDetail[p] = l - lb;
    }
  }

  const out = new Uint8Array(base.length);
  const strokeLen = Math.max(2, Math.round(2 + t * 5));
  const rippleStrength = 8 + t * 16;
  const brushBlend = 0.22 + t * 0.52;
  const dirGain = 0.16 + t * 0.34;

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const p = y * width + x;
      const i = p * 3;

      const xL = clampCoord(x - 1, width - 1);
      const xR = clampCoord(x + 1, width - 1);
      const yU = clampCoord(y - 1, height - 1);
      const yD = clampCoord(y + 1, height - 1);

      const gx = styleLuma[y * width + xR] - styleLuma[y * width + xL];
      const gy = styleLuma[yD * width + x] - styleLuma[yU * width + x];

      // Tangent direction (along stroke lines) from style gradients.
      let tx = -gy;
      let ty = gx;
      const mag = Math.sqrt(tx * tx + ty * ty) + 1e-6;
      tx /= mag;
      ty /= mag;

      // Use directional derivatives (not long-window averaging) to avoid blur.
      const fx = clampCoord(Math.round(x + tx * strokeLen), width - 1);
      const fy = clampCoord(Math.round(y + ty * strokeLen), height - 1);
      const bx = clampCoord(Math.round(x - tx * strokeLen), width - 1);
      const by = clampCoord(Math.round(y - ty * strokeLen), height - 1);
      const fi = (fy * width + fx) * 3;
      const bi = (by * width + bx) * 3;

      const nrx = clampCoord(Math.round(x + tx * 0 - ty * 2), width - 1);
      const nry = clampCoord(Math.round(y + ty * 0 + tx * 2), height - 1);
      const nlx = clampCoord(Math.round(x + tx * 0 + ty * 2), width - 1);
      const nly = clampCoord(Math.round(y + ty * 0 - tx * 2), height - 1);
      const nri = (nry * width + nrx) * 3;
      const nli = (nly * width + nlx) * 3;

      // Subtle wave/ripple from style detail to evoke internal swirls.
      const detail = styleDetail[p] / 128;
      const ripple = detail * rippleStrength;

      const dirR = (base[fi] - base[bi]) * dirGain;
      const dirG = (base[fi + 1] - base[bi + 1]) * dirGain;
      const dirB = (base[fi + 2] - base[bi + 2]) * dirGain;

      // Cross-normal component adds brush ridge feel without smearing.
      const ridgeR = (base[nri] - base[nli]) * 0.08 * t;
      const ridgeG = (base[nri + 1] - base[nli + 1]) * 0.08 * t;
      const ridgeB = (base[nri + 2] - base[nli + 2]) * 0.08 * t;

      const mixedR = base[i] + dirR + ridgeR + ripple;
      const mixedG = base[i + 1] + dirG + ridgeG + ripple * 0.9;
      const mixedB = base[i + 2] + dirB + ridgeB + ripple * 1.05;

      out[i] = blendChannel(base[i], mixedR, brushBlend);
      out[i + 1] = blendChannel(base[i + 1], mixedG, brushBlend);
      out[i + 2] = blendChannel(base[i + 2], mixedB, brushBlend);
    }
  }

  return sharp(Buffer.from(out), { raw: { width, height, channels: 3 } }).sharpen(1.15, 1, 2).png().toBuffer();
}

function computePipelineStrength(controls: StyleControls): number {
  return clamp01(controls.color_transfer * 0.34 + controls.texture_transfer * 0.33 + controls.brushstroke_transfer * 0.33);
}

export async function runStyleTransfer(params: {
  contentImage: Buffer;
  styleImage: Buffer;
  parsedIntent: ParsedIntent;
  qualityPreset: QualityPreset;
}): Promise<{ png: Buffer; modelId: string; width: number; height: number }> {
  const { contentImage, styleImage, parsedIntent, qualityPreset } = params;
  const preset = RESOLUTION_PRESETS[qualityPreset];
  const controls = parsedIntent.elements;

  const normalizedContent = await normalizeImage(contentImage, preset.longestSide, { stride: MODEL_INPUT_STRIDE });
  const normalizedStyle = await normalizeImage(styleImage, preset.longestSide, { stride: MODEL_INPUT_STRIDE });

  let current = normalizedContent.buffer;
  const styleStrength = computePipelineStrength(controls);

  current = await applyColorTransfer(current, normalizedStyle.buffer, clamp01(controls.color_transfer * (0.9 + styleStrength * 0.2)));
  current = await applyTextureTransfer(current, normalizedStyle.buffer, controls.texture_transfer);
  current = await applyBrushstrokeEffect(current, controls.brushstroke_transfer * 0.45);
  current = await applyStyleDrivenBrushstroke(current, normalizedStyle.buffer, controls.brushstroke_transfer);
  current = await applyColorTransfer(current, normalizedStyle.buffer, controls.color_transfer * 0.35);

  return {
    png: current,
    modelId: "classical-controls-v2",
    width: normalizedContent.width,
    height: normalizedContent.height,
  };
}
