import sharp from "sharp";
import { MAX_RESOLUTION, MAX_UPLOAD_BYTES } from "./config.js";

export interface PreparedImage {
  buffer: Buffer;
  width: number;
  height: number;
  mimeType: "image/png";
}

interface NormalizeImageOptions {
  stride?: number;
}

export function assertUploadLimit(buffer: Buffer): void {
  if (buffer.byteLength > MAX_UPLOAD_BYTES) {
    throw new Error(`Image exceeds max upload size of ${MAX_UPLOAD_BYTES} bytes`);
  }
}

function snapToStride(value: number, stride: number): number {
  if (stride <= 1) return value;
  const snapped = Math.floor(value / stride) * stride;
  return Math.max(stride, snapped);
}

export async function normalizeImage(input: Buffer, longestSide: number, options: NormalizeImageOptions = {}): Promise<PreparedImage> {
  assertUploadLimit(input);
  const stride = options.stride ?? 1;

  const img = sharp(input, { failOn: "warning" }).rotate();
  const meta = await img.metadata();

  if (!meta.width || !meta.height) {
    throw new Error("Invalid image: width/height could not be determined");
  }

  if (meta.width > MAX_RESOLUTION * 3 || meta.height > MAX_RESOLUTION * 3) {
    throw new Error("Input image is too large to process safely");
  }

  const sourceWidth = meta.width;
  const sourceHeight = meta.height;
  const scale = longestSide / Math.max(sourceWidth, sourceHeight);
  const resizedWidth = Math.max(1, Math.round(sourceWidth * scale));
  const resizedHeight = Math.max(1, Math.round(sourceHeight * scale));
  const targetWidth = snapToStride(resizedWidth, stride);
  const targetHeight = snapToStride(resizedHeight, stride);

  const resized = await img
    .resize({
      width: targetWidth,
      height: targetHeight,
      fit: "fill",
    })
    .png()
    .toBuffer({ resolveWithObject: true });

  return {
    buffer: resized.data,
    width: resized.info.width,
    height: resized.info.height,
    mimeType: "image/png",
  };
}

export async function rawRgb(input: Buffer, width: number, height: number): Promise<Uint8Array> {
  const out = await sharp(input)
    .resize(width, height, { fit: "fill" })
    .removeAlpha()
    .raw()
    .toBuffer();
  return new Uint8Array(out);
}

export function meanStd(rgb: Uint8Array): { mean: [number, number, number]; std: [number, number, number] } {
  const sums = [0, 0, 0];
  const sqSums = [0, 0, 0];
  const count = rgb.length / 3;

  for (let i = 0; i < rgb.length; i += 3) {
    for (let c = 0; c < 3; c += 1) {
      const v = rgb[i + c];
      sums[c] += v;
      sqSums[c] += v * v;
    }
  }

  const mean = sums.map((x) => x / count) as [number, number, number];
  const std = sqSums.map((x, i) => Math.sqrt(Math.max(1e-6, x / count - mean[i] * mean[i]))) as [number, number, number];

  return { mean, std };
}
