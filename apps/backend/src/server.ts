import Fastify from "fastify";
import multipart from "@fastify/multipart";
import { z } from "zod";
import { MAX_RESOLUTION, MAX_UPLOAD_SIZE_MB, RESOLUTION_PRESETS } from "./config.js";
import { normalizeImage } from "./image.js";
import { parseIntent } from "./intent.js";
import { runStyleTransfer } from "./styleTransfer.js";
import { DEFAULT_CONTROLS, type QualityPreset, type StyleControls } from "./types.js";

const app = Fastify({ logger: true });

const corsOrigins = (process.env.CORS_ORIGIN ?? "")
  .split(",")
  .map((origin) => origin.trim())
  .filter(Boolean);

function resolveCorsOrigin(originHeader?: string): string {
  if (!originHeader) return "*";
  if (corsOrigins.length === 0) return originHeader;
  return corsOrigins.includes(originHeader) ? originHeader : "";
}

await app.register(multipart, {
  limits: {
    fileSize: MAX_UPLOAD_SIZE_MB * 1024 * 1024,
    files: 2,
  },
});

app.addHook("onRequest", async (request, reply) => {
  const requestOrigin = request.headers.origin;
  const allowedOrigin = resolveCorsOrigin(requestOrigin);

  if (allowedOrigin) {
    reply.header("Access-Control-Allow-Origin", allowedOrigin);
    reply.header("Vary", "Origin");
  }
  reply.header("Access-Control-Allow-Methods", "GET,POST,OPTIONS");
  reply.header("Access-Control-Allow-Headers", "Content-Type, Authorization");

  if (request.method === "OPTIONS") {
    return reply.code(204).send();
  }
});

const parseIntentSchema = z.object({
  prompt: z.string().default(""),
});

const generationConfigSchema = z.object({
  qualityPreset: z.enum(["preview", "export"]).default("preview"),
  elementWeights: z
    .object({
      color_transfer: z.number().min(0).max(1).optional(),
      texture_transfer: z.number().min(0).max(1).optional(),
      brushstroke_transfer: z.number().min(0).max(1).optional(),
    })
    .optional(),
});

function mergeControls(overrides?: Partial<StyleControls>): StyleControls {
  return {
    color_transfer: overrides?.color_transfer ?? DEFAULT_CONTROLS.color_transfer,
    texture_transfer: overrides?.texture_transfer ?? DEFAULT_CONTROLS.texture_transfer,
    brushstroke_transfer: overrides?.brushstroke_transfer ?? DEFAULT_CONTROLS.brushstroke_transfer,
  };
}

app.get("/api/capabilities", async () => {
  return {
    supportedElements: [
      "color_transfer",
      "texture_transfer",
      "brushstroke_transfer",
    ],
    limits: {
      maxUploadSizeMb: MAX_UPLOAD_SIZE_MB,
      maxResolution: MAX_RESOLUTION,
      qualityPresets: Object.keys(RESOLUTION_PRESETS),
      runtime: process.env.HF_DEVICE ?? "dml",
      backend: "huggingface-transformers-js",
    },
  };
});

app.post("/api/parse-intent", async (request) => {
  const body = parseIntentSchema.parse(request.body ?? {});
  return parseIntent(body.prompt);
});

app.post("/api/normalize-image", async (request, reply) => {
  const parts = request.parts();
  for await (const part of parts) {
    if (part.type === "file") {
      const data = await part.toBuffer();
      const normalized = await normalizeImage(data, RESOLUTION_PRESETS.preview.longestSide);
      return {
        imageId: `temp-${Date.now()}`,
        width: normalized.width,
        height: normalized.height,
        mimeType: normalized.mimeType,
      };
    }
  }
  return reply.code(400).send({ error: "Image file is required" });
});

app.post("/api/generation-config", async (request) => {
  const body = generationConfigSchema.parse(request.body ?? {});
  const controls = mergeControls(body.elementWeights);

  const target = RESOLUTION_PRESETS[body.qualityPreset];
  const styleIntensity =
    controls.color_transfer * 0.34 + controls.texture_transfer * 0.33 + controls.brushstroke_transfer * 0.33;

  return {
    targetResolution: [target.longestSide, target.longestSide],
    maxIterations: target.maxIterations,
    engine: "classical-controls-v2",
    styleIntensity: Number(styleIntensity.toFixed(3)),
  };
});

app.post("/api/style-transfer", async (request, reply) => {
  const parts = request.parts();

  let contentImage: Buffer | null = null;
  let styleImage: Buffer | null = null;
  let prompt = "";
  let qualityPreset: QualityPreset = "preview";
  let controlsOverride: Partial<StyleControls> | null = null;

  for await (const part of parts) {
    if (part.type === "file") {
      if (part.fieldname === "contentImage") {
        contentImage = await part.toBuffer();
      } else if (part.fieldname === "styleImage") {
        styleImage = await part.toBuffer();
      }
      continue;
    }

    const value = String(part.value ?? "");
    if (part.fieldname === "prompt") prompt = value;
    if (part.fieldname === "qualityPreset" && (value === "preview" || value === "export")) {
      qualityPreset = value;
    }
    if (part.fieldname === "controls" && value.trim()) {
      try {
        controlsOverride = JSON.parse(value) as Partial<StyleControls>;
      } catch {
        return reply.code(400).send({ error: "Invalid controls JSON payload" });
      }
    }
  }

  if (!contentImage || !styleImage) {
    return reply.code(400).send({ error: "Both contentImage and styleImage are required" });
  }

  const parsedIntent = parseIntent(prompt);
  if (controlsOverride) {
    parsedIntent.elements = mergeControls({
      ...parsedIntent.elements,
      ...controlsOverride,
    });
    parsedIntent.explanation.push("Applied manual slider overrides on top of prompt parsing.");
  }

  const out = await runStyleTransfer({
    contentImage,
    styleImage,
    parsedIntent,
    qualityPreset,
  });

  return reply.send({
    mimeType: "image/png",
    imageBase64: out.png.toString("base64"),
    width: out.width,
    height: out.height,
    modelId: out.modelId,
    parsedIntent,
    qualityPreset,
  });
});

app.get("/api/health", async () => ({ ok: true }));

const port = Number(process.env.PORT ?? 3000);
await app.listen({ host: "0.0.0.0", port });
