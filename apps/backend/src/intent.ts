import { DEFAULT_CONTROLS, type ParsedIntent, type StyleControls } from "./types.js";

const POSITIVE_HINTS: Array<{ pattern: RegExp; updates: Partial<StyleControls>; reason: string }> = [
  {
    pattern: /(color|palette|vibrant|saturated|hue)/i,
    updates: { color_transfer: 0.9 },
    reason: "Detected emphasis on color palette and vibrance.",
  },
  {
    pattern: /(brush|painterly|paint|stroke|oil)/i,
    updates: { brushstroke_transfer: 0.9, texture_transfer: 0.65 },
    reason: "Detected painterly/brushstroke preference.",
  },
  {
    pattern: /(texture|grain|surface|fabric)/i,
    updates: { texture_transfer: 0.85 },
    reason: "Detected surface texture transfer intent.",
  },
  {
    pattern: /(soft|subtle|minimal|light touch)/i,
    updates: { brushstroke_transfer: 0.35, texture_transfer: 0.25, color_transfer: 0.55 },
    reason: "Detected subtle transfer preference.",
  },
];

const NEGATIVE_HINTS: Array<{ pattern: RegExp; updates: Partial<StyleControls>; reason: string }> = [
  {
    pattern: /(not too colorful|less color|desaturate)/i,
    updates: { color_transfer: 0.35 },
    reason: "Detected reduced color transfer request.",
  },
  {
    pattern: /(not too painterly|no brushstroke)/i,
    updates: { brushstroke_transfer: 0.15, texture_transfer: 0.2 },
    reason: "Detected reduced painterly effect request.",
  },
];

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function mergeControls(base: StyleControls, patch: Partial<StyleControls>): StyleControls {
  return {
    color_transfer: patch.color_transfer ?? base.color_transfer,
    texture_transfer: patch.texture_transfer ?? base.texture_transfer,
    brushstroke_transfer: patch.brushstroke_transfer ?? base.brushstroke_transfer,
  };
}

function normalize(controls: StyleControls): StyleControls {
  return {
    color_transfer: clamp01(controls.color_transfer),
    texture_transfer: clamp01(controls.texture_transfer),
    brushstroke_transfer: clamp01(controls.brushstroke_transfer),
  };
}

export function parseIntent(prompt: string): ParsedIntent {
  const text = (prompt ?? "").trim();
  const explanation: string[] = [];
  let controls = { ...DEFAULT_CONTROLS };

  if (!text) {
    explanation.push("No prompt provided. Applied beginner-safe balanced defaults.");
    return {
      elements: controls,
      explanation,
      inferredPreset: "balanced",
    };
  }

  for (const hint of POSITIVE_HINTS) {
    if (hint.pattern.test(text)) {
      controls = mergeControls(controls, hint.updates);
      explanation.push(hint.reason);
    }
  }

  for (const hint of NEGATIVE_HINTS) {
    if (hint.pattern.test(text)) {
      controls = mergeControls(controls, hint.updates);
      explanation.push(hint.reason);
    }
  }

  if (explanation.length === 0) {
    explanation.push("Prompt was broad. Applied balanced defaults for color, texture, and brushstroke.");
  }

  controls = normalize(controls);

  let inferredPreset: ParsedIntent["inferredPreset"] = "balanced";
  if (controls.brushstroke_transfer > 0.75 || controls.texture_transfer > 0.75) {
    inferredPreset = "painterly";
  } else if (controls.color_transfer > 0.8 && controls.brushstroke_transfer < 0.6) {
    inferredPreset = "color-pop";
  }

  return { elements: controls, explanation, inferredPreset };
}
