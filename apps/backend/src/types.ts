export type QualityPreset = "preview" | "export";

export interface StyleControls {
  color_transfer: number;
  texture_transfer: number;
  brushstroke_transfer: number;
}

export interface ParsedIntent {
  elements: StyleControls;
  explanation: string[];
  inferredPreset: "balanced" | "painterly" | "color-pop";
}

export const DEFAULT_CONTROLS: StyleControls = {
  color_transfer: 1.0,
  texture_transfer: 1.0,
  brushstroke_transfer: 1.0,
};
