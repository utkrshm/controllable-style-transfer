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
  color_transfer: 0.7,
  texture_transfer: 0.4,
  brushstroke_transfer: 0.5,
};
