import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
} from "react";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgpu";

import contentCity from "./samples/content-city.svg";
import contentMountain from "./samples/content-mountain.svg";
import contentPortrait from "./samples/content-portrait.svg";
import styleStarry from "./samples/style-starry.svg";
import styleWave from "./samples/style-wave.svg";
import styleMosaic from "./samples/style-mosaic.svg";

const STYLE_NET_URL =
  "https://reiinakano.github.io/arbitrary-image-stylization-tfjs/saved_model_style_js/model.json";
const TRANSFORM_NET_URL =
  "https://reiinakano.github.io/arbitrary-image-stylization-tfjs/saved_model_transformer_js/model.json";
const MAX_CONTENT_SIZE = 800;
const SOURCE_DOWNSCALE_THRESHOLD = 4096;
const LIVE_DEBOUNCE_MS = 250;
const COLOR_DEBOUNCE_MS = 60;
const LIVE_MAX_DURATION_MS = 2500;

type AppStatus = "loading-backend" | "loading-models" | "ready" | "error";
type ViewMode = "result" | "compare";
type ErrorScope = "global" | "transfer" | null;
type UploadKind = "content" | "style";

const CONTENT_SAMPLES = [
  { url: contentCity, label: "City" },
  { url: contentMountain, label: "Mountain" },
  { url: contentPortrait, label: "Portrait" },
];

const STYLE_SAMPLES = [
  { url: styleStarry, label: "Starry" },
  { url: styleWave, label: "Wave" },
  { url: styleMosaic, label: "Mosaic" },
];

function brushLabel(n: number): string {
  if (n <= 224) return "Fine";
  if (n <= 320) return "Medium";
  return "Coarse";
}

function loadImg(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("Failed to load image"));
    img.src = src;
  });
}

function downscaleSource(
  img: HTMLImageElement
): HTMLCanvasElement | HTMLImageElement {
  const maxDim = Math.max(img.naturalWidth, img.naturalHeight);
  if (maxDim <= SOURCE_DOWNSCALE_THRESHOLD) return img;
  const scale = SOURCE_DOWNSCALE_THRESHOLD / maxDim;
  const w = Math.round(img.naturalWidth * scale);
  const h = Math.round(img.naturalHeight * scale);
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d")!;
  ctx.drawImage(img, 0, 0, w, h);
  return canvas;
}

// ── YCbCr helpers (BT.601) ────────────────────────────
function rgbToYCbCr(
  rgb: tf.Tensor3D
): [tf.Tensor2D, tf.Tensor2D, tf.Tensor2D] {
  const channels = tf.split(rgb, 3, 2) as tf.Tensor3D[];
  const r = channels[0].squeeze([2]) as tf.Tensor2D;
  const g = channels[1].squeeze([2]) as tf.Tensor2D;
  const b = channels[2].squeeze([2]) as tf.Tensor2D;
  const y = r.mul(0.299).add(g.mul(0.587)).add(b.mul(0.114)) as tf.Tensor2D;
  const cb = r
    .mul(-0.168736)
    .add(g.mul(-0.331264))
    .add(b.mul(0.5))
    .add(0.5) as tf.Tensor2D;
  const cr = r.mul(0.5).add(g.mul(-0.418688)).add(b.mul(-0.081312)).add(0.5) as tf.Tensor2D;
  channels.forEach((c) => c.dispose());
  return [y, cb, cr];
}

function yCbCrToRgb(
  y: tf.Tensor2D,
  cb: tf.Tensor2D,
  cr: tf.Tensor2D
): tf.Tensor3D {
  const cbC = cb.sub(0.5);
  const crC = cr.sub(0.5);
  const r = y.add(crC.mul(1.402));
  const g = y.add(cbC.mul(-0.344136)).add(crC.mul(-0.714136));
  const bCh = y.add(cbC.mul(1.772));
  cbC.dispose();
  crC.dispose();
  return tf.stack([r, g, bCh], 2).clipByValue(0, 1) as tf.Tensor3D;
}

function applyColorMix(
  stylized: tf.Tensor3D,
  content: tf.Tensor3D,
  colorAmt: number
): tf.Tensor3D {
  if (colorAmt >= 0.999) return stylized.clone();
  return tf.tidy(() => {
    const [yS, cbS, crS] = rgbToYCbCr(stylized);
    const [, cbC, crC] = rgbToYCbCr(content);
    const cbMix = cbS.mul(colorAmt).add(cbC.mul(1 - colorAmt));
    const crMix = crS.mul(colorAmt).add(crC.mul(1 - colorAmt));
    [cbS, crS, cbC, crC].forEach((t) => t.dispose());
    return yCbCrToRgb(yS, cbMix as tf.Tensor2D, crMix as tf.Tensor2D);
  });
}

export default function App() {
  const [status, setStatus] = useState<AppStatus>("loading-backend");
  const [backendName, setBackendName] = useState("");
  const [modelProgress, setModelProgress] = useState(0);

  const [errorScope, setErrorScope] = useState<ErrorScope>(null);
  const [errorMsg, setErrorMsg] = useState("");

  const [contentSrc, setContentSrc] = useState("");
  const [styleSrc, setStyleSrc] = useState("");
  const [contentName, setContentName] = useState("");
  const [styleName, setStyleName] = useState("");

  // Three independent style controls
  const [colorAmt, setColorAmt] = useState(1.0);
  const [textureAmt, setTextureAmt] = useState(0.85);
  const [brushSize, setBrushSize] = useState(256);

  const [processing, setProcessing] = useState(false);
  const [hasOutput, setHasOutput] = useState(false);
  const [dragTarget, setDragTarget] = useState<UploadKind | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>("result");
  const [liveMode, setLiveMode] = useState(true);
  const [wasDownscaled, setWasDownscaled] = useState(false);

  const styleNetRef = useRef<tf.GraphModel | null>(null);
  const transformNetRef = useRef<tf.GraphModel | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const runIdRef = useRef(0);
  const lastDurationRef = useRef(0);
  const contentUrlRef = useRef("");
  const styleUrlRef = useRef("");

  // Cached tensors for the color-only fast path
  const lastRawRef = useRef<tf.Tensor3D | null>(null);
  const lastContentRef = useRef<tf.Tensor3D | null>(null);

  const clearCachedTensors = () => {
    lastRawRef.current?.dispose();
    lastRawRef.current = null;
    lastContentRef.current?.dispose();
    lastContentRef.current = null;
  };

  // ── Init backend + load models ──────────────────────
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        let backend = "webgpu";
        try {
          await tf.setBackend("webgpu");
          await tf.ready();
        } catch {
          backend = "webgl";
          await tf.setBackend("webgl");
          await tf.ready();
        }
        if (cancelled) return;
        setBackendName(backend);
        setStatus("loading-models");

        let p1 = 0;
        let p2 = 0;
        const update = () => setModelProgress((p1 + p2) / 2);

        const [sNet, tNet] = await Promise.all([
          tf.loadGraphModel(STYLE_NET_URL, {
            onProgress: (f) => {
              p1 = f;
              update();
            },
          }),
          tf.loadGraphModel(TRANSFORM_NET_URL, {
            onProgress: (f) => {
              p2 = f;
              update();
            },
          }),
        ]);
        if (cancelled) return;
        styleNetRef.current = sNet;
        transformNetRef.current = tNet;
        setModelProgress(1);
        setStatus("ready");
      } catch (e) {
        if (cancelled) return;
        setErrorScope("global");
        setErrorMsg(e instanceof Error ? e.message : "Initialization failed");
        setStatus("error");
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // ── Cleanup object URLs + tensors on unmount ────────
  useEffect(() => {
    return () => {
      if (contentUrlRef.current) URL.revokeObjectURL(contentUrlRef.current);
      if (styleUrlRef.current) URL.revokeObjectURL(styleUrlRef.current);
      clearCachedTensors();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const setSource = useCallback(
    (type: UploadKind, src: string, name: string, isObjectUrl: boolean) => {
      if (type === "content") {
        if (contentUrlRef.current) URL.revokeObjectURL(contentUrlRef.current);
        contentUrlRef.current = isObjectUrl ? src : "";
        setContentSrc(src);
        setContentName(name);
      } else {
        if (styleUrlRef.current) URL.revokeObjectURL(styleUrlRef.current);
        styleUrlRef.current = isObjectUrl ? src : "";
        setStyleSrc(src);
        setStyleName(name);
      }
      setHasOutput(false);
      setWasDownscaled(false);
      clearCachedTensors();
    },
    []
  );

  const handleFile = useCallback(
    (type: UploadKind, file: File) => {
      if (!file.type.startsWith("image/")) return;
      setSource(type, URL.createObjectURL(file), file.name, true);
    },
    [setSource]
  );

  const onDrop = useCallback(
    (type: UploadKind, e: React.DragEvent) => {
      e.preventDefault();
      setDragTarget(null);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(type, file);
    },
    [handleFile]
  );

  const onFileInput = useCallback(
    (type: UploadKind, e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) handleFile(type, file);
      e.target.value = "";
    },
    [handleFile]
  );

  const clearSource = useCallback(
    (type: UploadKind, e: React.MouseEvent) => {
      e.preventDefault();
      e.stopPropagation();
      if (type === "content") {
        if (contentUrlRef.current) URL.revokeObjectURL(contentUrlRef.current);
        contentUrlRef.current = "";
        setContentSrc("");
        setContentName("");
      } else {
        if (styleUrlRef.current) URL.revokeObjectURL(styleUrlRef.current);
        styleUrlRef.current = "";
        setStyleSrc("");
        setStyleName("");
      }
      setHasOutput(false);
      setWasDownscaled(false);
      clearCachedTensors();
    },
    []
  );

  const pickSample = useCallback(
    (type: UploadKind, url: string, label: string) => {
      setSource(type, url, label, false);
    },
    [setSource]
  );

  // ── Color-only repaint (no network re-run) ──────────
  const repaintWithColor = useCallback(async () => {
    const raw = lastRawRef.current;
    const content = lastContentRef.current;
    const canvas = canvasRef.current;
    if (!raw || !content || !canvas) return;
    const final = applyColorMix(raw, content, colorAmt);
    await tf.browser.toPixels(final, canvas);
    final.dispose();
  }, [colorAmt]);

  // ── Run full style transfer ─────────────────────────
  const runTransfer = useCallback(async () => {
    const sNet = styleNetRef.current;
    const tNet = transformNetRef.current;
    if (!sNet || !tNet || !contentSrc || !styleSrc) return;

    const myId = ++runIdRef.current;
    const started = performance.now();

    setProcessing(true);
    setErrorScope(null);
    setErrorMsg("");

    try {
      const [contentImgRaw, styleImg] = await Promise.all([
        loadImg(contentSrc),
        loadImg(styleSrc),
      ]);
      if (myId !== runIdRef.current) return;

      const downscaled = downscaleSource(contentImgRaw);
      const didDownscale = downscaled !== contentImgRaw;
      setWasDownscaled(didDownscale);

      const sourceW =
        downscaled instanceof HTMLCanvasElement
          ? downscaled.width
          : downscaled.naturalWidth;
      const sourceH =
        downscaled instanceof HTMLCanvasElement
          ? downscaled.height
          : downscaled.naturalHeight;

      let w = sourceW;
      let h = sourceH;
      if (Math.max(w, h) > MAX_CONTENT_SIZE) {
        const scale = MAX_CONTENT_SIZE / Math.max(w, h);
        w = Math.round(w * scale);
        h = Math.round(h * scale);
      }

      // Run network; keep raw stylized + resized content for color fast-path
      const contentRawTensor = tf.tidy(() =>
        tf.browser.fromPixels(downscaled).toFloat().div(255)
      ) as tf.Tensor3D;

      const rawStylized = tf.tidy(() => {
        const styleRaw = tf.browser.fromPixels(styleImg).toFloat().div(255);

        const content4d = tf.image
          .resizeBilinear(contentRawTensor as tf.Tensor3D, [h, w])
          .expandDims(0);
        // Brushstroke: style resize dimension controls stroke scale
        const style4d = tf.image
          .resizeBilinear(styleRaw as tf.Tensor3D, [brushSize, brushSize])
          .expandDims(0);
        const content256 = tf.image
          .resizeBilinear(contentRawTensor as tf.Tensor3D, [256, 256])
          .expandDims(0);

        const styleEmbed = sNet.predict(style4d) as tf.Tensor;
        const identityEmbed = sNet.predict(content256) as tf.Tensor;

        // Texture: blend style vs identity embedding
        const blended = tf.add(
          tf.mul(styleEmbed, textureAmt),
          tf.mul(identityEmbed, 1 - textureAmt)
        );

        const stylized = tNet.predict([content4d, blended]) as tf.Tensor;
        return stylized.squeeze() as tf.Tensor3D;
      }) as tf.Tensor3D;

      // Resized content at output resolution (for color mixing)
      const resizedContent = tf.tidy(() =>
        tf.image
          .resizeBilinear(contentRawTensor as tf.Tensor3D, [h, w])
          .clipByValue(0, 1)
      ) as tf.Tensor3D;

      contentRawTensor.dispose();

      if (myId !== runIdRef.current) {
        rawStylized.dispose();
        resizedContent.dispose();
        return;
      }

      // Cache for color fast-path
      clearCachedTensors();
      lastRawRef.current = rawStylized;
      lastContentRef.current = resizedContent;

      // Color: luminance-preserving chroma mix
      const canvas = canvasRef.current!;
      canvas.width = w;
      canvas.height = h;
      const final = applyColorMix(rawStylized, resizedContent, colorAmt);
      await tf.browser.toPixels(final, canvas);
      final.dispose();

      if (myId !== runIdRef.current) return;

      const duration = performance.now() - started;
      lastDurationRef.current = duration;
      if (duration > LIVE_MAX_DURATION_MS) setLiveMode(false);

      setHasOutput(true);
    } catch (e) {
      if (myId !== runIdRef.current) return;
      setErrorScope("transfer");
      setErrorMsg(e instanceof Error ? e.message : "Style transfer failed");
    } finally {
      if (myId === runIdRef.current) setProcessing(false);
    }
  }, [contentSrc, styleSrc, textureAmt, brushSize, colorAmt]);

  // ── Debounced live re-render on texture/brush change ─
  useEffect(() => {
    if (!hasOutput || !liveMode || processing) return;
    const t = setTimeout(() => {
      runTransfer();
    }, LIVE_DEBOUNCE_MS);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [textureAmt, brushSize]);

  // ── Debounced color-only repaint ────────────────────
  useEffect(() => {
    if (!hasOutput || processing) return;
    const t = setTimeout(() => {
      repaintWithColor();
    }, COLOR_DEBOUNCE_MS);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [colorAmt]);

  const download = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !hasOutput) return;
    const a = document.createElement("a");
    a.download = "stylized.png";
    a.href = canvas.toDataURL("image/png");
    a.click();
  }, [hasOutput]);

  const canRun =
    status === "ready" && !!contentSrc && !!styleSrc && !processing;

  const onWorkspaceKeyDown = useCallback(
    (e: KeyboardEvent<HTMLElement>) => {
      if (e.key === "Enter" && canRun) {
        runTransfer();
      } else if (e.key === "Escape") {
        (document.activeElement as HTMLElement | null)?.blur?.();
      }
    },
    [canRun, runTransfer]
  );

  const colorFill = useMemo(
    () => ({ ["--fill" as string]: `${Math.round(colorAmt * 100)}%` }),
    [colorAmt]
  );
  const textureFill = useMemo(
    () => ({ ["--fill" as string]: `${Math.round(textureAmt * 100)}%` }),
    [textureAmt]
  );
  const brushFill = useMemo(
    () => ({
      ["--fill" as string]: `${Math.round(((brushSize - 192) / 320) * 100)}%`,
    }),
    [brushSize]
  );

  const showLiveDot = hasOutput && liveMode && status === "ready";

  const dismissError = () => {
    setErrorScope(null);
    setErrorMsg("");
  };

  return (
    <div className="app">
      <header className="header">
        <div className="header-inner">
          <div className="logo">
            <svg
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <polygon points="12 2 2 7 12 12 22 7 12 2" />
              <polyline points="2 17 12 22 22 17" />
              <polyline points="2 12 12 17 22 12" />
            </svg>
            <span>Controllable Style Transfer</span>
          </div>
          <div
            className={`badge badge-${
              status === "ready"
                ? "ok"
                : status === "error"
                ? "err"
                : "wait"
            }`}
          >
            {(status === "loading-backend" ||
              status === "loading-models") && (
              <span className="spinner sm" />
            )}
            {status === "loading-backend" && "Initializing GPU…"}
            {status === "loading-models" && (
              <>
                Downloading
                <span className="badge-progress">
                  <span
                    className="badge-progress-fill"
                    style={{ width: `${Math.round(modelProgress * 100)}%` }}
                  />
                </span>
              </>
            )}
            {status === "ready" && (
              <>
                Ready{" "}
                <span className="backend-tag">
                  {backendName.toUpperCase()}
                </span>
              </>
            )}
            {status === "error" && "Failed"}
          </div>
        </div>
      </header>

      <div className="disclaimer">
        This app runs neural style transfer entirely in your browser using <strong>WebGPU</strong> (with WebGL fallback). No images are uploaded to any server.
      </div>

      <main className="main" onKeyDown={onWorkspaceKeyDown}>
        {errorScope === "global" && errorMsg && (
          <div className="banner" role="alert">
            <span className="banner-msg">{errorMsg}</span>
            <button
              className="banner-close"
              onClick={dismissError}
              aria-label="Dismiss"
            >
              ×
            </button>
          </div>
        )}

        <div className="workspace">
          {/* ── Left panel: inputs + controls ── */}
          <section className="card left">
            <h2>Input</h2>

            {errorScope === "transfer" && errorMsg && (
              <div className="banner banner-inline" role="alert">
                <span className="banner-msg">{errorMsg}</span>
                <button
                  className="banner-close"
                  onClick={dismissError}
                  aria-label="Dismiss"
                >
                  ×
                </button>
              </div>
            )}

            <div className="upload-row">
              {(["content", "style"] as const).map((type) => {
                const src = type === "content" ? contentSrc : styleSrc;
                const name =
                  type === "content" ? contentName : styleName;
                const samples =
                  type === "content" ? CONTENT_SAMPLES : STYLE_SAMPLES;
                const id = `${type}-file`;
                return (
                  <div className="upload-zone" key={type}>
                    <span className="upload-label">
                      {type === "content" ? "Content" : "Style"}
                    </span>
                    <label
                      htmlFor={id}
                      className={`upload-area${
                        dragTarget === type ? " drag-over" : ""
                      }${src ? " filled" : ""}`}
                      onDragOver={(e) => {
                        e.preventDefault();
                        setDragTarget(type);
                      }}
                      onDragLeave={() => setDragTarget(null)}
                      onDrop={(e) => onDrop(type, e)}
                    >
                      {src ? (
                        <>
                          <img src={src} alt={type} className="thumb" />
                          <button
                            type="button"
                            className="clear-btn"
                            onClick={(e) => clearSource(type, e)}
                            aria-label={`Clear ${type}`}
                          >
                            ×
                          </button>
                        </>
                      ) : (
                        <div className="placeholder">
                          <svg
                            width="24"
                            height="24"
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="1.5"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                          >
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                            <polyline points="17 8 12 3 7 8" />
                            <line x1="12" y1="3" x2="12" y2="15" />
                          </svg>
                          <span>Drop or click</span>
                        </div>
                      )}
                      <input
                        id={id}
                        type="file"
                        accept="image/*"
                        className="sr-only"
                        onChange={(e) => onFileInput(type, e)}
                      />
                    </label>
                    {name && <span className="filename">{name}</span>}
                    <div className="sample-row">
                      <span className="sample-row-label">Samples</span>
                      {samples.map((s) => (
                        <button
                          key={s.url}
                          type="button"
                          className="sample-thumb"
                          onClick={() => pickSample(type, s.url, s.label)}
                          aria-label={`Use ${s.label} sample`}
                          title={s.label}
                        >
                          <img src={s.url} alt={s.label} />
                        </button>
                      ))}
                    </div>
                  </div>
                );
              })}
            </div>

            {/* ── Style controls ── */}
            <div className="control-group">
              <div className="control-row">
                <span className="control-label">
                  Style controls
                  {showLiveDot && (
                    <span className="live-dot" title="Live updates" />
                  )}
                </span>
              </div>
              <p className="control-helper">
                Dial the style's color, texture, and stroke size independently.
              </p>

              <div className="control-row">
                <span className="control-label">Color</span>
                <span className="control-value">
                  {Math.round(colorAmt * 100)}%
                </span>
              </div>
              <input
                type="range"
                min={0}
                max={1}
                step={0.01}
                value={colorAmt}
                onChange={(e) => setColorAmt(Number(e.target.value))}
                className="slider"
                style={colorFill}
                aria-label="Color transfer amount"
              />

              <div className="control-row">
                <span className="control-label">Texture</span>
                <span className="control-value">
                  {Math.round(textureAmt * 100)}%
                </span>
              </div>
              <input
                type="range"
                min={0}
                max={1}
                step={0.01}
                value={textureAmt}
                onChange={(e) => setTextureAmt(Number(e.target.value))}
                className="slider"
                style={textureFill}
                aria-label="Texture transfer amount"
              />

              <div className="control-row">
                <span className="control-label">Brushstroke</span>
                <span className="control-value">{brushLabel(brushSize)}</span>
              </div>
              <input
                type="range"
                min={192}
                max={512}
                step={32}
                value={brushSize}
                onChange={(e) => setBrushSize(Number(e.target.value))}
                className="slider"
                style={brushFill}
                aria-label="Brushstroke scale"
              />
            </div>

            <button
              className="btn-primary"
              disabled={!canRun}
              onClick={runTransfer}
            >
              {processing ? (
                <>
                  <span className="spinner" /> Processing…
                </>
              ) : (
                "Apply Style"
              )}
            </button>
          </section>

          {/* ── Right panel: output ── */}
          <section className="card right">
            <div className="output-top">
              <h2>Output</h2>
              <div className="output-actions">
                {hasOutput && contentSrc && (
                  <div className="segmented" role="tablist">
                    <button
                      className={viewMode === "result" ? "active" : ""}
                      onClick={() => setViewMode("result")}
                    >
                      Result
                    </button>
                    <button
                      className={viewMode === "compare" ? "active" : ""}
                      onClick={() => setViewMode("compare")}
                    >
                      Compare
                    </button>
                  </div>
                )}
                {hasOutput && (
                  <button className="btn-ghost" onClick={download}>
                    <svg
                      width="15"
                      height="15"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    >
                      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                      <polyline points="7 10 12 15 17 10" />
                      <line x1="12" y1="15" x2="12" y2="3" />
                    </svg>
                    Download
                  </button>
                )}
              </div>
            </div>

            <div className="output-area">
              {viewMode === "compare" && hasOutput ? (
                <div className="compare-grid">
                  <div className="compare-cell">
                    <span className="compare-label">Original</span>
                    <img src={contentSrc} alt="original" />
                  </div>
                  <div className="compare-cell">
                    <span className="compare-label">Stylized</span>
                    <canvas
                      ref={canvasRef}
                      className="output-canvas visible"
                    />
                  </div>
                </div>
              ) : (
                <canvas
                  ref={canvasRef}
                  className={`output-canvas${hasOutput ? " visible" : ""}`}
                />
              )}
              {!hasOutput && (
                <div className="output-placeholder">
                  {processing ? (
                    <>
                      <span className="spinner lg" />
                      <span>Running style transfer…</span>
                    </>
                  ) : (
                    <>
                      <svg
                        width="36"
                        height="36"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="1"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      >
                        <rect x="3" y="3" width="18" height="18" rx="2" />
                        <circle cx="8.5" cy="8.5" r="1.5" />
                        <path d="M21 15l-5-5L5 21" />
                      </svg>
                      <span>Upload images and click "Apply Style"</span>
                    </>
                  )}
                </div>
              )}
            </div>
            {hasOutput && wasDownscaled && (
              <p className="resize-hint">
                Source image was larger than {SOURCE_DOWNSCALE_THRESHOLD}px —
                downscaled before processing.
              </p>
            )}
          </section>
        </div>
      </main>
    </div>
  );
}
