import { useEffect, useMemo, useState } from "react";

type StyleControls = {
  color_transfer: number;
  texture_transfer: number;
  brushstroke_transfer: number;
};

type ParseIntentResponse = {
  elements: StyleControls;
  explanation: string[];
  inferredPreset: "balanced" | "painterly" | "color-pop";
};

const DEFAULT_CONTROLS: StyleControls = {
  color_transfer: 1.0,
  texture_transfer: 1.0,
  brushstroke_transfer: 1.0,
};

const CONTROL_LABELS: Record<keyof StyleControls, string> = {
  color_transfer: "Color Transfer",
  texture_transfer: "Texture Transfer",
  brushstroke_transfer: "Brushstroke Transfer",
};

const API_BASE = (import.meta.env.VITE_BACKEND_URL as string | undefined)?.replace(/\/+$/, "") ?? "";

function apiUrl(path: string): string {
  return API_BASE ? `${API_BASE}${path}` : path;
}

function App() {
  const [contentImage, setContentImage] = useState<File | null>(null);
  const [styleImage, setStyleImage] = useState<File | null>(null);
  const [contentPreview, setContentPreview] = useState<string>("");
  const [stylePreview, setStylePreview] = useState<string>("");
  const [prompt, setPrompt] = useState("Transfer the entire style");
  const [controls, setControls] = useState<StyleControls>(DEFAULT_CONTROLS);
  const [previewUrl, setPreviewUrl] = useState<string>("");
  const [error, setError] = useState<string>("");
  const [loading, setLoading] = useState<false | "creating">(false);

  const readyToGenerate = useMemo(() => Boolean(contentImage && styleImage), [contentImage, styleImage]);

  useEffect(() => {
    if (!contentImage) {
      setContentPreview("");
      return;
    }
    const url = URL.createObjectURL(contentImage);
    setContentPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [contentImage]);

  useEffect(() => {
    if (!styleImage) {
      setStylePreview("");
      return;
    }
    const url = URL.createObjectURL(styleImage);
    setStylePreview(url);
    return () => URL.revokeObjectURL(url);
  }, [styleImage]);

  async function createStyleTransfer() {
    if (!contentImage || !styleImage) {
      setError("Both content and style images are required.");
      return;
    }

    setError("");
    setLoading("creating");
    try {
      let resolvedControls = controls;
      if (prompt.trim().length > 0) {
        const parseRes = await fetch(apiUrl("/api/parse-intent"), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ prompt }),
        });
        if (parseRes.ok) {
          const parsedData = (await parseRes.json()) as ParseIntentResponse;
          resolvedControls = parsedData.elements;
          setControls(parsedData.elements);
        }
      }

      const form = new FormData();
      form.append("contentImage", contentImage);
      form.append("styleImage", styleImage);
      form.append("prompt", prompt);
      form.append("qualityPreset", "preview");
      form.append("controls", JSON.stringify(resolvedControls));

      const res = await fetch(apiUrl("/api/style-transfer"), {
        method: "POST",
        body: form,
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`Generation failed (${res.status}): ${text}`);
      }
      const data = (await res.json()) as { mimeType: string; imageBase64: string };
      setPreviewUrl(`data:${data.mimeType};base64,${data.imageBase64}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to generate preview");
    } finally {
      setLoading(false);
    }
  }

  function updateControl(key: keyof StyleControls, value: number) {
    setControls((prev) => ({ ...prev, [key]: value }));
  }

  return (
    <main className="page">
      <section className="shell">
        <header className="hero">
          <div>
            <h1>Controllable Style Transfer App</h1>
            <p className="subtitle">Upload two images, then steer style transfer with simple controls.</p>
          </div>
        </header>

        <div className="workspace">
          <section className="controlPane">
            <section className="inputBoard">
              <div className="sectionHead">Images</div>
              <div className="uploadGrid">
                <label htmlFor="content-file" className="uploadCard">
                  <div className="uploadMeta">
                    <span className="uploadTitle">Content</span>
                    <span className="uploadHint">{contentImage ? "Replace" : "Upload"}</span>
                  </div>
                  <div className="uploadViewport">
                    {contentPreview ? (
                      <img src={contentPreview} alt="Content preview" className="uploadPreview" />
                    ) : (
                      <p className="emptyUpload">Add content image</p>
                    )}
                  </div>
                </label>

                <label htmlFor="style-file" className="uploadCard">
                  <div className="uploadMeta">
                    <span className="uploadTitle">Style</span>
                    <span className="uploadHint">{styleImage ? "Replace" : "Upload"}</span>
                  </div>
                  <div className="uploadViewport">
                    {stylePreview ? (
                      <img src={stylePreview} alt="Style preview" className="uploadPreview" />
                    ) : (
                      <p className="emptyUpload">Add style image</p>
                    )}
                  </div>
                </label>
              </div>

              <div className="controlsBlock">
                <div className="sectionHead">Controls</div>
                <label htmlFor="prompt-input" className="intentLabel">
                  Creative Intent
                </label>
                <textarea
                  id="prompt-input"
                  placeholder="Transfer the entire style"
                  rows={2}
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                />
                <div className="promptCaptionWrap">
                  <button type="button" className="promptCaption" onClick={() => setPrompt("")}>
                    Remove this text if you want to set controls manually.
                  </button>
                </div>

                <div className="sliders">
                  {(Object.keys(controls) as Array<keyof StyleControls>).map((key) => (
                    <div key={key} className="sliderRow">
                      <span>{CONTROL_LABELS[key]}</span>
                      <input
                        type="range"
                        min={0}
                        max={1}
                        step={0.01}
                        value={controls[key]}
                        onChange={(e) => updateControl(key, Number(e.target.value))}
                      />
                      <strong>{controls[key].toFixed(2)}</strong>
                    </div>
                  ))}
                </div>
              </div>

              <div className="createRow">
                <button className="primary" type="button" onClick={createStyleTransfer} disabled={!readyToGenerate || loading !== false}>
                  {loading === "creating" ? "Creating..." : "Create"}
                </button>
                {error && <p className="error">{error}</p>}
              </div>
            </section>

            <input
              id="content-file"
              className="hiddenInput"
              type="file"
              accept="image/*"
              onChange={(e) => setContentImage(e.target.files?.[0] ?? null)}
            />
            <input
              id="style-file"
              className="hiddenInput"
              type="file"
              accept="image/*"
              onChange={(e) => setStyleImage(e.target.files?.[0] ?? null)}
            />
          </section>

          <section className="outputCard">
            <div className="outputHeader">
              <h2>Output</h2>
            </div>
            {previewUrl ? (
              <img src={previewUrl} alt="Generated preview" className="previewImage" />
            ) : (
              <p className="emptyPreview">Your styled output appears here.</p>
            )}
            {previewUrl ? (
              <a className="primary downloadButton" href={previewUrl} download="style-transfer-output.png">
                Download
              </a>
            ) : (
              <span className="primary downloadButton disabled">Download</span>
            )}
          </section>
        </div>
      </section>
    </main>
  );
}

export default App;
