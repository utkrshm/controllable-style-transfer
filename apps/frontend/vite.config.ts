import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(() => {
  const repoName = process.env.GITHUB_REPOSITORY?.split("/")[1];
  const base =
    process.env.VITE_BASE_PATH ??
    (process.env.GITHUB_ACTIONS === "true" && repoName ? `/${repoName}/` : "/");

  return {
    base,
    plugins: [react()],
    server: {
      proxy: {
        "/api": {
          target: "http://localhost:3000",
          changeOrigin: true,
        },
      },
    },
  };
});
