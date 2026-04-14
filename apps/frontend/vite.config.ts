import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/model-proxy": {
        target: "https://reiinakano.github.io",
        changeOrigin: true,
        followRedirects: true,
        rewrite: (path) => path.replace(/^\/model-proxy/, ""),
      },
    },
  },
});
