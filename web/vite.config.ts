import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In dev, proxy /api to the FastAPI backend so the SPA and API share an origin.
// `base: "./"` makes asset URLs relative so the same build works when served at
// a domain root (subdomain) or under a sub-path (e.g. /uvr) behind Caddy.
export default defineConfig({
  base: "./",
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: process.env.VITE_API_TARGET || "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});
