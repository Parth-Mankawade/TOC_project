import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Built files are served by FastAPI at /app/ (see backend/main.py).
// During development, `npm run dev` proxies API calls to the backend on port 8000.
export default defineConfig({
  base: "/app/",
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8000",
      "/docs": "http://127.0.0.1:8000",
      "/openapi.json": "http://127.0.0.1:8000",
    },
  },
});
