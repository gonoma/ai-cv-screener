import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// /api is proxied to the backend, so the browser only ever talks to one origin
// and FastAPI needs no CORS middleware. A cross-origin error in the console
// means this block is wrong — fix it here rather than loosening the API.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
        // The backend serves /chat, not /api/chat: the prefix exists only to
        // tell the dev server what to forward.
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
