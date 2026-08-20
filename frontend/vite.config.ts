import react from "@vitejs/plugin-react";
import { defineConfig, loadEnv } from "vite";

// /api is proxied to the backend, so the browser only ever talks to one origin
// and FastAPI needs no CORS middleware. A cross-origin error in the console
// means this block is wrong — fix it here rather than loosening the API.
export default defineConfig(({ mode }) => {
  // The backend URL is one setting for the whole repo, so it is read from the
  // root .env — ".." of this directory, since vite is always started from
  // frontend/ — rather than duplicated here. The empty prefix is what lets a
  // plain BACKEND_API (no VITE_ in front) be seen; it is read by the dev
  // server and never reaches the browser bundle.
  const env = loadEnv(mode, "..", "");

  return {
    plugins: [react()],
    server: {
      proxy: {
        "/api": {
          target: env.BACKEND_API || "http://localhost:8000",
          changeOrigin: true,
          // The backend serves /chat, not /api/chat: the prefix exists only to
          // tell the dev server what to forward.
          rewrite: (path) => path.replace(/^\/api/, ""),
        },
      },
    },
  };
});
