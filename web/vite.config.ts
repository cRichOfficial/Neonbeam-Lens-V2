import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  base: "/annotate/",
  server: {
    proxy: {
      "/api": "http://localhost:8100",
      "/health": "http://localhost:8100",
    },
  },
});
