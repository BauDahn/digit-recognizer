import { resolve } from 'node:path';
import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

// Two pages: the handwriting app for people (index.html) and the lab for looking inside the networks (lab.html)
// BASE_PATH lets GitHub Pages serve the site from /<repo>/; locally it stays at the root
export default defineConfig({
  base: process.env.BASE_PATH ?? '/',
  plugins: [react(), tailwindcss()],
  build: {
    rollupOptions: {
      input: {
        app: resolve(import.meta.dirname, 'index.html'),
        lab: resolve(import.meta.dirname, 'lab.html'),
      },
    },
  },
});
