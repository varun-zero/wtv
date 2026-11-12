import { defineConfig } from 'vite';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
  plugins: [tailwindcss()],
  server: {
    port: 3000,
    host: '0.0.0.0',
    proxy: {
      '/api/opensky': {
        target: 'https://opensky-network.org',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/opensky/, '/api'),
        headers: {
          'User-Agent': 'SkyPrint-Workbench/1.0',
        },
      },
      '/api/adsb-lol': {
        target: 'https://api.adsb.lol',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/adsb-lol/, ''),
        headers: {
          'User-Agent': 'SkyPrint-Workbench/1.0',
        },
      },
    },
  },
  preview: {
    port: 3000,
    host: '0.0.0.0',
  },
});
