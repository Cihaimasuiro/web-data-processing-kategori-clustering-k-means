import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  base: '/Aplikasi-web-analisis-penjualan-dengan-K-Means/',
  plugins: [react()],
})
