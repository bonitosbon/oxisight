import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
import { VitePWA } from 'vite-plugin-pwa';
export default defineConfig({plugins:[react(),tailwindcss(),VitePWA({registerType:'autoUpdate',manifest:{name:'OXI SIGHT — RustScan AI',short_name:'OXI SIGHT',description:'Camera-based surface inspection',theme_color:'#101015',background_color:'#101015',display:'standalone',start_url:'/',icons:[{src:'/icon-192.png',sizes:'192x192',type:'image/png'},{src:'/icon-512.png',sizes:'512x512',type:'image/png',purpose:'any maskable'}]},workbox:{globPatterns:['**/*.{js,css,html,svg,png}'],navigateFallbackDenylist:[/^\/api\//]}})],server:{proxy:{'/api':'http://127.0.0.1:8000'}}});
