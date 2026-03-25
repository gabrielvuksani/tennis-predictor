
// Service worker — cache static assets for faster loads
const CACHE='tp-v2';
const ASSETS=['./','assets/css/style.css','assets/js/app.js'];
self.addEventListener('install',e=>{e.waitUntil(caches.open(CACHE).then(c=>c.addAll(ASSETS)));self.skipWaiting()});
self.addEventListener('activate',e=>{e.waitUntil(caches.keys().then(ks=>Promise.all(ks.filter(k=>k!==CACHE).map(k=>caches.delete(k)))));self.clients.claim()});
self.addEventListener('fetch',e=>{
  if(e.request.url.includes('predictions.json')){e.respondWith(fetch(e.request));return}
  e.respondWith(caches.match(e.request).then(r=>r||fetch(e.request)));
});
