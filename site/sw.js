
// Service worker for Tennis Predictor notifications
self.addEventListener('install', e => self.skipWaiting());
self.addEventListener('activate', e => e.waitUntil(self.clients.claim()));
self.addEventListener('push', e => {
  const data = e.data ? e.data.json() : {title: 'Tennis Predictor', body: 'New predictions available!'};
  e.waitUntil(self.registration.showNotification(data.title, {body: data.body, icon: '/tennis-predictor/favicon.ico'}));
});
