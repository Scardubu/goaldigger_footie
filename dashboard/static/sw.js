/**
 * GoalDiggers Service Worker
 * Provides offline functionality, caching, and push notifications
 * for the AI-powered football intelligence platform
 */

const CACHE_NAME = 'goaldiggers-v4.1.0';
const STATIC_CACHE = 'goaldiggers-static-v4.1.0';
const DYNAMIC_CACHE = 'goaldiggers-dynamic-v4.1.0';

// Static assets to cache immediately
const STATIC_ASSETS = [
  '/',
  '/static/manifest.json',
  '/static/style.css',
  '/static/icons/icon-192x192.png',
  '/static/icons/icon-512x512.png',
  '/GoalDiggers_logo.png',
  '/GoalDiggers_favicon.ico'
];

// API endpoints to cache with stale-while-revalidate strategy
const API_ENDPOINTS = [
  '/api/v4.1/predict',
  '/api/v4.1/teams',
  '/api/v4.1/leagues',
  '/api/v4.1/cross-league',
  '/api/v4.1/analytics'
];

// Install event - cache static assets
self.addEventListener('install', event => {
  console.log('🚀 GoalDiggers Service Worker installing...');
  
  event.waitUntil(
    caches.open(STATIC_CACHE)
      .then(cache => {
        console.log('📦 Caching static assets...');
        return cache.addAll(STATIC_ASSETS);
      })
      .then(() => {
        console.log('✅ Static assets cached successfully');
        return self.skipWaiting();
      })
      .catch(error => {
        console.error('❌ Failed to cache static assets:', error);
      })
  );
});

// Activate event - clean up old caches
self.addEventListener('activate', event => {
  console.log('🔄 GoalDiggers Service Worker activating...');
  
  event.waitUntil(
    caches.keys()
      .then(cacheNames => {
        return Promise.all(
          cacheNames.map(cacheName => {
            if (cacheName !== STATIC_CACHE && 
                cacheName !== DYNAMIC_CACHE && 
                cacheName !== CACHE_NAME) {
              console.log('🗑️ Deleting old cache:', cacheName);
              return caches.delete(cacheName);
            }
          })
        );
      })
      .then(() => {
        console.log('✅ Service Worker activated');
        return self.clients.claim();
      })
  );
});

// Fetch event - implement caching strategies
self.addEventListener('fetch', event => {
  const { request } = event;
  const url = new URL(request.url);
  
  // Handle API requests with stale-while-revalidate
  if (url.pathname.startsWith('/api/')) {
    event.respondWith(staleWhileRevalidate(request));
    return;
  }
  
  // Handle static assets with cache-first
  if (STATIC_ASSETS.some(asset => url.pathname.endsWith(asset))) {
    event.respondWith(cacheFirst(request));
    return;
  }
  
  // Handle navigation requests with network-first
  if (request.mode === 'navigate') {
    event.respondWith(networkFirst(request));
    return;
  }
  
  // Default to network-first for other requests
  event.respondWith(networkFirst(request));
});

// Cache-first strategy for static assets
async function cacheFirst(request) {
  try {
    const cachedResponse = await caches.match(request);
    if (cachedResponse) {
      return cachedResponse;
    }
    
    const networkResponse = await fetch(request);
    const cache = await caches.open(STATIC_CACHE);
    cache.put(request, networkResponse.clone());
    
    return networkResponse;
  } catch (error) {
    console.error('Cache-first strategy failed:', error);
    return new Response('Offline - Asset not available', { status: 503 });
  }
}

// Network-first strategy for navigation and dynamic content
async function networkFirst(request) {
  try {
    const networkResponse = await fetch(request);
    
    // Cache successful responses
    if (networkResponse.ok) {
      const cache = await caches.open(DYNAMIC_CACHE);
      cache.put(request, networkResponse.clone());
    }
    
    return networkResponse;
  } catch (error) {
    console.log('Network failed, trying cache:', error);
    
    const cachedResponse = await caches.match(request);
    if (cachedResponse) {
      return cachedResponse;
    }
    
    // Return offline page for navigation requests
    if (request.mode === 'navigate') {
      return new Response(`
        <!DOCTYPE html>
        <html>
        <head>
          <title>GoalDiggers - Offline</title>
          <meta name="viewport" content="width=device-width, initial-scale=1">
          <style>
            body { font-family: Arial, sans-serif; text-align: center; padding: 50px; }
            .offline { color: #666; }
            .logo { width: 100px; height: 100px; margin: 20px auto; }
          </style>
        </head>
        <body>
          <div class="logo">⚽</div>
          <h1>GoalDiggers</h1>
          <p class="offline">You're currently offline. Please check your connection and try again.</p>
          <button onclick="window.location.reload()">Retry</button>
        </body>
        </html>
      `, {
        headers: { 'Content-Type': 'text/html' },
        status: 503
      });
    }
    
    return new Response('Offline', { status: 503 });
  }
}

// Stale-while-revalidate strategy for API requests
async function staleWhileRevalidate(request) {
  const cache = await caches.open(DYNAMIC_CACHE);
  const cachedResponse = await cache.match(request);
  
  // Fetch fresh data in background
  const fetchPromise = fetch(request).then(networkResponse => {
    if (networkResponse.ok) {
      cache.put(request, networkResponse.clone());
    }
    return networkResponse;
  });
  
  // Return cached version immediately if available
  if (cachedResponse) {
    return cachedResponse;
  }
  
  // Otherwise wait for network
  return fetchPromise;
}

// Push notification event
self.addEventListener('push', event => {
  console.log('📱 Push notification received');
  
  const options = {
    body: event.data ? event.data.text() : 'New football insights available!',
    icon: '/static/icons/icon-192x192.png',
    badge: '/static/icons/badge-72x72.png',
    vibrate: [200, 100, 200],
    data: {
      url: '/',
      timestamp: Date.now()
    },
    actions: [
      {
        action: 'view',
        title: 'View Insights',
        icon: '/static/icons/view-24x24.png'
      },
      {
        action: 'dismiss',
        title: 'Dismiss',
        icon: '/static/icons/dismiss-24x24.png'
      }
    ],
    requireInteraction: true,
    tag: 'goaldiggers-notification'
  };
  
  event.waitUntil(
    self.registration.showNotification('GoalDiggers', options)
  );
});

// Notification click event
self.addEventListener('notificationclick', event => {
  console.log('🔔 Notification clicked');
  
  event.notification.close();
  
  if (event.action === 'view') {
    event.waitUntil(
      clients.openWindow(event.notification.data.url || '/')
    );
  }
});

// Background sync event
self.addEventListener('sync', event => {
  console.log('🔄 Background sync triggered');
  
  if (event.tag === 'goaldiggers-sync') {
    event.waitUntil(syncData());
  }
});

// Sync data when back online
async function syncData() {
  try {
    // Sync any pending predictions or user data
    console.log('📊 Syncing data...');
    
    // Implementation would sync with backend
    // This is a placeholder for actual sync logic
    
    console.log('✅ Data sync completed');
  } catch (error) {
    console.error('❌ Data sync failed:', error);
  }
}

console.log('🎯 GoalDiggers Service Worker loaded successfully');
