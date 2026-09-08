const CACHE="hudyat-__BUILD_ID__";
self.addEventListener("install",event=>event.waitUntil((async()=>{
 const response=await fetch("/precache.json",{cache:"no-store"});
 if(!response.ok)throw new Error("Offline asset list unavailable");
 const paths=await response.json();const cache=await caches.open(CACHE);
 await cache.addAll(paths);await self.skipWaiting();
})()));
self.addEventListener("activate",event=>event.waitUntil((async()=>{
 for(const key of await caches.keys())if(key.startsWith("hudyat-")&&key!==CACHE)await caches.delete(key);
 await self.clients.claim();
})()));
self.addEventListener("fetch",event=>{
 const url=new URL(event.request.url);
 if(event.request.method!=="GET"||url.origin!==self.location.origin)return;
 if(event.request.mode==="navigate"){
  event.respondWith(fetch(event.request).catch(async()=>await caches.match("/index.html")||Response.error()));return;
 }
 event.respondWith((async()=>{
  const cached=await caches.match(event.request);if(cached)return cached;
  return fetch(event.request);
 })());
});
