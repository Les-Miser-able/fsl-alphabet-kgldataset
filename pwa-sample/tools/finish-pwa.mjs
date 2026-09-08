import fs from "node:fs/promises";import path from "node:path";import crypto from "node:crypto";
let dir;for(const candidate of ["out","dist/client","dist"]){try{await fs.access(path.join(candidate,"index.html"));dir=candidate;break}catch{}}
if(!dir)throw new Error("Static index.html not found");
async function walk(root){let files=[];for(const e of await fs.readdir(root,{withFileTypes:true})){const p=path.join(root,e.name);if(e.isDirectory())files.push(...await walk(p));else files.push(p)}return files}
const files=(await walk(dir)).filter(p=>!p.endsWith(".map")&&!p.endsWith("sw.js")&&!p.endsWith("precache.json"));
const digest=crypto.createHash("sha256");for(const p of files.sort())digest.update(await fs.readFile(p));
const version=digest.digest("hex").slice(0,14);
await fs.writeFile(path.join(dir,"precache.json"),JSON.stringify(files.map(p=>"/"+path.relative(dir,p).replaceAll("\\","/"))));
let sw=await fs.readFile("public/sw.js","utf8");await fs.writeFile(path.join(dir,"sw.js"),sw.replace("__BUILD_ID__",version));
console.log("PWA static output:",dir,"assets:",files.length,"version:",version);
