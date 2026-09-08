"use client";
import {useEffect,useRef,useState,useCallback} from "react";
import {Camera,Hand,LockKeyhole,Download,Square,Volume2} from "lucide-react";
import {Button} from "@/components/ui/button";
import {Tabs,TabsList,TabsTrigger} from "@/components/ui/tabs";
type Match={label:string;score:number};
type InstallEvent=Event&{prompt:()=>Promise<void>;userChoice:Promise<{outcome:string}>};
const CATEGORIES=["alphabet","numbers","family","wh_words"];
export default function Home(){
 const [category,setCategory]=useState("alphabet"),[active,setActive]=useState(false),[loading,setLoading]=useState(false);
 const [status,setStatus]=useState("Ready when you are"),[error,setError]=useState(""),[matches,setMatches]=useState<Match[]>([]);
 const [install,setInstall]=useState<InstallEvent|null>(null),[installHelp,setInstallHelp]=useState(false),[offline,setOffline]=useState(false);
 const video=useRef<HTMLVideoElement>(null),overlay=useRef<HTMLCanvasElement>(null),stream=useRef<MediaStream|null>(null);
 const frame=useRef(0),generation=useRef(0),running=useRef(false);
 const stop=useCallback(()=>{
  generation.current++;running.current=false;cancelAnimationFrame(frame.current);
  stream.current?.getTracks().forEach(t=>t.stop());stream.current=null;
  if(video.current)video.current.srcObject=null;
  overlay.current?.getContext("2d")?.clearRect(0,0,overlay.current.width,overlay.current.height);
  setActive(false);setLoading(false);setMatches([]);setStatus("Camera paused");
 },[]);
 const selectCategory=useCallback((c:string)=>{stop();setError("");setCategory(c)},[stop]);
 useEffect(()=>{
  const online=()=>setOffline(!navigator.onLine);online();
  const prompt=(e:Event)=>{e.preventDefault();setInstall(e as InstallEvent)};
  const hidden=()=>{if(document.hidden)stop()};
  window.addEventListener("beforeinstallprompt",prompt);window.addEventListener("online",online);window.addEventListener("offline",online);
  document.addEventListener("visibilitychange",hidden);
  if("serviceWorker" in navigator && process.env.NODE_ENV === "production")navigator.serviceWorker.register("/sw.js").catch(()=>{});
  return()=>{stop();window.removeEventListener("beforeinstallprompt",prompt);window.removeEventListener("online",online);window.removeEventListener("offline",online);document.removeEventListener("visibilitychange",hidden)};
 },[stop]);
 useEffect(()=>{
  const context=(document as unknown as {modelContext?:{registerTool:(tool:unknown,options:unknown)=>Promise<void>|void}}).modelContext;
  if(!context)return;
  const lifecycle=new AbortController();
  Promise.resolve(context.registerTool({name:"select_practice_category",description:"Select a HUDYAT practice category and stop any active camera.",
   inputSchema:{type:"object",properties:{category:{type:"string",enum:CATEGORIES}},required:["category"],additionalProperties:false},
   annotations:{readOnlyHint:false},execute:(input:unknown)=>{
    const c=(input as {category?:unknown})?.category;
    if(typeof c!=="string"||!CATEGORIES.includes(c))throw new Error("Unknown category");
    selectCategory(c);return {category:c,modelAvailable:c==="alphabet",camera:"stopped"};
   }},{signal:lifecycle.signal})).catch(()=>{});
  return()=>lifecycle.abort();
 },[selectCategory]);
 async function start(){
  if(loading||active)return;setError("");setLoading(true);setStatus("Loading recognition");
  const id=++generation.current;
  try{
   if(!navigator.mediaDevices?.getUserMedia)throw new Error("Camera access requires HTTPS or localhost in a supported browser.");
   const engine=await import("@/lib/recognition");
   const runtime=await engine.loadRuntime();
   if(id!==generation.current)return;
   const media=await navigator.mediaDevices.getUserMedia({video:{facingMode:"user",width:{ideal:640},height:{ideal:480}},audio:false});
   if(id!==generation.current){media.getTracks().forEach(t=>t.stop());return}
   stream.current=media;
   const v=video.current;if(!v)throw new Error("Camera preview unavailable");
   v.srcObject=media;await v.play();
   if(id!==generation.current)return;
   setActive(true);setLoading(false);running.current=true;
   const input=document.createElement("canvas");let lastTime=-1,lastRun=0;
   const draw=(now:number)=>{
    if(!running.current||id!==generation.current)return;
    try{
     if(v.readyState>=2&&v.currentTime!==lastTime&&now-lastRun>150){
      lastTime=v.currentTime;lastRun=now;input.width=v.videoWidth;input.height=v.videoHeight;
      const ctx=input.getContext("2d")!;
      // Match the original Python camera pipeline: mirror BEFORE hand extraction.
      ctx.translate(input.width,0);ctx.scale(-1,1);ctx.drawImage(v,0,0);ctx.setTransform(1,0,0,1,0,0);
      const result=runtime.hand.detectForVideo(input,performance.now());
      const canvas=overlay.current;
      if(canvas){
       canvas.width=input.width;canvas.height=input.height;
       const paint=canvas.getContext("2d")!;paint.clearRect(0,0,canvas.width,canvas.height);
       for(const hand of result.landmarks)for(const p of hand){paint.beginPath();paint.arc(p.x*canvas.width,p.y*canvas.height,3,0,Math.PI*2);paint.fillStyle="#66e4bf";paint.fill()}
      }
      if(result.landmarks.length!==1){
       setMatches([]);setStatus(result.landmarks.length>1?"Use one hand for this model":"Bring one hand into view");
      }else{
       const ranked=engine.predict(runtime.model,result.landmarks[0],runtime.meta);
       setMatches(ranked);setStatus(ranked[0].score>=.6?"Possible letter match":"Not certain yet");
      }
     }
     frame.current=requestAnimationFrame(draw);
    }catch(e){stop();setError(e instanceof Error?e.message:"Recognition stopped. Try again.")}
   };
   frame.current=requestAnimationFrame(draw);
  }catch(e){if(id===generation.current){stop();setError(e instanceof Error?e.message:"Could not start the camera. Check permissions.")}}
 }
 const best=matches[0],confident=best&&best.score>=.6;
 async function installApp(){if(install){await install.prompt();await install.userChoice;setInstall(null)}else setInstallHelp(v=>!v)}
 function speak(){if(best&&"speechSynthesis" in window){window.speechSynthesis.cancel();window.speechSynthesis.speak(new SpeechSynthesisUtterance("Letter "+best.label))}}
 return <main className="shell">
 <header><a className="brand" href="/"><span className="brandmark"><Hand size={25}/></span>HUDYAT<small>practice studio</small></a><Button variant="outline" onClick={installApp}><Download/>Install app</Button></header>
 {installHelp&&<p className="offline-note">Use your browser’s Install app option. On iPhone or iPad, open this page in Safari, tap Share, then Add to Home Screen.</p>}
 {offline&&<p className="offline-note">You’re offline. Previously cached recognition assets can still be used.</p>}
 <section className="intro"><div><p className="eyebrow">FILIPINO SIGN LANGUAGE</p><h1>A little practice.<br/><span>A clearer connection.</span></h1></div><p>Bring a sign into view.<br/>Explore what the model sees.</p></section>
 <Tabs value={category} onValueChange={v=>selectCategory(String(v))}><TabsList className="category-tabs">{CATEGORIES.map(c=><TabsTrigger key={c} value={c}>{c==="wh_words"?"WH Words":c[0].toUpperCase()+c.slice(1)}{c!=="alphabet"&&<span className="soon">not trained</span>}</TabsTrigger>)}</TabsList></Tabs>
 {category==="alphabet"?<section className="workspace"><div className="camera-card"><div className="card-heading"><span><i className={"status-dot "+(active?"live":"")}/>{active?"Camera live":"Camera preview"}</span><span>ALPHABET / 01</span></div>
 <div className="camera-stage"><video ref={video} muted autoPlay playsInline style={{transform:"scaleX(-1)",visibility:active?"visible":"hidden"}}/><canvas ref={overlay} aria-hidden="true"/>
 {!active&&<div className="camera-empty"><Camera size={42}/><h2>Your hands tell the story</h2><p>Use one hand and keep it inside the camera view.</p><Button className="primary-button" onClick={start} disabled={loading}>{loading?"Loading recognition…":"Start camera"}</Button>{loading&&<Button variant="ghost" onClick={stop}>Cancel</Button>}</div>}
 {active&&<span className="live-help">{status}</span>}<span className="camera-corner">Frames stay on your device</span></div>
 <div className="camera-footer"><span>Good light. A clear background. One hand.</span>{active&&<Button variant="outline" onClick={stop}><Square/>Stop camera</Button>}</div>
 {error&&<p className="error" role="alert">{error}</p>}</div>
 <aside className="prediction-card"><p className="eyebrow">MODEL PREDICTION</p><div className="big-letter">{confident?best.label:"—"}</div><h2 aria-live="polite">{status}</h2>
 {!best?<p>Start your camera and hold an alphabet pose.</p>:<><div className="confidence"><span>Model score · {best.label}</span><strong>{(best.score*100).toFixed(1)}%</strong></div><div className="confidence-track"><span style={{width:best.score*100+"%"}}/></div><div className="top-list">{matches.map(m=><div key={m.label}><span>Letter {m.label}</span><span>{(m.score*100).toFixed(1)}%</span></div>)}</div><Button className="speech" variant="outline" onClick={speak}><Volume2/>Hear letter</Button></>}
 <div className="model-note"><strong>About this sample</strong><p>Uses the current single-frame alphabet model. J and Z predictions describe a pose, not the complete movement. Scores are estimates, not confirmation of a correct sign.</p></div></aside></section>:<section className="unavailable"><Hand size={36}/><h2>This category is waiting for its model</h2><p>The sample currently has an alphabet model. Train and export this category before enabling recognition.</p><Button onClick={()=>selectCategory("alphabet")}>Practice the alphabet</Button></section>}
 <footer><span>HUDYAT · Alphabet model sample</span><span className="private-note"><LockKeyhole size={14}/> Camera frames are never uploaded</span></footer></main>
}
