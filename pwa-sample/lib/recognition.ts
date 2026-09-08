import * as tf from "@tensorflow/tfjs";
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";
export type Meta={classes:string[];mean:number[];std:number[]};
export function normalize(points:{x:number;y:number;z:number}[],meta:Meta){
 if(points.length!==21)throw new Error("Expected 21 landmarks");
 return points.map(p=>[p.x,p.y,p.z].map((v,i)=>(v-meta.mean[i])/meta.std[i]));
}
let pending:Promise<{model:tf.LayersModel;hand:HandLandmarker;meta:Meta}>|null=null;
export function loadRuntime(){
 if(!pending)pending=(async()=>{
  await tf.ready();
  const [model,response,vision]=await Promise.all([
   tf.loadLayersModel("/models/alphabet/model.json"),
   fetch("/models/alphabet/metadata.json"),
   FilesetResolver.forVisionTasks("/mediapipe/wasm")
  ]);
  if(!response.ok)throw new Error("Could not load alphabet labels.");
  const meta=await response.json() as Meta;
  if(!Array.isArray(meta.classes)||meta.mean?.length!==3||meta.std?.length!==3||!meta.std.every(v=>Number.isFinite(v)&&v>0))throw new Error("Invalid model metadata.");
  if(model.inputs[0].shape.slice(1).join(",")!=="21,3"||model.outputs[0].shape[1]!==meta.classes.length)
   throw new Error("The model and label list do not match.");
  const hand=await HandLandmarker.createFromOptions(vision,{
   baseOptions:{modelAssetPath:"/mediapipe/hand_landmarker.task"},
   runningMode:"VIDEO",numHands:2,minHandDetectionConfidence:.5,minTrackingConfidence:.5
  });
  return {model,hand,meta};
 })().catch(e=>{pending=null;throw e});
 return pending!;
}
export function predict(model:tf.LayersModel,points:{x:number;y:number;z:number}[],meta:Meta){
 return tf.tidy(()=>{
  const input=tf.tensor3d([normalize(points,meta)]);
  const output=model.predict(input) as tf.Tensor;
  return Array.from(output.dataSync()).map((score,i)=>({label:meta.classes[i],score}))
   .sort((a,b)=>b.score-a.score).slice(0,3);
 });
}
