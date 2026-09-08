import fs from "node:fs/promises";
import * as tf from "@tensorflow/tfjs";
await tf.setBackend("cpu");
await tf.ready();
const base=new URL("../public/models/alphabet/",import.meta.url);
const spec=JSON.parse(await fs.readFile(new URL("raw-spec.json",base),"utf8"));
const buf=await fs.readFile(new URL("raw-weights.bin",base));
const values=new Float32Array(buf.buffer.slice(buf.byteOffset,buf.byteOffset+buf.byteLength));
const model=tf.sequential();
model.add(tf.layers.conv1d({inputShape:[21,3],filters:32,kernelSize:3,padding:"same",activation:"relu",name:"conv1d"}));
model.add(tf.layers.batchNormalization({epsilon:.001,name:"batch_normalization"}));
model.add(tf.layers.conv1d({filters:64,kernelSize:3,padding:"same",activation:"relu",name:"conv1d_1"}));
model.add(tf.layers.batchNormalization({epsilon:.001,name:"batch_normalization_1"}));
model.add(tf.layers.maxPooling1d({poolSize:2,strides:2}));
model.add(tf.layers.dropout({rate:.3}));
model.add(tf.layers.lstm({units:64,activation:"tanh",recurrentActivation:"sigmoid",returnSequences:false,name:"lstm"}));
model.add(tf.layers.dropout({rate:.3}));
model.add(tf.layers.dense({units:64,activation:"relu",name:"dense"}));
model.add(tf.layers.dropout({rate:.3}));
model.add(tf.layers.dense({units:26,activation:"softmax",name:"dense_1"}));
let offset=0;
const tensors=spec.map(s=>{const t=tf.tensor(values.slice(offset,offset+s.size),s.shape);offset+=s.size;return t;});
for (const name of new Set(spec.map(s=>s.layer))) {
 model.getLayer(name).setWeights(tensors.filter((_,i)=>spec[i].layer===name));
}
tensors.forEach(t=>t.dispose());
const ref=JSON.parse(await fs.readFile(new URL("reference.json",import.meta.url),"utf8"));
const input=tf.tensor(ref.inputs);
const pred=model.predict(input);
const output=await pred.array();
let maxError=0;
for(let i=0;i<output.length;i++) for(let j=0;j<26;j++) maxError=Math.max(maxError,Math.abs(output[i][j]-ref.outputs[i][j]));
if(maxError>0.0001) throw new Error("Conversion parity failed: "+maxError);
let saved;
await model.save(tf.io.withSaveHandler(async a=>{
 saved=a;
 await fs.writeFile(new URL("model.json",base),JSON.stringify({
 format:"layers-model",generatedBy:"HUDYAT verified Keras weight export",convertedBy:"TensorFlow.js "+tf.version.tfjs,
 modelTopology:a.modelTopology,weightsManifest:[{paths:["weights.bin"],weights:a.weightSpecs}]
 }));
 await fs.writeFile(new URL("weights.bin",base),Buffer.from(a.weightData));
 return {modelArtifactsInfo:{dateSaved:new Date(),modelTopologyType:"JSON"}};
}));
const loaded=await tf.loadLayersModel(tf.io.fromMemory(saved));
const reloaded=loaded.predict(input);
const maxRoundtrip=tf.max(tf.abs(pred.sub(reloaded))).dataSync()[0];
if(maxRoundtrip>1e-6) throw new Error("Saved model reload parity failed");
await fs.writeFile(new URL("verification.json",import.meta.url),JSON.stringify({cases:output.length,maxError,maxRoundtrip,backend:tf.getBackend()},null,2));
await fs.unlink(new URL("raw-spec.json",base));await fs.unlink(new URL("raw-weights.bin",base));
input.dispose();pred.dispose();reloaded.dispose();loaded.dispose();model.dispose();
console.log(JSON.stringify({cases:output.length,maxError,maxRoundtrip}));
