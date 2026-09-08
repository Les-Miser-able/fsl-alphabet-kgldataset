import io,json,zipfile,hashlib
from pathlib import Path
import h5py
import numpy as np
root=Path("C:/Users/Russel/Desktop/fsl-extract")
out=Path("C:/Users/Russel/Documents/Codex/2026-09-06/use/work/hudyat-pwa")
target=out/"public/models/alphabet"
target.mkdir(parents=True,exist_ok=True)
z=zipfile.ZipFile(root/"models/fsl_model.keras")
h=h5py.File(io.BytesIO(z.read("model.weights.h5")),"r")
names=["conv1d","batch_normalization","conv1d_1","batch_normalization_1","lstm","dense","dense_1"]
weights={}
for name in names:
 group=h["layers/"+name+("/cell" if name=="lstm" else "")+"/vars"]
 weights[name]=[np.array(group[k],dtype=np.float32) for k in sorted(group,key=int)]
flat=[];spec=[]
for name,arrays in weights.items():
 for i,array in enumerate(arrays):
  spec.append(dict(layer=name,index=i,shape=list(array.shape),size=int(array.size)))
  flat.append(array.reshape(-1))
(target/"raw-weights.bin").write_bytes(np.concatenate(flat).astype("<f4").tobytes())
(target/"raw-spec.json").write_text(json.dumps(spec))
mean=np.load(root/"models/norm_mean.npy").reshape(3)
std=np.load(root/"models/norm_std.npy").reshape(3)
labels=np.load(root/"models/label_classes.npy").tolist()
meta=dict(category="alphabet",classes=labels,input_shape=[21,3],mean=mean.tolist(),std=std.tolist(),
 model_type="single-frame-one-hand",source_sha256=hashlib.sha256((root/"models/fsl_model.keras").read_bytes()).hexdigest(),
 limitation="This model classifies single-frame hand poses, including J/Z poses; it does not recognize their motion.")
(target/"metadata.json").write_text(json.dumps(meta,indent=2))
def conv(x,w):
 padded=np.pad(x,((0,0),(1,1),(0,0)))
 return np.maximum(sum(padded[:,k:k+x.shape[1]]@w[0][k] for k in range(3))+w[1],0)
def bn(x,w):
 return (x-w[2])/np.sqrt(w[3]+.001)*w[0]+w[1]
def sigmoid(x): return 1/(1+np.exp(-np.clip(x,-80,80)))
def predict(x):
 x=bn(conv(x,weights["conv1d"]),weights["batch_normalization"])
 x=bn(conv(x,weights["conv1d_1"]),weights["batch_normalization_1"])
 x=np.maximum(x[:,0:20:2],x[:,1:21:2])
 hstate=np.zeros((len(x),64),np.float32);c=hstate.copy()
 w=weights["lstm"]
 for step in range(x.shape[1]):
  gates=x[:,step]@w[0]+hstate@w[1]+w[2]
  i,f,candidate,o=np.split(gates,4,axis=-1)
  c=sigmoid(f)*c+sigmoid(i)*np.tanh(candidate)
  hstate=sigmoid(o)*np.tanh(c)
 x=np.maximum(hstate@weights["dense"][0]+weights["dense"][1],0)
 x=x@weights["dense_1"][0]+weights["dense_1"][1]
 x=np.exp(x-x.max(axis=1,keepdims=True))
 return x/x.sum(axis=1,keepdims=True)
rng=np.random.default_rng(42)
x=rng.normal(size=(4,21,3)).astype(np.float32)
original=np.load(root/"extracted_data/X.npy",mmap_mode="r")
if original.shape[1:]==(21,3):
 x=np.concatenate([x,(np.array(original[np.linspace(0,len(original)-1,8,dtype=int)])-mean)/std])
reference=dict(inputs=x.tolist(),outputs=predict(x).tolist())
(out/"tools").mkdir(exist_ok=True)
(out/"tools/reference.json").write_text(json.dumps(reference))
print("Extracted trained weights and normalization. Reference cases:",len(x))

