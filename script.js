import{FilesetResolver,HandLandmarker}from"https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/+esm";

const v=document.querySelector("#video"),c=document.querySelector("#canvas"),x=c.getContext("2d");
const status=document.querySelector("#status"),start=document.querySelector("#start"),msg=document.querySelector("#msg"),matrix=document.querySelector("#matrix");
const gestureOut=document.querySelector("#gesture"),countOut=document.querySelector("#count"),rmseOut=document.querySelector("#rmse");
const before=document.querySelector("#before"),after=document.querySelector("#after"),result=document.querySelector("#result");
const sampleStatus=document.querySelector("#sampleStatus"),validationResult=document.querySelector("#validationResult"),auditTable=document.querySelector("#auditTable");
const rawAccuracy=document.querySelector("#rawAccuracy"),compactAccuracy=document.querySelector("#compactAccuracy");
const rawDetail=document.querySelector("#rawDetail"),compactDetail=document.querySelector("#compactDetail");
const trainingCount=document.querySelector("#trainingCount"),testingCount=document.querySelector("#testingCount"),sampleTable=document.querySelector("#sampleTable");
let lm,lastData=null,samples=[];
const model="https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const labels=["HELLO","YES","STOP"];

function pc(a){return[5,9,13,17].reduce((r,i)=>({x:r.x+a[i].x/4,y:r.y+a[i].y/4,z:r.z+a[i].z/4}),{x:0,y:0,z:0})}
function reps(a){return[0,4,8,12,16,20].map(i=>a[i]).concat(pc(a))}
function norm(a){let w=a[0],q=a.map(p=>({x:p.x-w.x,y:p.y-w.y,z:p.z-w.z})),s=Math.max(...q.map(p=>Math.hypot(p.x,p.y,p.z)))||1;return q.map(p=>({x:p.x/s,y:p.y/s,z:p.z/s}))}
function q(v){return Math.max(0,Math.min(127,Math.round((v+1)*63.5)))}
function dq(v){return v/63.5-1}
function bits(m){return m.flat().map(v=>v.toString(2).padStart(7,"0")).join("")}
function rev(s){let b=s.split("").map(Number);for(let i=0;i+2<b.length;i+=3)b[i+2]^=b[i]&b[i+1];return b.join("")}

function draw(a){
  c.width=v.videoWidth;c.height=v.videoHeight;x.clearRect(0,0,c.width,c.height);
  x.strokeStyle="#f05a00";x.fillStyle="#fff";
  [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]].forEach(([i,j])=>{x.beginPath();x.moveTo(a[i].x*c.width,a[i].y*c.height);x.lineTo(a[j].x*c.width,a[j].y*c.height);x.stroke()});
  a.forEach(p=>{x.beginPath();x.arc(p.x*c.width,p.y*c.height,4,0,7);x.fill()})
}
function clearDraw(){if(c.width&&c.height)x.clearRect(0,0,c.width,c.height)}

function fingerExtended(a,tip,pip){return a[tip].y<a[pip].y-0.025}
function gesture(a){
  const extended=[fingerExtended(a,8,6),fingerExtended(a,12,10),fingerExtended(a,16,14),fingerExtended(a,20,18)];
  const count=extended.filter(Boolean).length;
  const palmSpan=Math.hypot(a[5].x-a[17].x,a[5].y-a[17].y)||1;
  const thumbDx=Math.abs(a[4].x-a[2].x),thumbDy=a[2].y-a[4].y;
  const thumbUp=thumbDy>0.18*palmSpan && Math.abs(a[4].x-a[2].x)<0.55*palmSpan;
  const thumbClosed=thumbDx<0.38*palmSpan && Math.abs(a[4].y-a[2].y)<0.65*palmSpan;
  if(count>=3)return"HELLO";
  if(count===0&&thumbClosed)return"STOP";
  if(thumbUp&&count<=1)return"YES";
  return"TRACKING";
}

function vector21(a){return norm(a).flatMap(p=>[p.x,p.y,p.z])}
function vector7(a){return norm(reps(a)).flatMap(p=>[q(p.x),q(p.y),q(p.z)])}
function dist(a,b){let s=0;for(let i=0;i<a.length;i++){const d=a[i]-b[i];s+=d*d}return Math.sqrt(s/a.length)}
function centroid(rows){const n=rows.length,d=rows[0].length,out=Array(d).fill(0);rows.forEach(r=>r.forEach((v,i)=>out[i]+=v));return out.map(v=>v/n)}
function predict(sample,key){
  const train=samples.filter(s=>s.split==="train");
  const groups=Object.fromEntries(labels.map(l=>[l,[]]));
  train.forEach(s=>groups[s.label].push(s[key]));
  const usable=labels.filter(l=>groups[l].length);
  if(usable.length<2)return null;
  const cents=Object.fromEntries(usable.map(l=>[l,centroid(groups[l])]));
  return usable.reduce((best,l)=>{const d=dist(sample[key],cents[l]);return !best||d<best.d?{label:l,d}:best},null).label;
}
function renderValidation(){
  const counts=Object.fromEntries(labels.map(l=>[l,{train:0,test:0}]));
  samples.forEach(s=>counts[s.label][s.split]++);
  sampleTable.innerHTML=labels.map(l=>"<tr><td>"+l+"</td><td>"+counts[l].train+"</td><td>"+counts[l].test+"</td></tr>").join("");
  const trainCount=samples.filter(s=>s.split==="train").length,testRows=samples.filter(s=>s.split==="test"),testCount=testRows.length;
  trainingCount.textContent=trainCount;testingCount.textContent=testCount;
  sampleStatus.textContent=samples.length+" samples · "+trainCount+" training · "+testCount+" held-out testing";
  if(testCount===0){rawAccuracy.textContent=compactAccuracy.textContent="—";rawDetail.textContent=compactDetail.textContent="0 / 0 test samples";auditTable.innerHTML='<tr><td colspan="6">No held-out samples yet.</td></tr>';validationResult.textContent="Capture more samples. First 6 per gesture train; later samples test.";return}
  const rows=testRows.map((s,i)=>({s,i,raw:predict(s,"raw"),compact:predict(s,"compact")}));
  const rawHits=rows.filter(r=>r.raw===r.s.label).length;
  const compactHits=rows.filter(r=>r.compact===r.s.label).length;
  rawAccuracy.textContent=Math.round(rawHits/testCount*100)+"%";
  compactAccuracy.textContent=Math.round(compactHits/testCount*100)+"%";
  rawDetail.textContent=rawHits+" / "+testCount+" test samples";
  compactDetail.textContent=compactHits+" / "+testCount+" test samples";
  auditTable.innerHTML=rows.map(r=>{const rawOk=r.raw===r.s.label,compactOk=r.compact===r.s.label;return "<tr><td>"+(r.i+1)+"</td><td>"+(r.s.image?'<img class="sample-thumb" src="'+r.s.image+'" alt="Captured '+r.s.label+' sample">':"—")+"</td><td>"+r.s.label+"</td><td>"+r.s.split.toUpperCase()+"</td><td class='"+(rawOk?"ok":"bad")+"'>"+r.raw+" "+(rawOk?"✓":"✗")+"</td><td class='"+(compactOk?"ok":"bad")+"'>"+r.compact+" "+(compactOk?"✓":"✗")+"</td></tr>"}).join("");
  validationResult.textContent="Held-out comparison: 21-landmark baseline "+rawHits+"/"+testCount+" · 147-bit compact "+compactHits+"/"+testCount+".";
}
function capture(label){
  if(!lastData){validationResult.textContent="No hand detected. Put one hand in view first.";return}
  const same=samples.filter(s=>s.label===label);
  const split=same.length<6?"train":"test";
  const snap=document.createElement("canvas");snap.width=320;snap.height=240;const sx=snap.getContext("2d");sx.drawImage(v,0,0,snap.width,snap.height);samples.push({label,split,raw:lastData.raw.slice(),compact:lastData.compact.slice(),image:snap.toDataURL("image/jpeg",0.65)});
  renderValidation();
}
document.querySelectorAll(".capture").forEach(b=>b.addEventListener("click",()=>capture(b.dataset.label)));
document.querySelector("#clearSamples").addEventListener("click",()=>{samples=[];renderValidation()});
document.querySelector("#exportSamples").addEventListener("click",()=>{
  if(!samples.length){validationResult.textContent="No samples to export.";return}
  const testRows=samples.filter(s=>s.split==="test");
  const rows=testRows.map((s,i)=>({index:i+1,label:s.label,split:s.split,rawPrediction:predict(s,"raw"),compactPrediction:predict(s,"compact"),raw:s.raw,compact:s.compact,image:s.image||null}));
  const payload={exportedAt:new Date().toISOString(),method:"First 6 samples per gesture train a nearest-centroid classifier; later samples are held-out tests.",samples:rows};
  const blob=new Blob([JSON.stringify(payload,null,2)],{type:"application/json"});
  const url=URL.createObjectURL(blob),a=document.createElement("a");a.href=url;a.download="signalink-mvp-2a-validation.json";a.click();URL.revokeObjectURL(url);
  validationResult.textContent="Dataset exported: "+rows.length+" held-out samples with snapshots, vectors, labels, and predictions.";
});

function process(r){
  if(!r.landmarks.length){
    clearDraw();lastData=null;countOut.textContent="0 / 21";gestureOut.textContent="—";rmseOut.textContent="—";
    msg.style.display="grid";msg.textContent="No hand detected.";return;
  }
  msg.style.display="none";const a=r.landmarks[0];draw(a);countOut.textContent="21 / 21";
  gestureOut.textContent=gesture(a);
  const n=norm(reps(a)),m=[n.map(p=>q(p.x)),n.map(p=>q(p.y)),n.map(p=>q(p.z))];
  matrix.innerHTML='<div class="matrix-head"><span></span><span>W</span><span>TT</span><span>IT</span><span>MT</span><span>RT</span><span>PT</span><span>PC</span></div>'+m.map((row,i)=>'<div class="row"><b>'+["X","Y","Z"][i]+"</b>"+row.map(val=>"<span>"+val+"</span>").join("")+"</div>").join("");
  const actual=n.flatMap(p=>[p.x,p.y,p.z]),rec=m.flat().map(dq);
  const rm=Math.sqrt(actual.reduce((s,val,i)=>s+(val-rec[i])**2,0)/actual.length);
  const b=bits(m),z=rev(b);
  rmseOut.textContent=rm.toFixed(4);before.textContent=b.slice(0,84)+"…";after.textContent=z.slice(0,84)+"…";
  result.textContent="Reversible transform applied · token length = "+b.length+" bits";
  lastData={raw:vector21(a),compact:vector7(a)};
}
async function run(){
  status.textContent="LOADING";
  const f=await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm");
  lm=await HandLandmarker.createFromOptions(f,{baseOptions:{modelAssetPath:model},runningMode:"VIDEO",numHands:1});
  const s=await navigator.mediaDevices.getUserMedia({video:true,audio:false});v.srcObject=s;await v.play();
  msg.style.display="grid";msg.textContent="No hand detected.";status.textContent="LIVE";start.textContent="CAMERA RUNNING";
  (function loop(){const r=lm.detectForVideo(v,performance.now());process(r);requestAnimationFrame(loop)})()
}
start.onclick=()=>run().catch(e=>{console.error(e);status.textContent="START FAILED";msg.style.display="grid";msg.textContent="Use HTTPS/localhost and allow camera access."});
renderValidation();