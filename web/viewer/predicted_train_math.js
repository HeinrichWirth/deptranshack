'use strict';
// Saved pair geometry only. No recorded trajectory, map or future cloud input.
const PredictedTrainMath=(()=>{
 const add=(a,b)=>a.map((v,i)=>v+b[i]),sub=(a,b)=>a.map((v,i)=>v-b[i]),mul=(a,k)=>a.map(v=>v*k);
 const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0),cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
 const norm=a=>Math.hypot(...a),unit=a=>{const n=norm(a);if(n<1e-9)throw Error('Вырожденная геометрия пути');return mul(a,1/n);};
 const lerp=(a,b,t)=>a.map((v,i)=>v+(b[i]-v)*t),valid=p=>Array.isArray(p)&&p.length===3&&p.every(Number.isFinite);
 function build(curves,sourceUp=[0,0,1]){
  const {left,right}=curves;
  if(!left||!right||left.length<2||left.length!==right.length)throw Error('В LAS нет пары предсказанных рельсов');
  const nodes=[];let distance=0;
  for(let i=0;i<left.length;i++){
   if(!valid(left[i])||!valid(right[i]))throw Error('Некорректные координаты рельсов');
   const center=mul(add(left[i],right[i]),.5),across=sub(left[i],right[i]);
   if(norm(across)<.1)throw Error('Вырожденная пара рельсов');
   if(i){const step=norm(sub(center,nodes[i-1].center));if(step<1e-8||step>5)throw Error('Разрыв или повтор станции: проезд остановлен');distance+=step;}
   nodes.push({center,across,distance});
  }
  const route={nodes,length:distance,up:unit(sourceUp),offset:[0,0,0]};
  const first=axesAt(route,0),delta=mul(nodes[0].center,-1);
  route.offset=[dot(delta,first.left),dot(delta,first.forward),dot(delta,first.up)];
  return route;
 }
 function geometry(route,d){
  d=Math.max(0,Math.min(route.length,d));let lo=0,hi=route.nodes.length-1;
  while(hi-lo>1){const m=(lo+hi)>>1;if(route.nodes[m].distance<=d)lo=m;else hi=m;}
  const a=route.nodes[lo],b=route.nodes[hi],t=(d-a.distance)/(b.distance-a.distance);
  return {center:lerp(a.center,b.center,t),across:lerp(a.across,b.across,t)};
 }
 function axesAt(route,d){
  const a=geometry(route,d-2.5),b=geometry(route,d+2.5),p=geometry(route,d);
  const forward=unit(sub(b.center,a.center));let left=unit(sub(p.across,mul(forward,dot(p.across,forward))));
  let up=unit(cross(forward,left));if(dot(up,route.up)<0){left=mul(left,-1);up=mul(up,-1);}
  return {forward,left,up};
 }
 function sample(route,d){
  if(!Number.isFinite(d))throw Error('Некорректная позиция');
  d=Math.max(0,Math.min(route.length,d));const axes=axesAt(route,d),center=geometry(route,d).center;
  let position=center.slice();[axes.left,axes.forward,axes.up].forEach((a,j)=>position=add(position,mul(a,route.offset[j])));
  return {distance:d,center,position,...axes};
 }
 function project(point,pose){const delta=sub(point,pose.position);return [dot(delta,pose.left),dot(delta,pose.forward),dot(delta,pose.up)];}
 function slice(rows,pose,width){
  if(!Number.isFinite(width)||width<=0)throw Error('Толщина должна быть положительной');
  const output=new Float32Array(rows.length/4*3);let n=0;
  for(let i=0;i<rows.length;i+=4){
   const dx=rows[i]-pose.position[0],dy=rows[i+1]-pose.position[1],dz=rows[i+2]-pose.position[2];
   const depth=dx*pose.forward[0]+dy*pose.forward[1]+dz*pose.forward[2];
   if(depth<0||depth>width)continue;
   output[n++]=dx*pose.left[0]+dy*pose.left[1]+dz*pose.left[2];
   output[n++]=dx*pose.up[0]+dy*pose.up[1]+dz*pose.up[2];output[n++]=rows[i+3];
  }
  return output.slice(0,n);
 }
 // Unwrap points along the fixed pair; synthetic rows never count as intrusions.
 function prepareInspection(rows,route){
  const nodes=route.nodes,segments=[];
  for(let k=0;k<nodes.length-1;k++){
   const a=nodes[k],b=nodes[k+1],v=sub(b.center,a.center),l2=dot(v,v);
   segments.push({a,b,v,l2,forward:unit(v),length:Math.sqrt(l2)});
  }
  const result=new Float32Array(rows.length/4*6);let used=0;
  for(let i=0;i<rows.length;i+=4){
   const x=rows[i],y=rows[i+1],z=rows[i+2];if(![x,y,z].every(Number.isFinite))continue;
   let best=Infinity,chosen=null,t=0,unclamped=0;
   for(const seg of segments){
    const dx=x-seg.a.center[0],dy=y-seg.a.center[1],dz=z-seg.a.center[2];
    const u=(dx*seg.v[0]+dy*seg.v[1]+dz*seg.v[2])/seg.l2,c=Math.max(0,Math.min(1,u));
    const d=(dx-c*seg.v[0])**2+(dy-c*seg.v[1])**2+(dz-c*seg.v[2])**2;
    if(d<best){best=d;chosen=seg;t=c;unclamped=u;}
   }
   if(!chosen||(chosen===segments[0]&&unclamped<0)||(chosen===segments.at(-1)&&unclamped>1))continue;
   const center=lerp(chosen.a.center,chosen.b.center,t),across=lerp(chosen.a.across,chosen.b.across,t);
   let left=unit(sub(across,mul(chosen.forward,dot(across,chosen.forward)))),up=unit(cross(chosen.forward,left));
   if(dot(up,route.up)<0){left=mul(left,-1);up=mul(up,-1);}
   const delta=[x-center[0],y-center[1],z-center[2]];
   result[used++]=dot(delta,left);result[used++]=dot(delta,up);result[used++]=rows[i+3];
   result[used++]=Math.hypot(x,y,z);result[used++]=chosen.a.distance+t*chosen.length;result[used++]=i/4;
  }
  return result.slice(0,used);
 }
 function penetration(x,z){return Math.min(1.05-Math.abs(x),z,3-z);}
 function intrusion(x,z,layer){return layer<20&&z>.10+1e-6&&Math.abs(x)<1.05-.03-1e-6&&z<3-.03-1e-6;}
 function inspectSlice(prepared,distance,width){
  if(!Number.isFinite(distance)||!Number.isFinite(width)||width<=0)throw Error('Некорректный срез');
  const result=new Float32Array(prepared.length);let n=0,count=0;const nearest=[];
  for(let i=0;i<prepared.length;i+=6){
   if(prepared[i+4]<distance||prepared[i+4]>distance+width)continue;
   for(let j=0;j<6;j++)result[n++]=prepared[i+j];
   if(intrusion(prepared[i],prepared[i+1],prepared[i+2])){
    count++;nearest.push({x:prepared[i],height:prepared[i+1],layer:prepared[i+2],range:prepared[i+3],station:prepared[i+4],row:prepared[i+5],penetration:penetration(prepared[i],prepared[i+1])});
   }
  }
  nearest.sort((a,b)=>a.range-b.range);
  return {points:result.slice(0,n),count,nearest:nearest.slice(0,10)};
 }
 // Screen horizontal matches the slice; forward is drawn upward by the canvas.
 function topView(point,axes){return [dot(point,axes.left),dot(point,axes.forward)];}
 return {build,sample,project,slice,prepareInspection,inspectSlice,penetration,intrusion,topView};
})();
if(typeof module!=='undefined')module.exports=PredictedTrainMath;

