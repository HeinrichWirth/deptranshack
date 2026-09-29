'use strict';
importScripts('/predicted_train_math.js');
let rows=null,token=0,prepared=null;
function planeInspection(rows,plane){
 const p=new Float32Array(rows.length/4*6);let n=0;
 const dot=(x,y,z,a)=>x*a[0]+y*a[1]+z*a[2];
 for(let i=0;i<rows.length;i+=4){const x=rows[i],y=rows[i+1],z=rows[i+2],u=dot(x,y,z,plane.forward);if(u<0||u>8)continue;
  p[n++]=-dot(x,y,z,plane.right);p[n++]=dot(x,y,z,plane.up)-plane.floor;p[n++]=rows[i+3];p[n++]=Math.hypot(x,y,z);p[n++]=u;p[n++]=i/4;
 }return p.slice(0,n);
}
onmessage=event=>{
 const m=event.data;
 try{
  if(m.type==='load'){rows=new Float32Array(m.buffer);token=m.token;const start=performance.now();prepared=m.inspection_plane?planeInspection(rows,m.inspection_plane):PredictedTrainMath.prepareInspection(rows,m.route);postMessage({type:'ready',token,ms:performance.now()-start});return;}
  if(m.type==='slice'&&rows&&m.token===token){
   const start=performance.now(),result=PredictedTrainMath.inspectSlice(prepared,m.pose.distance,m.width),points=result.points;
   postMessage({type:'slice',token,request:m.request,distance:m.pose.distance,width:m.width,ms:performance.now()-start,count:result.count,nearest:result.nearest,buffer:points.buffer},[points.buffer]);
  }
 }catch(error){postMessage({type:'error',token,error:error.message});}
};
