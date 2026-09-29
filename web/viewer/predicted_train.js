'use strict';
// Bind every result request to the selected run; no global "current bag" state.
const resultJob=new URLSearchParams(location.search).get('job');
const originalFetch=window.fetch.bind(window);
window.fetch=(resource,options)=>{
 const url=new URL(resource,location.href);
 if(resultJob&&url.origin===location.origin&&url.pathname.startsWith('/api/'))url.searchParams.set('job',resultJob);
 return originalFetch(url,options);
};
(()=>{
 const $=id=>document.getElementById(id),M=PredictedTrainMath;
 const colors={4:'#ff58c7',0:'#7d8b9d',1:'#48ca8b',2:'#f02db4',20:'#ffd22d',21:'#00c8ff',22:'#ff7d23',23:'#9b82dc',30:'#ffffff',31:'#ffffff',32:'#ffffff'};
 let runs=[],run=null,index=0,meta=null,route=null,worker=null,abort=null,token=0;
 let distance=0,playing=false,lastTime=0,animation=0,busy=false,ready=false,request=0,pending=false,ended=false;
 let slicePoints=null,sliceDistance=0,sliceWidth=0,topPoints=[],topTransform=null,topAxes=null,pan=[0,0],drag=null;
 let selectedPoint=null;
 const initialParams=new URLSearchParams(location.search);let initialSource=true,pendingDistance=Number(initialParams.get('distance')||0);if(!Number.isFinite(pendingDistance))pendingDistance=0;
 if(Number(initialParams.get('width'))>0)$('width').value=Number(initialParams.get('width'));
 const active=()=>{const s=new Set([...document.querySelectorAll('[data-layer]:checked')].map(e=>+e.dataset.layer));if(s.has(30)){s.add(31);s.add(32);}return s;};
 const status=(text,error=false)=>{$('status').textContent=text;$('status').classList.toggle('error',error);};
 const number=(id,fallback)=>{const x=Number($(id).value);return Number.isFinite(x)&&x>0?x:fallback;};
 function pause(){playing=false;cancelAnimationFrame(animation);animation=0;$('play').textContent='▶ Ехать';}
 function enable(ok){for(const id of ['play','back','forward','start','positionSeek'])$(id).disabled=!ok;}
 function canvas(id){const c=$(id),r=c.getBoundingClientRect(),d=window.devicePixelRatio||1,w=Math.max(1,r.width),h=Math.max(1,r.height);
  if(c.width!==Math.round(w*d)||c.height!==Math.round(h*d)){c.width=Math.round(w*d);c.height=Math.round(h*d);}const ctx=c.getContext('2d');ctx.setTransform(d,0,0,d,0,0);ctx.clearRect(0,0,w,h);return {ctx,w,h};}
 function textCenter(c,text){c.ctx.fillStyle='#a1b5c5';c.ctx.font='15px system-ui';c.ctx.textAlign='center';c.ctx.fillText(text,c.w/2,c.h/2);}
 function drawSlice(){
  const {ctx,w,h}=canvas('slice'),scale=number('zoom',65),cx=w/2+pan[0],cy=h*.8+pan[1],layers=active();
  ctx.font='11px system-ui';ctx.textAlign='left';ctx.lineWidth=1;
  const tick=scale<25?5:scale<55?2:1;
  for(let x=Math.floor(-cx/scale/tick)*tick;x<=(w-cx)/scale;x+=tick){const px=cx+x*scale;ctx.strokeStyle=x===0?'#385063':'#1c2c39';ctx.beginPath();ctx.moveTo(px,0);ctx.lineTo(px,h);ctx.stroke();ctx.fillStyle='#6f899d';ctx.fillText(x+' м',px+4,15);}
  for(let z=Math.floor((cy-h)/scale/tick)*tick;z<=cy/scale;z+=tick){const py=cy-z*scale;ctx.strokeStyle=z===0?'#385063':'#1c2c39';ctx.beginPath();ctx.moveTo(0,py);ctx.lineTo(w,py);ctx.stroke();ctx.fillStyle='#6f899d';ctx.fillText(z+' м',5,py-4);}
  if(!route){textCenter({ctx,w,h},meta?'Для этого кадра нет предсказанной пары':'Выберите прогон и кадр');return;}
  if(slicePoints)for(const layer of [0,1,2,4,23,20,21,22,30,31,32]){
   if(!layers.has(layer))continue;ctx.fillStyle=meta?.held_plane&&[21,22].includes(layer)?'#b0a398':colors[layer];const size=layer>=20?4:layer?2.5:1.5;
   for(let i=0;i<slicePoints.length;i+=6)if(slicePoints[i+2]===layer){const x=cx+slicePoints[i]*scale,y=cy-slicePoints[i+1]*scale;if(x>=0&&x<w&&y>=0&&y<h)ctx.fillRect(x-size/2,y-size/2,size,size);}
  }
  if(slicePoints&&$('showIntrusions').checked){ctx.fillStyle='#ff3d48';for(let i=0;i<slicePoints.length;i+=6)if(M.intrusion(slicePoints[i],slicePoints[i+1],slicePoints[i+2]))ctx.fillRect(cx+slicePoints[i]*scale-1.5,cy-slicePoints[i+1]*scale-1.5,3,3);}
  ctx.strokeStyle='#e9f2fb';ctx.lineWidth=2;ctx.strokeRect(cx-1.05*scale,cy-3*scale,2.1*scale,3*scale);
  ctx.setLineDash([4,4]);ctx.strokeStyle='#d4898c';ctx.lineWidth=1;ctx.strokeRect(cx-1.02*scale,cy-2.97*scale,2.04*scale,2.87*scale);ctx.setLineDash([]);
  ctx.fillStyle='#dce7f0';ctx.fillText(meta?.held_plane?'2,1 × 3 м · удержанная опорная плоскость':'2,1 м × 3 м · от головок рельсов',cx-1.05*scale,cy-3*scale-9);
  const lidarY=cy-1.5*scale;ctx.strokeStyle='#ecf5fb';ctx.beginPath();ctx.moveTo(cx-5,lidarY);ctx.lineTo(cx+5,lidarY);ctx.moveTo(cx,lidarY-5);ctx.lineTo(cx,lidarY+5);ctx.stroke();ctx.fillText('Центр поезда',cx+9,lidarY-7);
  if(selectedPoint){const x=cx+selectedPoint.x*scale,y=cy-selectedPoint.height*scale;ctx.strokeStyle='#fff';ctx.beginPath();ctx.arc(x,y,7,0,Math.PI*2);ctx.stroke();ctx.fillStyle='#fff';ctx.fillText(`${selectedPoint.range.toFixed(2)} м от лидара`,Math.min(w-160,Math.max(8,x+12)),Math.max(20,y-10));}
 }
 function selectPoint(p){selectedPoint=p;pause();$('pointInfo').textContent=`${p.layer>=20?'Расчётная точка':'Реальная точка'} · ${p.range.toFixed(3)} м от исходного лидара · по пути ${p.station.toFixed(2)} м · высота от головок ${p.height.toFixed(3)} м${M.intrusion(p.x,p.height,p.layer)?' · внутри на '+(M.penetration(p.x,p.height)*100).toFixed(1)+' см':''}`;drawSlice();}
 function showContacts(m){
  selectedPoint=null;$('pointInfo').textContent='Нажмите на точку среза или запись ниже: расстояние от исходного лидара.';
  $('contactInfo').textContent=`В габарите: ${m.count.toLocaleString('ru')} точек${m.nearest.length?' · ближайшая '+m.nearest[0].range.toFixed(2)+' м от лидара':''}`;
  $('contactList').replaceChildren();for(const p of m.nearest){const b=document.createElement('button');b.textContent=`${p.range.toFixed(2)} м от лидара · путь ${p.station.toFixed(2)} м`;b.onclick=()=>selectPoint(p);$('contactList').append(b);}
 }
 const dot=(p,a)=>p[0]*a[0]+p[1]*a[1]+p[2]*a[2];
 function topXY(p){return M.topView(p,topAxes);}
 function drawTop(){
  const c=canvas('top'),{ctx,w,h}=c;if(!route||!meta){topTransform=null;textCenter(c,'Путь появится при наличии прогноза');return;}
  const lines=[meta.curves.left,meta.curves.right,meta.curves.contact],all=lines.flat().map(topXY);
  const xs=all.map(p=>p[0]),ys=all.map(p=>p[1]);let xmin=Math.min(...xs)-3,xmax=Math.max(...xs)+3,ymin=Math.min(...ys)-2,ymax=Math.max(...ys)+2;
  const scale=Math.min((w-50)/(xmax-xmin),(h-40)/(ymax-ymin));const ox=(w-(xmax-xmin)*scale)/2-xmin*scale,oy=(h+(ymax-ymin)*scale)/2+ymin*scale;
  topTransform=p=>[ox+p[0]*scale,oy-p[1]*scale];const layers=active();
  for(const layer of [0,1,2])if(layers.has(layer)){ctx.fillStyle=colors[layer];ctx.globalAlpha=layer?0.9:.3;for(const p of topPoints)if(p[2]===layer){const [x,y]=topTransform(p);if(x>0&&x<w&&y>0&&y<h)ctx.fillRect(x,y,layer?2:1,layer?2:1);}}ctx.globalAlpha=1;
  for(const [name,layer] of [['left',21],['right',22],['contact',20],['center',23]]){
   if(!layers.has(layer))continue;ctx.strokeStyle=colors[layer];ctx.fillStyle=colors[layer];ctx.lineWidth=1;ctx.beginPath();
   const end=meta.extrapolation?.source_count??meta.curves[name].length;
   meta.curves[name].slice(0,end).forEach((p,i)=>{const [x,y]=topTransform(topXY(p));if(i)ctx.lineTo(x,y);else ctx.moveTo(x,y);});ctx.stroke();
   if(end<meta.curves[name].length){ctx.setLineDash([5,4]);ctx.beginPath();meta.curves[name].slice(end-1).forEach((p,i)=>{const [x,y]=topTransform(topXY(p));if(i)ctx.lineTo(x,y);else ctx.moveTo(x,y);});ctx.stroke();ctx.setLineDash([]);}
   for(let i=0;i<meta.curves[name].length;i+=5){const [x,y]=topTransform(topXY(meta.curves[name][i]));ctx.fillRect(x-1,y-1,2,2);}
  }
  if(layers.has(30)&&meta.run!=='OLD'&&meta.comparison_curves){ctx.strokeStyle='#ffffff';ctx.setLineDash([5,4]);for(const name of ['left','right','contact']){ctx.beginPath();meta.comparison_curves[name].forEach((p,i)=>{const [x,y]=topTransform(topXY(p));i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();}ctx.setLineDash([]);}
  const pose=M.sample(route,distance),p=topTransform(topXY(pose.position)),end=topTransform(topXY(pose.position.map((v,i)=>v+pose.forward[i]*3)));
  const angle=Math.atan2(end[1]-p[1],end[0]-p[0]);ctx.save();ctx.translate(...p);ctx.rotate(angle);ctx.fillStyle='#fff';ctx.beginPath();ctx.moveTo(10,0);ctx.lineTo(-6,-6);ctx.lineTo(-6,6);ctx.closePath();ctx.fill();ctx.restore();
  ctx.fillStyle='#a4bccf';ctx.textAlign='left';ctx.font='11px system-ui';ctx.fillText('Вперёд ↑ · стороны совпадают со срезом · белая стрелка — поезд',12,h-10);
 }
 function requestSlice(){
  if(!route||!ready)return;if(busy){pending=true;return;}
  const width=Math.min(number('width',1),Math.max(.0001,route.length-distance));busy=true;pending=false;
  worker.postMessage({type:'slice',token,request:++request,pose:M.sample(route,distance),width});
 }
 function sourceMessage(){return `${meta.real_points.toLocaleString('ru')} реальных + ${meta.synthetic_points.toLocaleString('ru')} расчётных точек. Кадр T=${meta.source_frame_index??index}; пакет ${meta.source_clouds.join(', ')}; облако фиксировано.${meta.extrapolation?.added_length_m?' За текущим ближним участком: '+Number(meta.extrapolation.added_length_m).toFixed(2)+' м.':''}`;}
 function setDistance(d){if(!route)return;distance=Math.max(0,Math.min(route.length,d));$('positionSeek').value=distance;$('position').textContent=`${distance.toFixed(2)} / ${route.length.toFixed(2)} м`;drawTop();requestSlice();if(distance>=route.length){pause();ended=true;status('Конец доступного прогноза. Выберите другой исходный кадр для отдельного проезда.');}else if(ended){ended=false;status(sourceMessage());}}
 function animate(t){animation=0;if(!playing)return;const dt=Math.min(.1,(t-lastTime)/1000);lastTime=t;setDistance(distance+dt*Math.min(160,number('speed',30))/3.6);if(playing)animation=requestAnimationFrame(animate);}
 function toggle(){if(!route)return;if(playing){pause();return;}if(distance>=route.length)setDistance(0);playing=true;$('play').textContent='Ⅱ Пауза';lastTime=performance.now();animation=requestAnimationFrame(animate);}
 async function loadFrame(i){
  pause();if(!run)return;index=Math.max(0,Math.min(run.frames.length-1,Math.round(i)));const source=run.frames[index],current=++token;
  abort?.abort();abort=new AbortController();worker?.terminate();worker=null;meta=null;route=null;ready=false;busy=false;pending=false;ended=false;slicePoints=null;selectedPoint=null;distance=0;enable(false);
  $('diagnosticNotice').textContent='';$('diagnosticNotice').hidden=true;
  $('contactInfo').textContent='Точки внутри габарита —';$('contactList').replaceChildren();$('pointInfo').textContent='Нажмите на точку среза, чтобы увидеть расстояние.';
  $('frame').value=source.source_frame_index;$('frameSeek').value=index;$('source').textContent=`${source.name} · ${index+1} / ${run.frames.length}`;
  $('sliceInfo').textContent='—';$('position').textContent='—';$('download').removeAttribute('href');$('positionSeek').value=0;drawSlice();drawTop();status('Читаю сохранённый пакет облаков и прогноз…');
  try{
   const url=`/api/predicted-train/frame?run=${encodeURIComponent(run.id)}&index=${index}&model=${initialParams.get('model')||''}&object=${initialParams.get('object')||''}&new_object=${initialParams.get('new_object')||''}&track_run=${initialParams.get('track_run')||''}`,response=await fetch(url,{signal:abort.signal});
   if(!response.ok)throw Error((await response.json()).error||'Не удалось прочитать исходные данные');
   const data=await response.arrayBuffer();if(current!==token)return;
   const size=new DataView(data).getUint32(0,true);meta=JSON.parse(new TextDecoder().decode(new Uint8Array(data,4,size)));
   if(meta.diagnostic_warning)meta.diagnostic_warning=meta.diagnostic_warning.replaceAll('HELD_COMPLETE_PIPELINE_PATH','Путь по полной цепочке').replaceAll('RAIL_FALLBACK_AFTER_FULL_STEP2_FAILURE','Коррекция прежнего пути по ходовым рельсам').replaceAll('STEP2: RIGHT','Контактный рельс: справа').replaceAll('STEP2: LEFT','Контактный рельс: слева');
   if(meta.diagnostic_warning){$('diagnosticNotice').textContent=meta.diagnostic_warning;$('diagnosticNotice').hidden=false;}
   const rows=new Float32Array(data,4+size);if(rows.length!==meta.point_count*4)throw Error('Неполный ответ с точками');
   $('download').href='/?job='+resultJob;$('download').textContent='Вернуться к результатам';
   if(!meta.available){status(`Прогноз отсутствует: ${meta.status} · ${meta.reason}. В облаке ${meta.real_points.toLocaleString('ru')} реальных точек.`,true);drawSlice();drawTop();return;}
   route=M.build(meta.curves,meta.up);route.offset=[0,0,1.5];topAxes=M.sample(route,0);topPoints=[];
   if(initialParams.get('focus')==='1'){let selected=[];for(let j=0;j<meta.real_points*4;j+=4)if(rows[j+3]===4)selected.push([rows[j],rows[j+1],rows[j+2]]);if(selected.length){let c=[0,1,2].map(k=>{let a=selected.map(p=>p[k]).sort((a,b)=>a-b);return a[Math.floor(a.length/2)];}),best=Infinity,at=0;for(let k=0;k<route.nodes.length-1;k++){let a=route.nodes[k],b=route.nodes[k+1],v=b.center.map((x,i)=>x-a.center[i]),w=c.map((x,i)=>x-a.center[i]),len=v.reduce((s,x)=>s+x*x,0),t=Math.max(0,Math.min(1,v.reduce((s,x,i)=>s+x*w[i],0)/len)),err=v.reduce((s,x,i)=>s+(w[i]-t*x)**2,0);if(err<best){best=err;at=a.distance+t*(b.distance-a.distance);}}pendingDistance=Math.max(0,at-number('width',1)/2);}}

   for(let j=0;j<rows.length;j+=4){const layer=rows[j+3];if(layer>=20||(!layer&&(j/4)%19))continue;const p=topXY(rows.subarray(j,j+3));if(p[1]>=-10&&p[1]<=route.length+10&&Math.abs(p[0])<20)topPoints.push([...p,layer]);}
   $('positionSeek').max=route.length;enable(true);pan=[0,0];
   worker=new Worker('/predicted_train_worker.js');
   worker.onmessage=event=>{
    const m=event.data;if(m.token!==token)return;
    if(m.type==='ready'){ready=true;status(sourceMessage()+` Подготовка проезда: ${m.ms.toFixed(0)} мс.`);const d=pendingDistance;pendingDistance=0;setDistance(d);return;}
    if(m.type==='error'){pause();busy=false;status(m.error,true);return;}
    if(m.type==='slice'){
     busy=false;
     slicePoints=new Float32Array(m.buffer);sliceDistance=m.distance;sliceWidth=m.width;
     const assumed=meta.extrapolation?.source_length_m!=null&&sliceDistance+sliceWidth>meta.extrapolation.source_length_m;
     $('sliceInfo').textContent=`Срез на ${sliceDistance.toFixed(2)} м · слой ${sliceWidth.toFixed(2)} м · ${(slicePoints.length/6).toLocaleString('ru')} точек · ${m.ms.toFixed(1)} мс${assumed?' · ДАЛЬНЯЯ ГЕОМЕТРИЯ':''}`;showContacts(m);drawSlice();
     if(pending)requestSlice();
    }
   };
   worker.onerror=()=>{pause();ready=false;busy=false;status('Ошибка расчёта среза. Перезагрузите выбранный кадр.',true);};
   const buffer=rows.slice().buffer;worker.postMessage({type:'load',token,buffer,route,inspection_plane:meta.inspection_plane},[buffer]);
   status(sourceMessage());
  }catch(error){if(current===token&&error.name!=='AbortError'){status(error.message,true);meta=null;route=null;enable(false);drawSlice();drawTop();}}
 }
 function selectRun(){const oldSource=meta?.source_frame_index??run?.frames[index]?.source_frame_index;if(!initialSource)pendingDistance=distance;run=runs.find(r=>r.id===$('run').value);if(!run)return;$('frame').max=run.frames.at(-1).source_frame_index;$('frameSeek').max=run.frames.length-1;const wanted=initialSource?Number(initialParams.get('source')||820):oldSource;const source=run.frames.findIndex(f=>f.source_frame_index===wanted);initialSource=false;loadFrame(Math.max(0,source));}
 $('run').onchange=selectRun;$('frame').onchange=()=>{const s=Number($('frame').value);let k=run.frames.findIndex(f=>f.source_frame_index>=s);loadFrame(k<0?run.frames.length-1:k);};$('frameSeek').onchange=()=>loadFrame(Number($('frameSeek').value));
 $('prevFrame').onclick=()=>loadFrame(index-1);$('nextFrame').onclick=()=>loadFrame(index+1);
 $('play').onclick=toggle;$('back').onclick=()=>{pause();setDistance(distance-number('step',1));};$('forward').onclick=()=>{pause();setDistance(distance+number('step',1));};
 $('start').onclick=()=>{pause();setDistance(0);};$('positionSeek').oninput=()=>{pause();setDistance(+$('positionSeek').value);};
 $('width').onchange=()=>{if(!Number.isFinite(+$('width').value)||+$('width').value<=0){$('width').value=1;}requestSlice();};
 $('zoom').oninput=drawSlice;$('resetView').onclick=()=>{pan=[0,0];$('zoom').value=65;drawSlice();};
 $('showIntrusions').onchange=drawSlice;
 document.querySelectorAll('[data-step]').forEach(b=>b.onclick=()=>{$('step').value=b.dataset.step;});
 document.querySelectorAll('[data-layer]').forEach(e=>e.onchange=()=>{drawSlice();drawTop();});
 $('slice').onpointerdown=e=>{drag=[e.clientX,e.clientY,...pan];$('slice').setPointerCapture(e.pointerId);};
 $('slice').onpointermove=e=>{if(drag){pan=[drag[2]+e.clientX-drag[0],drag[3]+e.clientY-drag[1]];drawSlice();}};
 $('slice').onpointerup=e=>{if(drag&&Math.hypot(e.clientX-drag[0],e.clientY-drag[1])<4&&slicePoints){
  const rect=$('slice').getBoundingClientRect(),scale=number('zoom',65),cx=rect.width/2+pan[0],cy=rect.height*.8+pan[1],px=e.clientX-rect.left,py=e.clientY-rect.top,layers=active();let best=81,p=null;
  for(let i=0;i<slicePoints.length;i+=6){if(!layers.has(slicePoints[i+2])&&!($('showIntrusions').checked&&M.intrusion(slicePoints[i],slicePoints[i+1],slicePoints[i+2])))continue;const d=(cx+slicePoints[i]*scale-px)**2+(cy-slicePoints[i+1]*scale-py)**2;if(d<best){best=d;p={x:slicePoints[i],height:slicePoints[i+1],layer:slicePoints[i+2],range:slicePoints[i+3],station:slicePoints[i+4],row:slicePoints[i+5]};}}
  if(p)selectPoint(p);
 }drag=null;};$('slice').onpointercancel=()=>{drag=null;};
 $('slice').addEventListener('wheel',e=>{e.preventDefault();$('zoom').value=Math.max(10,Math.min(240,number('zoom',65)*Math.exp(-e.deltaY*.001)));drawSlice();},{passive:false});
 $('top').onclick=e=>{if(!route||!topTransform)return;const r=$('top').getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top;let best=Infinity,at=0;route.nodes.forEach(n=>{const p=topTransform(topXY(n.center)),d=(p[0]-x)**2+(p[1]-y)**2;if(d<best){best=d;at=n.distance;}});pause();setDistance(at);};
 document.addEventListener('keydown',e=>{if(['INPUT','SELECT','TEXTAREA','BUTTON','A'].includes(e.target.tagName))return;if(e.code==='Space'){e.preventDefault();toggle();}if(['ArrowLeft','ArrowRight'].includes(e.code)){e.preventDefault();pause();setDistance(distance+(e.code==='ArrowLeft'?-1:1)*number('step',1));}});
 document.addEventListener('visibilitychange',()=>{if(document.hidden)pause();});
 new ResizeObserver(()=>{drawSlice();drawTop();}).observe(document.querySelector('.workspace'));
 async function init(){enable(false);try{const response=await fetch('/api/predicted-train/runs');if(!response.ok)throw Error('Сервер сравнения недоступен');const data=await response.json();runs=data.runs;for(const r of runs){const o=document.createElement('option');o.value=r.id;o.textContent=r.label;$('run').append(o);}if(!runs.length){status('Нет сохранённых прогнозов',true);return;}$('run').value=initialParams.get('run')||resultJob||runs[0].id;selectRun();const cases=await (await fetch('/api/review-cases')).json();for(const c of cases){const b=document.createElement('button');b.dataset.reviewSource=c.source;b.textContent=`${c.source} · ${['HELD_COMPLETE_PIPELINE_PATH','HELD_PREVIOUS_VERIFIED_FULL_CHAIN'].includes(c.label)?'полный прогноз':c.label==='HELD_NEAR_PLANE'?'ближняя опора':c.label.startsWith('NEAR_ONLY')||c.label==='CURRENT_NEAR_ONLY'?'только 8 м':c.label==='RAIL_FALLBACK_AFTER_FULL_STEP2_FAILURE'?'по ходовым':c.near.count?'попадания':'осмотр'}`;b.onclick=()=>{pause();$('run').value='FULL';run=runs.find(r=>r.id==='FULL');$('frame').max=run.frames.at(-1).source_frame_index;$('frameSeek').max=run.frames.length-1;pendingDistance=c.near.count?Math.min(7,Math.floor(c.near.nearest_range_m*2)/2):c.default_distance;loadFrame(run.frames.findIndex(f=>f.source_frame_index===c.source));};$(c.far||c.source%2?'cases':'failureCases').append(b);}}catch(e){status(e.message,true);}}
 init();
})();


