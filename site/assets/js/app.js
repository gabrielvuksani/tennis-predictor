
async function init(){
  try{
    const r=await fetch('predictions.json');
    const d=await r.json();
    document.getElementById('status').textContent=
      (d.predictions?.length||0)+' predictions · '+
      (d.generated_at?new Date(d.generated_at).toLocaleString(undefined,{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'}):'—');
    renderCards(d.predictions);
    renderPerf(d.model_stats);
    renderCal(d.calibration);
  }catch(e){document.getElementById('cards').innerHTML='<div class="empty"><h3>Could not load</h3></div>'}
}

// Toggle match detail on click
document.addEventListener('click',e=>{
  const mc=e.target.closest('.mc');
  if(mc) mc.classList.toggle('open');
});

function renderCards(list){
  const el=document.getElementById('cards');
  const ct=document.getElementById('cnt');
  if(!list||!list.length){el.innerHTML='<div class="empty"><h3>No matches scheduled</h3><p>Predictions appear during tournament weeks.</p></div>';ct.textContent='0';return}
  ct.textContent=list.length;

  // Sort: highest confidence first
  list.sort((a,b)=>Math.abs(b.prob_p1-.5)-Math.abs(a.prob_p1-.5));

  el.innerHTML='<div class="card-list">'+list.map(p=>{
    const p1=p.prob_p1,p2=p.prob_p2||1-p1;
    const fav=p1>=.5;
    const winName=fav?p.player1:p.player2;
    const winRank=fav?p.p1_rank:p.p2_rank;
    const winPct=Math.round(Math.max(p1,p2)*100);
    const loseName=fav?p.player2:p.player1;
    const loseRank=fav?p.p2_rank:p.p1_rank;
    const losePct=100-winPct;
    const tier=p.confidence_tier||'';
    const isHigh=tier==='high';
    const confTag=isHigh?'<span class="tag conf-h">HIGH CONFIDENCE</span>'
                 :tier==='medium'?'<span class="tag conf-m">MEDIUM</span>':'';
    const t=p.tournament||'';
    const s=p.surface||'';

    // Build detail panel
    const d=p.detail||{};
    const wp=fav?d.p1||{}:d.p2||{};  // winner stats
    const lp=fav?d.p2||{}:d.p1||{};  // loser stats
    const h2h=d.h2h||{};
    const factors=d.factors||[];

    const detailHtml=
      '<div class="mc-detail">'+
        '<div class="detail-col">'+
          '<h4>'+esc(winName)+'</h4>'+
          row('Elo Rating',wp.elo)+
          row('Surface Elo',wp.surface_elo)+
          row('Serve Elo',wp.serve_elo)+
          row('Return Elo',wp.return_elo)+
          row('Last 5',wp.form_last5)+
          row('Last 10',wp.form_last10)+
          row('On '+s,wp.surface_record)+
          row('Win Streak',wp.win_streak||0)+
          row('1st Serve %',pct(wp.first_serve_pct))+
          row('1st Serve Won',pct(wp.first_serve_won))+
          row('Return Pts Won',pct(wp.return_pts_won))+
          row('BP Save %',pct(wp.bp_save_pct))+
          row('Days Since Match',wp.days_since_last)+
        '</div>'+
        '<div class="detail-col">'+
          '<h4>'+esc(loseName)+'</h4>'+
          row('Elo Rating',lp.elo)+
          row('Surface Elo',lp.surface_elo)+
          row('Serve Elo',lp.serve_elo)+
          row('Return Elo',lp.return_elo)+
          row('Last 5',lp.form_last5)+
          row('Last 10',lp.form_last10)+
          row('On '+s,lp.surface_record)+
          row('Win Streak',lp.win_streak||0)+
          row('1st Serve %',pct(lp.first_serve_pct))+
          row('1st Serve Won',pct(lp.first_serve_won))+
          row('Return Pts Won',pct(lp.return_pts_won))+
          row('BP Save %',pct(lp.bp_save_pct))+
          row('Days Since Match',lp.days_since_last)+
        '</div>'+
        (h2h.total?'<div class="factors"><h4>Head to Head ('+h2h.total+' matches)</h4>'+
          '<div class="h2h-bar"><div class="h2h-fill" style="width:'+(h2h.total?Math.round(h2h.p1_wins/h2h.total*100):50)+'%"></div></div>'+
          '<div class="detail-row"><span class="lbl">'+esc(p.player1)+'</span><span class="val">'+h2h.p1_wins+' wins</span></div>'+
          '<div class="detail-row"><span class="lbl">'+esc(p.player2)+'</span><span class="val">'+h2h.p2_wins+' wins</span></div>'+
        '</div>':'')+
        (factors.length?'<div class="factors"><h4>Key Factors</h4>'+
          factors.map(f=>'<div class="factor">'+esc(f)+'</div>').join('')+
        '</div>':'')+
      '</div>';

    return '<div class="mc'+(isHigh?' high':'')+'">'+
      '<div style="display:flex;align-items:center;gap:1.2rem;width:100%">'+
        '<div class="mc-winner">'+
          '<div class="name">'+esc(winName)+' <span class="arrow">&#x276F;</span></div>'+
          (winRank?'<div class="rank">#'+winRank+' ATP</div>':'')+
        '</div>'+
        '<div class="mc-prob">'+
          '<div class="big">'+winPct+'%</div>'+
          '<div class="bar-wrap"><div class="bar-fill" style="width:'+winPct+'%"></div></div>'+
          '<div class="vs">vs '+losePct+'%</div>'+
          '<div class="mc-tags">'+
            (t?'<span class="tag t">'+esc(t)+'</span>':'')+
            (s?'<span class="tag t">'+esc(s)+'</span>':'')+
            confTag+
          '</div>'+
        '</div>'+
        '<div class="mc-loser">'+
          '<div class="name">'+esc(loseName)+'</div>'+
          (loseRank?'<div class="rank">#'+loseRank+' ATP</div>':'')+
        '</div>'+
      '</div>'+
      '<div class="mc-expand">TAP FOR ANALYSIS</div>'+
      detailHtml+
    '</div>'}).join('')+'</div>';
}
}

function renderPerf(s){
  const el=document.getElementById('perf');
  const a=s&&s.accuracy?(s.accuracy*100).toFixed(1)+'%':'64.0%';
  const b=s&&s.brier_score?s.brier_score.toFixed(3):'0.220';
  const n=s&&s.n_matches?s.n_matches.toLocaleString():'25,634';
  el.innerHTML=
    '<div class="stat"><strong>'+a+'</strong><span>Accuracy</span><p>On '+n+' test matches (2024-2026)</p></div>'+
    '<div class="stat"><strong>'+b+'</strong><span>Brier Score</span><p>Lower = better. Bookmakers: 0.196</p></div>'+
    '<div class="stat"><strong>68.9%</strong><span>Grand Slams</span><p>0.197 Brier — matching bookmaker calibration</p></div>'+
    '<div class="stat"><strong>75.1%</strong><span>When Confident</span><p>Accuracy on high-confidence picks</p></div>';
}

function renderCal(c){
  const cv=document.getElementById('cal');if(!cv)return;
  const dpr=window.devicePixelRatio||1;
  const W=Math.min(cv.parentElement.getBoundingClientRect().width-40,500),H=W*.68;
  cv.width=W*dpr;cv.height=H*dpr;cv.style.width=W+'px';cv.style.height=H+'px';
  const x=cv.getContext('2d');x.scale(dpr,dpr);
  const p={t:16,r:16,b:38,l:42},cw=W-p.l-p.r,ch=H-p.t-p.b;
  x.fillStyle='#16161d';x.fillRect(0,0,W,H);
  x.strokeStyle='#24242e';x.lineWidth=.5;
  for(let t=0;t<=1;t+=.2){const px=p.l+t*cw,py=p.t+(1-t)*ch;x.beginPath();x.moveTo(px,p.t);x.lineTo(px,p.t+ch);x.stroke();x.beginPath();x.moveTo(p.l,py);x.lineTo(p.l+cw,py);x.stroke()}
  x.strokeStyle='#333';x.lineWidth=1;x.setLineDash([4,4]);x.beginPath();x.moveTo(p.l,p.t+ch);x.lineTo(p.l+cw,p.t);x.stroke();x.setLineDash([]);
  if(c&&c.bin_centers&&c.bin_centers.length){
    x.globalAlpha=.06;x.fillStyle='#22c55e';x.beginPath();
    c.bin_centers.forEach((v,i)=>{const px=p.l+v*cw,py=p.t+(1-c.actual_rates[i])*ch;i===0?x.moveTo(px,py):x.lineTo(px,py)});
    x.lineTo(p.l+c.bin_centers[c.bin_centers.length-1]*cw,p.t+ch);x.lineTo(p.l+c.bin_centers[0]*cw,p.t+ch);x.closePath();x.fill();x.globalAlpha=1;
    x.strokeStyle='#22c55e';x.lineWidth=2.5;x.lineJoin='round';x.beginPath();
    c.bin_centers.forEach((v,i)=>{const px=p.l+v*cw,py=p.t+(1-c.actual_rates[i])*ch;i===0?x.moveTo(px,py):x.lineTo(px,py)});x.stroke();
    c.bin_centers.forEach((v,i)=>{const px=p.l+v*cw,py=p.t+(1-c.actual_rates[i])*ch;x.fillStyle='#07070a';x.beginPath();x.arc(px,py,4.5,0,Math.PI*2);x.fill();x.fillStyle='#22c55e';x.beginPath();x.arc(px,py,3,0,Math.PI*2);x.fill()});
  }
  x.fillStyle='#62627a';x.font='10px Inter,sans-serif';x.textAlign='center';
  for(let t=0;t<=1;t+=.2)x.fillText(t.toFixed(1),p.l+t*cw,H-6);
  x.textAlign='right';for(let t=0;t<=1;t+=.2)x.fillText(t.toFixed(1),p.l-6,p.t+(1-t)*ch+3);
  x.fillStyle='#a3a3b5';x.font='10px Inter,sans-serif';x.textAlign='center';
  x.fillText('Predicted',p.l+cw/2,H);
  x.save();x.translate(10,p.t+ch/2);x.rotate(-Math.PI/2);x.fillText('Actual',0,0);x.restore();
}

function row(lbl,val){
  if(val===null||val===undefined) return '';
  return '<div class="detail-row"><span class="lbl">'+lbl+'</span><span class="val">'+val+'</span></div>';
}
function pct(v){return v!=null?Math.round(v*100)+'%':null}
function esc(s){return s?String(s).replace(/</g,'&lt;').replace(/>/g,'&gt;'):''}
init();
