"""Phone as a wireless scanner: a tiny web server on the local network.
The phone only needs a browser. Only the pages of this program are served - no file of the computer is reachable."""
import hmac
import io
import ipaddress
import json
import os
import re
import secrets
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from PIL import Image
from PySide6.QtCore import QObject, Signal

import scanner

MAX_BODY = 60 * 1024 * 1024
EXT = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "BMP": ".bmp", "GIF": ".gif", "TIFF": ".tif",
       "MPO": ".jpg", "HEIF": ".heic"}


def local_ip():
    """The address of this computer on the Wi-Fi / LAN (None if there is no network)."""
    cands = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        cands.append(s.getsockname()[0])
        s.close()
    except Exception:
        pass
    try:
        cands += socket.gethostbyname_ex(socket.gethostname())[2]
    except Exception:
        pass
    for ip in cands:
        try:
            a = ipaddress.ip_address(ip)
            if a.is_private and not a.is_loopback and not a.is_link_local:
                return ip
        except ValueError:
            pass
    return None


class _Handler(BaseHTTPRequestHandler):
    server_version = "ImageFixer"
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    # ---- helpers
    def _send(self, code, body=b"", ctype="text/plain; charset=utf-8", extra=None):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj), "application/json")

    def _allowed(self):
        """(route, query) if the client is on the local network and the token is right, else None."""
        try:
            a = ipaddress.ip_address(self.client_address[0].split("%")[0])
            if not (a.is_private or a.is_loopback):
                self._send(403, "Only the local network can use this.")
                return None
        except ValueError:
            self._send(403, "")
            return None
        u = urlparse(self.path)
        self.server.owner.touch()
        return (u.path or "/"), parse_qs(u.query)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > MAX_BODY:
            self._json({"ok": False, "error": "File too big or empty"}, 413)
            return None
        return self.rfile.read(n)

    # ---- routes
    def do_GET(self):
        r = self._allowed()
        if r is None:
            return
        route, _ = r
        if route == "/":
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif route == "/api/ping":
            self._json({"ok": True})
        else:
            self._send(404, "")

    def do_POST(self):
        r = self._allowed()
        if r is None:
            return
        route, q = r
        data = self._body()
        if data is None:
            return
        try:
            if route == "/api/detect":
                im = Image.open(io.BytesIO(data))
                im.load()
                quad, found = scanner.detect_document(im)
                self._json({"ok": True, "w": im.width, "h": im.height, "found": found,
                            "quad": [[x, y] for x, y in quad]})
            elif route == "/api/warp":
                vals = [float(v) for v in q.get("q", [""])[0].split(",")]
                if len(vals) != 8:
                    raise ValueError("4 corners needed")
                im = Image.open(io.BytesIO(data)).convert("RGB")
                quad = [(vals[i], vals[i + 1]) for i in range(0, 8, 2)]
                out = scanner.straighten(im, quad)
                buf = io.BytesIO()
                out.save(buf, "JPEG", quality=93, subsampling=0)
                self._send(200, buf.getvalue(), "image/jpeg")
            elif route == "/api/upload":
                self._upload(data, q)
            else:
                self._send(404, "")
        except Exception as ex:
            self._json({"ok": False, "error": str(ex)[:200]}, 400)

    def _upload(self, data, q):
        try:
            im = Image.open(io.BytesIO(data))
            fmt = im.format
            im.verify()
        except Exception:
            self._json({"ok": False, "error": "Not a picture this program can open"}, 400)
            return
        batch = re.sub(r"[^A-Za-z0-9]", "", q.get("batch", ["x"])[0])[:24] or "x"
        i = max(0, int(q.get("i", ["0"])[0]))
        n = max(1, min(500, int(q.get("n", ["1"])[0])))
        kind = q.get("kind", ["file"])[0]
        name = os.path.basename(q.get("name", ["picture"])[0]) or "picture"
        stem = re.sub(r"[^\w\-. ]", "_", os.path.splitext(name)[0])[:60] or "picture"
        ext = EXT.get(fmt, ".jpg")
        self._json({"ok": True, **self.server.owner.store(batch, i, n, kind, stem, ext, data)})


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class PhoneServer(QObject):
    received = Signal(list)          # file paths, in the order the phone sent them
    activity = Signal()

    def __init__(self, inbox, parent=None):
        super().__init__(parent)
        self.inbox = inbox
        os.makedirs(inbox, exist_ok=True)
        self.token = secrets.token_urlsafe(16)
        self.port = 0
        self.httpd = None
        self.last_seen = 0.0
        self.total = 0
        self._batches = {}
        self._lock = threading.Lock()

    # ---- control
    def start(self, preferred_port=0):
        if self.httpd is not None:
            return True
        for port in ([preferred_port] if preferred_port else []) + [0]:
            try:
                self.httpd = _Server(("0.0.0.0", port), _Handler)
                break
            except OSError:
                self.httpd = None
        if self.httpd is None:
            return False
        self.httpd.token = self.token
        self.httpd.owner = self
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return True

    def stop(self):
        if self.httpd is not None:
            try:
                self.httpd.shutdown()
                self.httpd.server_close()
            except Exception:
                pass
        self.httpd = None

    def running(self):
        return self.httpd is not None

    def url(self):
        ip = local_ip()
        return None if ip is None or not self.running() else "http://%s:%d" % (ip, self.port)

    def phone_online(self):
        return time.time() - self.last_seen < 12.0

    # ---- called from the server threads
    def touch(self):
        self.last_seen = time.time()
        self.activity.emit()

    def store(self, batch, i, n, kind, stem, ext, data):
        with self._lock:
            b = self._batches.setdefault(batch, {"n": n, "files": {}, "t": time.time()})
            b["n"] = n
            fname = ("Scan_%s_%02d%s" % (time.strftime("%Y%m%d_%H%M%S"), i + 1, ext)) if kind == "scan" else \
                (stem + ext)
            folder = os.path.join(self.inbox, batch)
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(folder, "%03d_%s" % (i, fname))
            with open(path, "wb") as f:
                f.write(data)
            b["files"][i] = path
            done = len(b["files"]) >= b["n"]
            if done:
                paths = [b["files"][k] for k in sorted(b["files"])]
                del self._batches[batch]
            for k in [k for k, v in self._batches.items() if time.time() - v["t"] > 3600]:
                del self._batches[k]
        if done:
            self.total += len(paths)
            self.received.emit(paths)
        return {"stored": len(b["files"]), "done": done}


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Image Fixer</title>
<style>
*{box-sizing:border-box;-webkit-tap-highlight-color:transparent}
body{margin:0;font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:#0e1116;color:#e8eaed;
min-height:100vh;display:flex;flex-direction:column}
header{padding:14px 16px;display:flex;align-items:center;gap:10px;border-bottom:1px solid #1c2130}
header b{font-size:18px;flex:1}
#dot{width:10px;height:10px;border-radius:50%;background:#e5484d}#dot.on{background:#30a46c}
main{flex:1;padding:16px;display:flex;flex-direction:column;gap:14px}
.big{display:flex;flex-direction:column;align-items:center;justify-content:center;gap:6px;flex:1;min-height:150px;
border-radius:22px;border:0;color:#fff;font-size:22px;font-weight:700}
.big small{font-weight:400;font-size:14px;opacity:.85}
.choose{background:#222836;border:1px solid #30374a}
.scan{background:linear-gradient(90deg,#6366f1,#a855f7)}
button{font:inherit}
.btn{border:0;border-radius:12px;padding:13px 16px;background:#222836;color:#e8eaed;font-size:16px;border:1px solid #30374a}
.btn.pri{background:linear-gradient(90deg,#6366f1,#a855f7);border:0;color:#fff;font-weight:600}
.btn.red{background:#3a1b22;border-color:#5a2630;color:#ffb4bd}
.row{display:flex;gap:10px}.row>*{flex:1}
.hide{display:none!important}
#msg{min-height:20px;color:#8d95a8;text-align:center;font-size:14px}
canvas{width:100%;touch-action:none;border-radius:10px;background:#000}
.pages{display:flex;flex-direction:column;gap:10px}
.pg{display:flex;gap:10px;align-items:center;background:#171b24;border:1px solid #262c3b;border-radius:14px;padding:8px}
.pg img{width:70px;height:90px;object-fit:cover;border-radius:8px;background:#000}
.pg .n{flex:1;font-weight:600}
.pg button{padding:9px 11px;font-size:15px}
#viewer{position:fixed;inset:0;background:#000d;display:flex;align-items:center;justify-content:center;padding:12px}
#viewer img{max-width:100%;max-height:100%}
#bar{height:6px;background:#222836;border-radius:3px;overflow:hidden}#bar i{display:block;height:100%;width:0;background:#6366f1}
</style></head><body>
<header><b>Image Fixer</b><span id="st">connecting...</span><span id="dot"></span></header>
<main>
 <section id="home" style="display:contents">
  <button class="big choose" id="bChoose">Choose file<small>pictures from this phone</small></button>
  <button class="big scan" id="bScan">Scan<small>photograph a paper</small></button>
 </section>
 <section id="edit" class="hide">
  <div class="row"><div id="tip" style="flex:3;color:#8d95a8;font-size:14px">Drag the 4 corners onto the corners of the paper.</div></div>
  <canvas id="cv"></canvas>
  <div class="row"><button class="btn" id="eRetake">Retake</button><button class="btn pri" id="eOk">Crop</button></div>
 </section>
 <section id="review" class="hide">
  <img id="rimg" style="width:100%;border-radius:10px;background:#000">
  <div class="row"><button class="btn" id="rAdj">Adjust corners</button><button class="btn pri" id="rUse">Use this page</button></div>
 </section>
 <section id="list" class="hide">
  <div class="row" style="align-items:center"><b id="lt" style="flex:2;font-size:18px"></b><button class="btn" id="lHome" style="flex:1">Home</button></div>
  <div class="pages" id="pages"></div>
  <button class="btn" id="lAdd">+ Add more</button>
  <button class="btn pri" id="lSend">Send to desktop</button>
 </section>
 <div id="bar" class="hide"><i></i></div>
 <div id="msg"></div>
</main>
<input type="file" id="fFiles" accept="image/*" multiple class="hide">
<input type="file" id="fCam" accept="image/*" capture="environment" class="hide">
<div id="viewer" class="hide"><img id="vimg"></div>
<script>
const $=id=>document.getElementById(id), base=location.pathname.replace(/\/$/,'');
let pages=[], retake=-1, cur=null, busy=false;
const api=(p,opt)=>fetch(base+p,opt);
function show(id){for(const s of ['home','edit','review','list'])$(s).classList.toggle('hide',s!==id);
 $('home').style.display=id==='home'?'contents':'';}
function msg(t,err){$('msg').textContent=t||'';$('msg').style.color=err?'#ffb4bd':'#8d95a8';}
function prog(f){$('bar').classList.toggle('hide',f==null);if(f!=null)$('bar').firstChild.style.width=(f*100)+'%';}
async function ping(){try{const r=await api('/api/ping');if(!r.ok)throw 0;$('dot').className='on';$('st').textContent='connected';}
 catch(e){$('dot').className='';$('st').textContent='not connected';}}
setInterval(ping,4000);ping();

function loadImg(src){return new Promise((ok,no)=>{const i=new Image();i.onload=()=>ok(i);i.onerror=no;i.src=src;});}
async function shrink(file,maxSide){ // photo -> upright JPEG blob, not bigger than maxSide
 const u=URL.createObjectURL(file);const im=await loadImg(u);
 let w=im.naturalWidth,h=im.naturalHeight,s=Math.min(1,maxSide/Math.max(w,h));
 const c=document.createElement('canvas');c.width=Math.round(w*s);c.height=Math.round(h*s);
 c.getContext('2d').drawImage(im,0,0,c.width,c.height);URL.revokeObjectURL(u);
 return new Promise(r=>c.toBlob(r,'image/jpeg',0.93));}

// ---------- choose files
$('bChoose').onclick=()=>{$('fFiles').value='';$('fFiles').click();};
$('fFiles').onchange=async e=>{
 const fs=[...e.target.files];if(!fs.length)return;
 const batch=Date.now().toString(36)+Math.random().toString(36).slice(2,6);let ok=0;
 for(let i=0;i<fs.length;i++){msg('Uploading '+(i+1)+' / '+fs.length+' ...');prog(i/fs.length);
  try{const r=await api('/api/upload?kind=file&batch='+batch+'&i='+i+'&n='+fs.length+'&name='+encodeURIComponent(fs[i].name),
     {method:'POST',body:fs[i]});const j=await r.json();if(j.ok)ok++;else msg(fs[i].name+': '+j.error,true);}
  catch(err){msg('Connection lost. Scan the QR code on the computer again.',true);prog(null);return;}}
 prog(null);msg(ok+' of '+fs.length+' picture(s) sent to the computer');};

// ---------- scan
$('bScan').onclick=()=>{retake=-1;openCam();};
function openCam(){$('fCam').value='';$('fCam').click();}
$('fCam').onchange=async e=>{const f=e.target.files[0];if(!f)return;
 msg('Finding the paper ...');prog(0.3);
 try{const blob=await shrink(f,2600);
  const r=await api('/api/detect',{method:'POST',body:blob});const j=await r.json();
  if(!j.ok)throw new Error(j.error);
  cur={orig:blob,w:j.w,h:j.h,quad:j.quad.map(p=>({x:p[0],y:p[1]})),img:await loadImg(URL.createObjectURL(blob))};
  $('tip').textContent=j.found?'Paper found. Move a corner if the outline is not exact.':'Paper not found - drag the 4 corners onto the paper.';
  prog(null);msg('');startEdit();
 }catch(err){prog(null);msg('Could not connect to the computer. Scan the QR code again.',true);}};

const cv=$('cv'),cx=cv.getContext('2d');let drag=-1,scale=1,touch=null;
function startEdit(){show('edit');const W=Math.min(innerWidth-32,900),r=cur.h/cur.w;
 const d=devicePixelRatio||1;cv.style.width=W+'px';cv.style.height=(W*r)+'px';cv.width=W*d;cv.height=W*r*d;scale=cv.width/cur.w;draw();}
function draw(){cx.clearRect(0,0,cv.width,cv.height);cx.drawImage(cur.img,0,0,cv.width,cv.height);
 const q=cur.quad.map(p=>({x:p.x*scale,y:p.y*scale}));
 cx.save();cx.beginPath();cx.rect(0,0,cv.width,cv.height);cx.moveTo(q[0].x,q[0].y);for(let i=3;i>=0;i--)cx.lineTo(q[i].x,q[i].y);
 cx.closePath();cx.fillStyle='rgba(0,0,0,.45)';cx.fill('evenodd');cx.restore();
 cx.beginPath();q.forEach((p,i)=>i?cx.lineTo(p.x,p.y):cx.moveTo(p.x,p.y));cx.closePath();
 cx.lineWidth=2*(devicePixelRatio||1);cx.strokeStyle='#a5b4fc';cx.stroke();
 const R=13*(devicePixelRatio||1);q.forEach(p=>{cx.beginPath();cx.arc(p.x,p.y,R,0,7);cx.fillStyle='rgba(255,255,255,.9)';cx.fill();cx.lineWidth=3;cx.strokeStyle='#4f46e5';cx.stroke();});
 if(drag>=0){ // magnifier
  const p=q[drag],M=70*(devicePixelRatio||1),z=3,mx=Math.min(Math.max(p.x,M),cv.width-M),my=p.y>M*2.6?p.y-M*1.6:p.y+M*1.6;
  cx.save();cx.beginPath();cx.arc(mx,my,M,0,7);cx.clip();
  cx.drawImage(cur.img,(p.x/scale)-M/z/scale*1,(p.y/scale)-M/z/scale*1,2*M/z/scale,2*M/z/scale,mx-M,my-M,2*M,2*M);
  cx.restore();cx.beginPath();cx.arc(mx,my,M,0,7);cx.lineWidth=3;cx.strokeStyle='#fff';cx.stroke();
  cx.beginPath();cx.moveTo(mx-10,my);cx.lineTo(mx+10,my);cx.moveTo(mx,my-10);cx.lineTo(mx,my+10);cx.strokeStyle='#f43f5e';cx.lineWidth=2;cx.stroke();}}
function pos(e){const r=cv.getBoundingClientRect(),d=cv.width/r.width;return{x:(e.clientX-r.left)*d,y:(e.clientY-r.top)*d};}
cv.onpointerdown=e=>{const p=pos(e);let best=-1,bd=1e9;cur.quad.forEach((c,i)=>{const d=Math.hypot(c.x*scale-p.x,c.y*scale-p.y);if(d<bd){bd=d;best=i;}});
 if(bd<60*(devicePixelRatio||1)){drag=best;cv.setPointerCapture(e.pointerId);draw();}};
cv.onpointermove=e=>{if(drag<0)return;const p=pos(e);cur.quad[drag]={x:Math.max(0,Math.min(cur.w,p.x/scale)),y:Math.max(0,Math.min(cur.h,p.y/scale))};draw();};
cv.onpointerup=cv.onpointercancel=()=>{drag=-1;draw();};
$('eRetake').onclick=()=>openCam();
async function crop(){msg('Straightening the page ...');prog(0.5);
 const q=cur.quad.map(p=>p.x.toFixed(1)+','+p.y.toFixed(1)).join(',');
 try{const r=await api('/api/warp?q='+q,{method:'POST',body:cur.orig});if(!r.ok)throw 0;
  cur.result=await r.blob();cur.url=URL.createObjectURL(cur.result);$('rimg').src=cur.url;prog(null);msg('');show('review');
 }catch(e){prog(null);msg('Could not crop. Try again.',true);}}
$('eOk').onclick=crop;
$('rAdj').onclick=()=>startEdit();
$('rUse').onclick=()=>{if(retake>=0)pages[retake]=cur;else pages.push(cur);retake=-1;cur=null;renderList();};

// ---------- page list
function renderList(){show('list');$('lt').textContent=pages.length+' page'+(pages.length==1?'':'s');const box=$('pages');box.innerHTML='';
 pages.forEach((p,i)=>{const d=document.createElement('div');d.className='pg';
  d.innerHTML='<img src="'+p.url+'"><div class="n">Page '+(i+1)+'</div>';
  const mk=(t,fn,c)=>{const b=document.createElement('button');b.className='btn '+(c||'');b.textContent=t;b.onclick=ev=>{ev.stopPropagation();fn();};d.appendChild(b);};
  mk('▲',()=>{if(i>0){[pages[i-1],pages[i]]=[pages[i],pages[i-1]];renderList();}});
  mk('▼',()=>{if(i<pages.length-1){[pages[i+1],pages[i]]=[pages[i],pages[i+1]];renderList();}});
  mk('Redo',()=>{retake=i;openCam();});
  mk('✕',()=>{pages.splice(i,1);pages.length?renderList():home();},'red');
  d.querySelector('img').onclick=()=>{$('vimg').src=p.url;$('viewer').classList.remove('hide');};
  box.appendChild(d);});
 $('lSend').textContent='Send '+pages.length+' page'+(pages.length==1?'':'s')+' to desktop';msg('');}
$('viewer').onclick=()=>$('viewer').classList.add('hide');
function home(){show('home');msg('');}
$('lHome').onclick=home;$('lAdd').onclick=()=>{retake=-1;openCam();};
$('lSend').onclick=async()=>{if(busy||!pages.length)return;busy=true;$('lSend').disabled=true;
 const batch=Date.now().toString(36)+Math.random().toString(36).slice(2,6),n=pages.length;
 try{for(let i=0;i<n;i++){msg('Sending page '+(i+1)+' of '+n+' ...');prog(i/n);
   const r=await api('/api/upload?kind=scan&batch='+batch+'&i='+i+'&n='+n+'&name=page'+(i+1)+'.jpg',{method:'POST',body:pages[i].result});
   const j=await r.json();if(!j.ok)throw new Error(j.error);}
  prog(null);pages=[];home();msg('✓ '+n+' page'+(n==1?'':'s')+' sent to the computer');
 }catch(e){prog(null);msg('Sending failed. Check the Wi-Fi and try again (scan the QR code again if needed).',true);}
 busy=false;$('lSend').disabled=false;};
</script></body></html>"""
