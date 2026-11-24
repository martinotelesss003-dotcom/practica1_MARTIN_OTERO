import os, json, base64, threading
from pathlib import Path
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
P=Path(__file__).parent
app=Flask(__name__)
app.config['MAX_CONTENT_LENGTH']=2500000
cv2.setNumThreads(1)
refs={n:np.frombuffer(base64.b64decode(s),np.uint8).reshape(-1,10) for n,s in json.loads((P/'refs.json').read_text()).items()}
annotations={
'085935':[('Bloque de electroválvulas',[.39,.32,.65,.93])],
'085940':[('Toma de corriente',[.36,.26,.67,.90])],
'085956':[('Controlador SMC',[.27,.26,.44,.55]),('Controlador SMC',[.44,.25,.68,.55]),('Interfaz J2',[.60,.33,.92,.49])],
'090005':[('Borneras X1',[.11,.39,.92,.57])],
'090009':[('Filtro F3',[.26,.39,.49,.60]),('Regulador R3',[.52,.25,.74,.47]),('Manómetro M3',[.55,.46,.75,.55]),('Válvula VE',[.19,.47,.30,.60])],
'090012':[('Bandeja de paletización',[.21,.32,.82,.45])],
'090014':[('Baliza BR',[.41,.45,.61,.64])],
'090020':[('Eje lineal horizontal',[.11,.36,.46,.52]),('Actuador vertical A',[.46,.24,.52,.56]),('Cadena portacables',[.45,.19,.64,.36]),('Eje transversal',[.51,.49,.74,.74])],
'090026':[('PLC Siemens S7-1200',[.22,.19,.59,.79]),('Módulo SM 1223',[.59,.25,.70,.79])],
'090032':[('Fuente Omron 24 V',[.38,.26,.51,.90]),('Borneras X1',[.76,.17,.99,.57])],
'090042':[('Paletizadora SMC',[.24,.14,.75,.85]),('Parada de emergencia',[.56,.64,.60,.72]),('Botonera',[.60,.65,.73,.74]),('Bandeja de paletización',[.33,.56,.45,.64]),('Baliza BR',[.27,.58,.32,.77])]}
orb=cv2.ORB_create(nfeatures=2500,fastThreshold=10)
bf=cv2.BFMatcher(cv2.NORM_HAMMING)
lock=threading.Lock()
@app.get('/')
def home():return send_from_directory(P,'index.html')
@app.get('/api/health')
def health():return jsonify(backend=True,references=len(refs),mode='ORB',yolo=False)
@app.post('/api/detect')
def detect():
 try:
  data=request.get_json(); raw=base64.b64decode(data['image'].split(',')[-1],validate=True)
  im=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_GRAYSCALE)
  if im is None:return jsonify(error='Imagen inválida'),400
  h,w=im.shape;s=960/max(w,h);im=cv2.resize(im,(round(w*s),round(h*s)));h,w=im.shape
  with lock:
   k,d=orb.detectAndCompute(im,None)
   candidates=[]
   if d is not None and len(d)>1:
    for n,a in refs.items():
     good=[m for pair in bf.knnMatch(a[:,2:].copy(),d[:,:8].copy(),k=2) if len(pair)==2 for m,b in [pair] if m.distance<.76*b.distance and m.distance<=19]
     if len(good)<8:continue
     src=np.float32([a[m.queryIdx,:2]/255*960 for m in good]);dst=np.float32([k[m.trainIdx].pt for m in good])
     H,mask=cv2.findHomography(src,dst,cv2.RANSAC,5)
     if H is None or mask is None or mask.sum()<7 or mask.sum()/len(good)<.55:continue
     candidates.append((int(mask.sum()),n,H))
  out=[]
  if candidates:
   score,n,H=max(candidates,key=lambda x:x[0])
   for label,(x,y,X,Y) in annotations[n]:
    q=cv2.perspectiveTransform(np.float32([[[x*960,y*960],[X*960,y*960],[X*960,Y*960],[x*960,Y*960]]]),H)[0]
    if not np.isfinite(q).all() or not cv2.isContourConvex(q):continue
    if abs(cv2.contourArea(q))<.0005*w*h or abs(cv2.contourArea(q))>1.3*w*h:continue
    c=cv2.perspectiveTransform(np.float32([[[(x+X)*480,(y+Y)*480]]]),H)[0,0]
    if not (0<=c[0]<w and 0<=c[1]<h):continue
    prompt=str(data.get('prompt','')).strip().lower()
    if prompt and not any(t.strip() in label.lower() for t in prompt.split(',') if t.strip()):continue
    out.append(dict(name=label,center=[float(c[0]/w),float(c[1]/h)],matches=score))
  return jsonify(detections=out,mode='ORB',defects_available=False)
 except Exception:
  app.logger.exception('Detection failed');return jsonify(error='No se pudo analizar el fotograma'),400
if __name__=='__main__':app.run(host='0.0.0.0',port=int(os.environ.get('PORT',8000)))
