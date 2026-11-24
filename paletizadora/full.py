import base64, cv2, numpy as np

def parse_pack(pack):
 if pack.get('format')!='smc-orb32-v1' or not 1<=len(pack['regions'])<=30:raise ValueError('Formato de referencias incorrecto')
 refs=[]
 for r in pack['regions']:
  a=np.frombuffer(base64.b64decode(r['data'],validate=True),np.uint8).reshape(-1,36)
  if not 12<=len(a)<=600:raise ValueError('Número de puntos incorrecto')
  w,h=int(r['width']),int(r['height'])
  if not 32<=min(w,h)<=max(w,h)<=1000:raise ValueError('Tamaño incorrecto')
  xy=a[:,:4].copy().view('<u2').reshape(-1,2).astype(np.float32)/65535*np.array([w,h],np.float32)
  refs.append((str(r['name'])[:80],xy,a[:,4:].copy(),w,h))
 return refs

def detect_full(im,refs):
 h,w=im.shape;orb=cv2.ORB_create(nfeatures=3200,fastThreshold=9,edgeThreshold=12,patchSize=31);k,d=orb.detectAndCompute(im,None)
 diag=dict(features=len(k),matches=0,inliers=0,geometric=0);out=[]
 if d is None or len(d)<2:return out,diag
 scene=np.float32([p.pt for p in k]);bf=cv2.BFMatcher(cv2.NORM_HAMMING)
 for name,xy,rd,rw,rh in refs:
  pairs=bf.knnMatch(rd,d,k=2);unique={}
  for pair in pairs:
   if len(pair)!=2:continue
   m,n=pair
   if m.distance<.80*n.distance and m.distance<=65 and (m.trainIdx not in unique or unique[m.trainIdx].distance>m.distance):unique[m.trainIdx]=m
  good=list(unique.values());diag['matches']=max(diag['matches'],len(good))
  if len(good)<8:continue
  src=np.float32([xy[m.queryIdx] for m in good]);dst=np.float32([scene[m.trainIdx] for m in good]);corners=np.float32([[[0,0],[rw,0],[rw,rh],[0,rh]]]);best=None
  for kind in ('affine','homography'):
   if kind=='affine':
    M,mask=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=5,maxIters=1500,confidence=.995)
    H=None if M is None else np.vstack([M,[0,0,1]])
   else:H,mask=cv2.findHomography(src,dst,cv2.RANSAC,4.5,maxIters=1500,confidence=.995)
   if H is None or mask is None or not np.isfinite(H).all():continue
   n=int(mask.sum());diag['geometric']=max(diag['geometric'],n)
   if n<(8 if kind=='affine' else 10) or n/len(good)<.32:continue
   a=src[mask.ravel()>0];b=dst[mask.ravel()>0]
   if cv2.contourArea(cv2.convexHull(a))<.018*rw*rh or cv2.contourArea(cv2.convexHull(b))<30:continue
   q=cv2.perspectiveTransform(corners,H)[0];c=cv2.perspectiveTransform(np.float32([[[rw/2,rh/2]]]),H)[0,0]
   if not np.isfinite(q).all() or not np.isfinite(c).all() or not cv2.isContourConvex(q):continue
   area=abs(cv2.contourArea(q));edges=np.linalg.norm(q-np.roll(q,1,axis=0),axis=1)
   if not .0005*w*h<area<1.8*w*h or min(edges)<5 or max(edges)/min(edges)>20 or not (0<=c[0]<w and 0<=c[1]<h):continue
   if cv2.pointPolygonTest(cv2.convexHull(a),(float(rw/2),float(rh/2)),True)<-.30*max(rw,rh):continue
   if best is None or n>best[0]:best=(n,c,q)
  if best is not None:
   n,c,q=best;diag['inliers']=max(diag['inliers'],n)
   out.append(dict(name=name,center=[float(c[0]/w),float(c[1]/h)],matches=n))
 keep=[]
 for o in sorted(out,key=lambda x:-x['matches']):
  if not any(o['name']==p['name'] and np.linalg.norm(np.array(o['center'])-p['center'])<.09 for p in keep):keep.append(o)
 return keep,diag
