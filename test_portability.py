"""Standalone CPU-only tests; no Windows input/camera opened."""
from pathlib import Path
import importlib.util, tempfile, io, os
from unittest.mock import patch
module_file=Path(__file__).with_name('gesture_pointer_step19_v3b_portable.py')
spec=importlib.util.spec_from_file_location('hmi',module_file)
hmi=importlib.util.module_from_spec(spec)
spec.loader.exec_module(hmi)
assert hmi.ROOT==module_file.parent.resolve()
assert hmi.find_numpy_mlp_model()==module_file.parent/'models_v2'/'gesture_mlp_v2.npz'
weights,mean,std,clip,classes=hmi.load_numpy_mlp()
assert set(classes)=={'peace','one','stop','no_gesture'}
assert hmi.numpy_mlp_probs(__import__('numpy').zeros((42,),dtype='float32'),weights).shape == (4,)
assert hmi.camera_cli_options(['--camera-index','2','--camera-backend','msmf'])==(2,'msmf')
assert hmi.camera_cli_options([])==(None,'auto')
for args in (['--camera-index','abc'],['--camera-index','22'],['--camera-backend','fake']):
    try: hmi.camera_cli_options(args)
    except ValueError: pass
    else: raise AssertionError(f'invalid args accepted {args}')

class Capture:
    def __init__(self,success): self.success=success; self.released=False;self.count=0
    def isOpened(self):return self.success
    def set(self,*args):return True
    def read(self):self.count+=1;return (self.success,FakeFrame() if self.success else None)
    def release(self): self.released=True
class FakeFrame: size=256
class FakeCV:
    CAP_DSHOW=701;CAP_MSMF=702;CAP_ANY=703
    CAP_PROP_FRAME_WIDTH=1;CAP_PROP_FRAME_HEIGHT=2;CAP_PROP_FPS=3;CAP_PROP_BUFFERSIZE=4
    def __init__(self): self.captures=[]
    def VideoCapture(self, idx, api):
        cap=Capture((idx,api)==(1,self.CAP_MSMF));self.captures.append((idx,api,cap));return cap
cv=FakeCV();logs=[]
cap,idx,api=hmi.open_compatible_camera(cv,max_index=2,reporter=logs.append)
assert (idx,api)==(1,'MSMF')
assert all(c.released for i,a,c in cv.captures if c is not cap)
assert not cap.released;cap.release()
assert logs[-1]=='CAMERA SELECTED: index=1, backend=MSMF'
cv=FakeCV()
try:hmi.open_compatible_camera(cv,index=0,backend='dshow',reporter=lambda s:None)
except RuntimeError as exc:assert 'No working camera' in str(exc)
else:raise AssertionError('Camera 0 should fail')
assert len(cv.captures)==1 and cv.captures[0][2].released

# Verify the model will be discovered after moving the entire folder.
with tempfile.TemporaryDirectory() as t:
    dummy=Path(t)/'models_v2';dummy.mkdir();(dummy/'gesture_mlp.pth').write_bytes(b'dummy')
    with patch.object(hmi,'ROOT',Path(t)),patch.object(hmi,'BUNDLE_ROOT',Path(t)):
        assert hmi.find_mlp_model()==dummy/'gesture_mlp.pth'
# Missing hand model is installed in cache, not read-only app directory.
with tempfile.TemporaryDirectory() as t:
    mlp=Path(t)/'model.pth';dest=Path(t)/'cached'/'hand_landmarker.task'
    with patch.object(hmi,'resource_locations',return_value=([mlp],[dest])):
        with patch.object(hmi.urllib.request,'urlopen',return_value=io.BytesIO(b'mock-task')):
            assert hmi.find_hand_model(download=True)==dest
        assert dest.read_bytes()==b'mock-task'
print('PORTABILITY TESTS PASSED: model discovery, CWD-independent ROOT, camera fallback, manual flags, cache download')
