from flask import *
from flask_compress import Compress
import requests
from PIL import Image
import io
import os
import time
import random
import numpy as np
import base64
import shutil

app = Flask(__name__)
Compress(app)
app.secret_key = __SECRET_KEY_HERE__

app.debug = True

PATHNAME_IN = 'bootstrap_img_fail'
PATHNAME_OUT = 'bootstrap_img_fail_tagged'

@app.template_filter('imgurl')
def filter_imgurl(px):
    assert len(px.shape) in [2,3]
    if len(px.shape)==2: # diff pic
        #print('imgurl diff')
        px = np.abs(px)
        ma = np.max(px)
        px = (px.astype(np.double)/ma*255).astype(np.uint8)
        im = Image.fromarray(px, 'L')
    else: # orig pic
        #print('imgurl orig')
        im = Image.fromarray(px.astype(np.uint8), 'RGB')
    
    bio = io.BytesIO()
    im.save(bio, 'PNG')
    return 'data:image/png;base64,'+base64.b64encode(bio.getvalue()).decode()

@app.before_request
def setup_session():
    if 'count' not in session:
        session['count'] = 0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/captcha_frame')
def captcha_frame():
    if 'serial' not in session or not os.path.isfile(session['serial']):
        session['serial'] = os.listdir(PATHNAME_IN)[0].partition('.')[0]
    
    with open('%s/%s.gif'%(PATHNAME_IN, session['serial']), 'rb') as f:
        rawim = f.read()
    assert rawim.startswith(b'GIF89a'), 'bad captcha response'

    im = Image.open(io.BytesIO(rawim))

    frame = [None]*16
    for f in range(16):
        im.seek(f)
        frame[f] = np.array(im.convert('RGB')).astype(np.int)

    diff = [np.sum(frame[x+4]-frame[x],2) for x in [3,7,11]]

    return render_template('captcha_frame.html', frame=frame, diff=diff)

@app.route('/submit_captcha', methods=['POST'])
def submit_captcha():
    assert 'serial' in session
    assert '/' not in session['serial']
    serial = session['serial']
    captcha = request.form['captcha']

    if captcha=='_':
        os.remove('%s/%s.gif'%(PATHNAME_IN, serial))
        flash('已删除')
        del session['serial']
        return redirect(url_for('index'))

    if not captcha.isalnum() or len(captcha)!=4:
        flash('格式错误')
        return redirect(url_for('index'))

    flash('填写完成')
    session['count']+=1
    serial_stem = serial.partition('=')[2]

    if serial.partition('=')[0]==captcha:
        flash('与原标签相同')
        return redirect(url_for('index'))

    shutil.move('%s/%s.gif'%(PATHNAME_IN, serial), '%s/%s=%s.gif'%(PATHNAME_OUT, captcha, serial_stem))

    del session['serial']
    return redirect(url_for('index'))

app.run('0.0.0.0', 10191)