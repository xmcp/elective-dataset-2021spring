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

adapter = requests.adapters.HTTPAdapter(pool_connections=3, pool_maxsize=3, pool_block=True)
s = requests.Session()
s.mount('http://elective.pku.edu.cn', adapter)
s.mount('https://elective.pku.edu.cn', adapter)

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
    if 'elective_cookie' not in session:
        session['elective_cookie'] = None
    if 'elective_xh' not in session:
        session['elective_xh'] = None
    if 'count' not in session:
        session['count'] = 0

@app.route('/')
def index():
    if not session['elective_cookie'] or not session['elective_xh']:
        return render_template('setup.html')
    else:
        return render_template('index.html')

@app.route('/captcha_frame')
def captcha_frame():
    if 'serial' in session:
        assert '/' not in session['serial']
        if os.path.isfile('img_serial/%s.gif'%session['serial']):
            os.remove('img_serial/%s.gif'%session['serial'])

    res = s.get(
        'https://elective.pku.edu.cn/elective2008/DrawServlet?Rand=114514',
        headers={
            'referer': 'https://elective.pku.edu.cn/elective2008/edu/pku/stu/elective/controller/supplement/SupplyCancel.do',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.190 Safari/537.36',
            'cookie': session['elective_cookie'],
        },
        timeout=(3,3),
    )
    res.raise_for_status()
    rawim = res.content
    assert rawim.startswith(b'GIF89a'), 'bad captcha response'

    im = Image.open(io.BytesIO(rawim))
    serial = '%d-%d'%(1000*time.time(), random.random()*1000)
    with open('img_serial/%s.gif'%serial, 'wb') as f:
        f.write(rawim)
    session['serial'] = serial

    frame = [None]*16
    for f in range(16):
        im.seek(f)
        frame[f] = np.array(im.convert('RGB')).astype(np.int)

    diff = [np.sum(frame[x+4]-frame[x],2) for x in [3,7,11]]

    return render_template('captcha_frame.html', frame=frame, diff=diff)

@app.route('/update_cookie', methods=['POST'])
def update_cookie():
    session['elective_cookie'] = request.form['cookie']
    session['elective_xh'] = request.form['xh']
    return redirect(url_for('index'))

@app.route('/submit_captcha', methods=['POST'])
def submit_captcha():
    assert 'serial' in session
    assert '/' not in session['serial']
    serial = session['serial']
    captcha = request.form['captcha']
    if not captcha.isalnum() or len(captcha)!=4:
        flash('格式错误')
        return redirect(url_for('index'))

    res = s.post(
        'https://elective.pku.edu.cn/elective2008/edu/pku/stu/elective/controller/supplement/validate.do',
        headers={
            'referer': 'https://elective.pku.edu.cn/elective2008/edu/pku/stu/elective/controller/supplement/SupplyCancel.do',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.190 Safari/537.36',
            'cookie': session['elective_cookie'],
        },
        data={
            'xh': session['elective_xh'],
            'validCode': captcha,
        },
        timeout=(3,3),
    )
    res.raise_for_status()
    json = res.json()

    if json['valid']!='2':
        flash('验证码错误')
    else:
        flash('填写正确')
        session['count']+=1
        shutil.move('img_serial/%s.gif'%serial, 'img_correct/%s=%s.gif'%(captcha, serial))

    return redirect(url_for('index'))

@app.route('/logout')
def logout():
    del session['elective_xh']
    del session['elective_cookie']
    return redirect(url_for('index'))

@app.route('/opensource')
def opensource():
    with open(__file__, encoding='utf-8') as f:
        content = f.read().replace(repr(app.secret_key), '***')
    
    resp = Response(content)
    resp.headers['content-type'] = 'text/plain; charset=utf-8'
    return resp

app.run('0.0.0.0', 10190)