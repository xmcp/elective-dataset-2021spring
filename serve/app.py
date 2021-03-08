from flask import *
import base64
import captcha
import time
import io
from PIL import Image

app = Flask(__name__)

def getlist():
    with open('list.txt') as f:
        li = f.read().splitlines()
    return set(li)

def log(uid):
    with open('log.txt', 'a') as f:
        f.write('%s|%s|%s\n'%(time.time(), time.ctime(),uid))

@app.route('/fire', methods=['POST'])
def fire():
    im_b64 = request.form['captcha_b64']
    uid = request.form['uid']
    assert len(im_b64)<250000
    assert len(uid)==10

    names = getlist()
    if uid not in names:
        return jsonify({'error': 'Not supported'})

    log(uid)

    bio = io.BytesIO(base64.b64decode(im_b64.encode()))
    im = Image.open(bio)
    res = captcha.recognize(im)

    return jsonify({
        'error': None,
        'result': res,
    })

app.run('0.0.0.0', 10192)
