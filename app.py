# app.py
# ══════════════════════════════════════════════════════════════
# AI-Based Skin Disease Detection and Awareness System
# Backend : Flask + SQLite
# Model   : MobileNetV2 (Transfer Learning)
# Run     : python app.py
# ══════════════════════════════════════════════════════════════

from flask import (Flask, request, jsonify, render_template,
                   redirect, url_for, session, send_from_directory)
import sqlite3, os, uuid, hashlib, numpy as np, json
from datetime import datetime
from PIL import Image
from werkzeug.utils import secure_filename 

import tensorflow as tf
from tensorflow.keras.models import load_model   

app = Flask(__name__)
app.secret_key = 'dermai_skin_2025_secret'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
ALLOWED_EXT = {'png', 'jpg', 'jpeg', 'webp'}
IMG_SIZE    = 128

# ── Disease Data ───────────────────────────────────────────────
CLASS_NAMES = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']

DISEASE_INFO = {
    'nv': {
        'name'    : 'Melanocytic Nevi (Moles)',
        'severity': 'Non-Severe',
        'type'    : 'Benign',
        'recommendation': 'Monitor regularly for changes in size, shape, or color. Visit a dermatologist if it changes rapidly.',
        'precautions': [
            'Avoid excessive sun exposure',
            'Apply sunscreen SPF 30+ daily',
            'Do monthly self-skin checks',
            'See a doctor if mole changes color or shape'
        ]
    },
    'mel': {
        'name'    : 'Melanoma (Skin Cancer)',
        'severity': 'Severe',
        'type'    : 'Malignant',
        'recommendation': 'Seek immediate medical attention. Melanoma is the most dangerous skin cancer — early treatment is critical.',
        'precautions': [
            'Consult an oncologist immediately',
            'Avoid all sun exposure on affected area',
            'Do not delay medical treatment',
            'Follow ABCDE rule: Asymmetry, Border, Color, Diameter, Evolving'
        ]
    },
    'bkl': {
        'name'    : 'Benign Keratosis',
        'severity': 'Non-Severe',
        'type'    : 'Benign',
        'recommendation': 'Usually harmless. Consult a dermatologist if it becomes itchy, painful, or changes appearance.',
        'precautions': [
            'Keep skin moisturized',
            'Avoid scratching or picking the lesion',
            'Use gentle skin care products',
            'Annual dermatologist check-up recommended'
        ]
    },
    'bcc': {
        'name'    : 'Basal Cell Carcinoma',
        'severity': 'Severe',
        'type'    : 'Malignant',
        'recommendation': 'Visit a dermatologist soon. BCC rarely spreads but requires prompt treatment to prevent growth.',
        'precautions': [
            'Consult a skin specialist immediately',
            'Use broad-spectrum sunscreen daily',
            'Avoid direct sun exposure between 10am–4pm',
            'Wear protective clothing outdoors'
        ]
    },
    'akiec': {
        'name'    : 'Actinic Keratosis',
        'severity': 'Severe',
        'type'    : 'Pre-Cancerous',
        'recommendation': 'Consult a dermatologist. Actinic Keratosis is pre-cancerous and can develop into cancer if untreated.',
        'precautions': [
            'Seek dermatologist evaluation promptly',
            'Apply SPF 50+ sunscreen every day',
            'Avoid tanning beds completely',
            'Consider cryotherapy treatment'
        ]
    },
    'vasc': {
        'name'    : 'Vascular Lesion',
        'severity': 'Non-Severe',
        'type'    : 'Benign',
        'recommendation': 'Generally harmless. See a dermatologist if it bleeds, grows, or causes discomfort.',
        'precautions': [
            'Avoid trauma or pressure on the lesion',
            'Do not attempt to remove at home',
            'Keep the area clean and dry',
            'Consult a doctor for cosmetic concerns'
        ]
    },
    'df': {
        'name'    : 'Dermatofibroma',
        'severity': 'Non-Severe',
        'type'    : 'Benign',
        'recommendation': 'Benign and harmless. No treatment needed unless it causes pain or cosmetic concern.',
        'precautions': [
            'No immediate action required',
            'Avoid irritating the bump',
            'See a doctor if it grows rapidly',
            'Surgical removal available if desired'
        ]
    }
}

# ── Load CNN Model ─────────────────────────────────────────────
print("\n🧠 Loading CNN Model (MobileNetV2)...")
MODEL = None
MODEL_ACCURACY = None
if os.path.exists('skin_model.h5'):
    try:
        MODEL = load_model('skin_model.h5')
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Model load error: {e}")
else:
    print("⚠️  skin_model.h5 not found — run train.py first!")

# Load accuracy
if os.path.exists('model_info.json'):
    with open('model_info.json') as f:
        _info = json.load(f)
        MODEL_ACCURACY = _info.get('accuracy', None)
    print(f"✅ Model accuracy: {MODEL_ACCURACY}%")

# ── Database ───────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect('dermai.db')
    conn.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL, email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL, created TEXT)''')
    conn.execute('''CREATE TABLE IF NOT EXISTS scans (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        scan_id TEXT UNIQUE, user_id INTEGER,
        filename TEXT, disease_key TEXT, disease_name TEXT,
        severity TEXT, disease_type TEXT,
        confidence REAL, recommendation TEXT,
        created TEXT,
        FOREIGN KEY(user_id) REFERENCES users(id))''')
    conn.commit()
    conn.close()

def hash_pw(p):
    return hashlib.sha256(p.encode()).hexdigest()

def get_user(email):
    conn = sqlite3.connect('dermai.db')
    conn.row_factory = sqlite3.Row
    r = conn.execute('SELECT * FROM users WHERE email=?', (email,)).fetchone()
    conn.close()
    return dict(r) if r else None

def create_user(name, email, password):
    try:
        conn = sqlite3.connect('dermai.db')
        conn.execute('INSERT INTO users(name,email,password,created) VALUES(?,?,?,?)',
            (name, email, hash_pw(password), datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        conn.commit(); conn.close(); return True
    except sqlite3.IntegrityError:
        return False

def save_scan(scan_id, user_id, filename, key, info, confidence):
    conn = sqlite3.connect('dermai.db')
    conn.execute('''INSERT INTO scans
        (scan_id,user_id,filename,disease_key,disease_name,severity,
         disease_type,confidence,recommendation,created)
        VALUES(?,?,?,?,?,?,?,?,?,?)''',
        (scan_id, user_id, filename, key,
         info['name'], info['severity'], info['type'],
         confidence, info['recommendation'],
         datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    conn.commit(); conn.close()

def get_scans(user_id):
    conn = sqlite3.connect('dermai.db')
    conn.row_factory = sqlite3.Row
    rows = conn.execute('SELECT * FROM scans WHERE user_id=? ORDER BY id DESC', (user_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_scan(scan_id, user_id):
    conn = sqlite3.connect('dermai.db')
    conn.row_factory = sqlite3.Row
    r = conn.execute('SELECT * FROM scans WHERE scan_id=? AND user_id=?', (scan_id, user_id)).fetchone()
    conn.close()
    return dict(r) if r else None

def del_scan(scan_id, user_id):
    conn = sqlite3.connect('dermai.db')
    r = conn.execute('SELECT filename FROM scans WHERE scan_id=? AND user_id=?', (scan_id, user_id)).fetchone()
    conn.execute('DELETE FROM scans WHERE scan_id=? AND user_id=?', (scan_id, user_id))
    conn.commit(); conn.close()
    if r:
        p = os.path.join(app.config['UPLOAD_FOLDER'], r[0])
        if os.path.exists(p): os.remove(p)

def allowed(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

def login_required(f):
    from functools import wraps
    @wraps(f)
    def dec(*a, **kw):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*a, **kw)
    return dec

# ── Prediction ─────────────────────────────────────────────────
def predict(image_path):
    img = Image.open(image_path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    arr = np.expand_dims(np.array(img, dtype='float32') / 255.0, 0)
    preds = MODEL.predict(arr, verbose=0)[0]
    top_idx   = int(np.argmax(preds))
    key       = CLASS_NAMES[top_idx]
    confidence= round(float(preds[top_idx]) * 100, 1)
    top3 = [{'name': DISEASE_INFO[CLASS_NAMES[i]]['name'],
              'confidence': round(float(preds[i]) * 100, 1)}
             for i in preds.argsort()[-3:][::-1]]
    return key, confidence, top3

# ── Auth ───────────────────────────────────────────────────────
@app.route('/signup', methods=['GET','POST'])
def signup():
    if 'user_id' in session: return redirect(url_for('index'))
    if request.method == 'POST':
        name  = request.form.get('name','').strip()
        email = request.form.get('email','').strip().lower()
        pw    = request.form.get('password','')
        if not name or not email or not pw:
            return render_template('signup.html', error='All fields are required.')
        if len(pw) < 6:
            return render_template('signup.html', error='Password must be at least 6 characters.', name=name, email=email)
        if create_user(name, email, pw):
            u = get_user(email)
            session['user_id']   = u['id']
            session['user_name'] = u['name']
            return redirect(url_for('index'))
        return render_template('signup.html', error='Email already registered. Please log in.', name=name)
    return render_template('signup.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if 'user_id' in session: return redirect(url_for('index'))
    if request.method == 'POST':
        email = request.form.get('email','').strip().lower()
        pw    = request.form.get('password','')
        u     = get_user(email)
        if u and u['password'] == hash_pw(pw):
            session['user_id']   = u['id']
            session['user_name'] = u['name']
            return redirect(url_for('index'))
        return render_template('login.html', error='Invalid email or password.', email=email)
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# ── Pages ──────────────────────────────────────────────────────
@app.route('/')
@login_required
def index():
    return render_template('index.html',
        user_name=session.get('user_name'),
        model_ready=MODEL is not None,
        model_accuracy=MODEL_ACCURACY)

@app.route('/history')
@login_required
def history():
    scans = get_scans(session['user_id'])
    severe     = sum(1 for s in scans if s['severity'] == 'Severe')
    non_severe = sum(1 for s in scans if s['severity'] == 'Non-Severe')
    return render_template('history.html',
        scans=scans, user_name=session.get('user_name'),
        severe=severe, non_severe=non_severe)

@app.route('/scan/<scan_id>')
@login_required
def scan_detail(scan_id):
    scan = get_scan(scan_id, session['user_id'])
    if not scan: return redirect(url_for('history'))
    info = DISEASE_INFO.get(scan['disease_key'], {})
    return render_template('detail.html',
        scan=scan, info=info,
        user_name=session.get('user_name'))

# ── API ────────────────────────────────────────────────────────
@app.route('/api/analyze', methods=['POST'])
@login_required
def analyze():
    if MODEL is None:
        return jsonify({'error': 'Model not loaded. Run train.py first!'}), 500
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    if not file or not allowed(file.filename):
        return jsonify({'error': 'Invalid file. Use JPG, PNG or WEBP.'}), 400

    ext      = file.filename.rsplit('.', 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        key, confidence, top3 = predict(filepath)
        info    = DISEASE_INFO[key]
        scan_id = uuid.uuid4().hex
        save_scan(scan_id, session['user_id'], filename, key, info, confidence)

        return jsonify({
            'success'       : True,
            'scan_id'       : scan_id,
            'disease_name'  : info['name'],
            'severity'      : info['severity'],
            'disease_type'  : info['type'],
            'confidence'    : confidence,
            'recommendation': info['recommendation'],
            'precautions'   : info['precautions'],
            'top3'          : top3,
            'image_url'     : f'/static/uploads/{filename}'
        })
    except Exception as e:
        if os.path.exists(filepath): os.remove(filepath)
        return jsonify({'error': str(e)}), 500

@app.route('/api/delete/<scan_id>', methods=['DELETE'])
@login_required
def delete(scan_id):
    del_scan(scan_id, session['user_id'])
    return jsonify({'success': True})

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    init_db()
    print("\n✅ DermAI running → http://127.0.0.1:5000\n")
    app.run(debug=True)
