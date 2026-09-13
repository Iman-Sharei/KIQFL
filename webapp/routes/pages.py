from flask import render_template
from webapp.routes import bp

@bp.route('/')
def home():
    return render_template('index.html')

@bp.route('/about')
def about():
    return render_template('about.html')

@bp.route('/contact')
def contact():
    return render_template('contact.html')

@bp.route('/simulate')
def simulate():
    return render_template('simulate.html')

@bp.route('/simulate_own')
def simulate_own():
    return render_template('simulate_own.html')

@bp.route('/choose')
def choose():
    return render_template('choose.html')

@bp.route('/compare')
def compare():
    return render_template('compare.html')
