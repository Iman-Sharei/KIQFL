from webapp.app import app
import os

app.run(
    host='127.0.0.1',
    port=int(os.environ.get('PORT', '5000')),
    debug=os.environ.get('FLASK_DEBUG', '0') == '1',
    threaded=True,
)
