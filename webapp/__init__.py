"""Web application package for KIQFL demo UI."""
import os

from flask import Flask


def create_app():
    pkg = os.path.dirname(os.path.abspath(__file__))
    app = Flask(
        __name__,
        template_folder=os.path.join(pkg, 'templates'),
        static_folder=os.path.join(pkg, 'static'),
    )
    app.secret_key = os.environ.get('KIQFL_SECRET_KEY', 'kiqfl_dev_secret')

    # Register routes on the blueprint before attaching it to the app
    from webapp.routes import api, pages  # noqa: F401
    from webapp.routes import bp
    app.register_blueprint(bp)
    return app
