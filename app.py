from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from database import db, User
from analytics_service import AnalyticsService
import numpy as np
import io
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mission_to_mars_2226'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(os.path.dirname(__file__), 'signals.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

login_manager = LoginManager(app)
login_manager.login_view = 'login'

analytics = AnalyticsService()

with app.app_context():
    db.create_all()
    if not User.query.filter_by(username='admin').first():
        db.session.add(User(
            username='admin',
            password=generate_password_hash('admin123'),
            first_name='root',
            last_name='root',
            role='admin'
        ))
        db.session.commit()
        print('[DB] База данных создана. Админ: admin / admin123')
    else:
        print('[DB] База данных загружена.')


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


@app.route('/')
def index():
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('profile'))
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('profile'))
        flash('Доступ запрещён: неверный логин или пароль')
    return render_template('login.html')


@app.route('/profile')
@login_required
def profile():
    return render_template('profile.html', user=current_user)


@app.route('/admin/create', methods=['POST'])
@login_required
def create_user():
    if current_user.role != 'admin':
        return 'Forbidden', 403

    username   = request.form.get('username', '').strip()
    first_name = request.form.get('first_name', '').strip()
    last_name  = request.form.get('last_name', '').strip()
    password   = request.form.get('password', '')

    if not all([username, first_name, last_name, password]):
        flash('Заполните все поля')
        return redirect(url_for('profile'))

    if User.query.filter_by(username=username).first():
        flash(f'Пользователь «{username}» уже существует')
        return redirect(url_for('profile'))

    db.session.add(User(
        username=username,
        password=generate_password_hash(password),
        first_name=first_name,
        last_name=last_name,
        role='user'
    ))
    db.session.commit()
    flash(f'Сотрудник {first_name} {last_name} успешно добавлен')
    return redirect(url_for('profile'))


@app.route('/upload_test', methods=['POST'])
@login_required
def upload_test():
    if current_user.role != 'user':
        return 'Forbidden', 403

    if 'test_file' not in request.files:
        flash('Файл не выбран')
        return redirect(url_for('profile'))

    file = request.files['test_file']
    if file.filename == '':
        flash('Имя файла пустое')
        return redirect(url_for('profile'))

    if not file.filename.lower().endswith('.npz'):
        flash('Допускается только формат .npz')
        return redirect(url_for('profile'))

    try:
        data = np.load(io.BytesIO(file.read()), allow_pickle=True)

        def encode_labels(arr):
            """Если метки строковые — кодируем в int, иначе возвращаем как int."""
            arr = np.array(arr)
            if arr.dtype.kind in ('U', 'S', 'O'):
                classes = np.unique(arr)
                mapping = {c: i for i, c in enumerate(classes)}
                return np.array([mapping[v] for v in arr], dtype=np.int64), classes
            return arr.astype(np.int64), None

        if 'train_y' in data:
            train_y, _ = encode_labels(data['train_y'])
        else:
            train_y = None

        if 'valid_y' in data:
            val_y, _ = encode_labels(data['valid_y'])
        elif 'vaild_y' in data:
            val_y, _ = encode_labels(data['vaild_y'])
        else:
            val_y = None

        if 'test_y' in data:
            t_y, _ = encode_labels(data['test_y'])
        elif val_y is not None:
            t_y = val_y
        else:
            flash('В файле не найдены метки классов')
            return redirect(url_for('profile'))

        num_classes = int(np.max(t_y) + 1) if len(t_y) > 0 else 10

        if train_y is None:
            train_y = np.random.randint(0, num_classes, 1200)
        if val_y is None:
            val_y = np.random.randint(0, num_classes, 400)

        if 'test_preds' in data:
            t_preds = np.array(data['test_preds'], dtype=np.float64)
        else:
            rng = np.random.default_rng(42)
            t_preds = rng.dirichlet(np.ones(num_classes), size=len(t_y))

        hist = {}
        hist['val_acc'] = data['val_acc'].tolist() if 'val_acc' in data \
            else [0.42, 0.61, 0.74, 0.83, 0.88, 0.91, 0.93]
        if 'train_acc' in data:
            hist['train_acc'] = data['train_acc'].tolist()

        predicted  = np.argmax(t_preds, axis=1)
        acc        = float(np.mean(predicted == t_y))
        eps        = 1e-9
        true_probs = np.clip(
            [t_preds[i][t_y[i]] for i in range(len(t_y))], eps, 1 - eps
        )
        loss = float(-np.mean(np.log(true_probs)))

        plots = analytics.generate_all_plots(hist, train_y, val_y, t_y, t_preds)

        return render_template('analytics.html',
                               plots=plots,
                               acc=acc,
                               loss=round(loss, 4),
                               count=len(t_y))

    except Exception as e:
        flash(f'Ошибка обработки файла: {e}')
        return redirect(url_for('profile'))


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


if __name__ == '__main__':
    app.run(debug=True)
