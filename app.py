from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)

# Модели базы данных
class Product(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    price = db.Column(db.Float, nullable=False)
    sales = db.Column(db.Integer, default=0)

class Order(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    customer_name = db.Column(db.String(100), nullable=False)
    product_id = db.Column(db.Integer, db.ForeignKey('product.id'), nullable=False)
    quantity = db.Column(db.Integer, nullable=False)

# Маршруты
@app.route('/')
def index():
    products = Product.query.all()
    return render_template('index.html', products=products)

@app.route('/admin', methods=['GET', 'POST'])
def admin():
    if request.method == 'POST':
        name = request.form['name']
        price = float(request.form['price'])
        new_product = Product(name=name, price=price)
        db.session.add(new_product)
        db.session.commit()
    products = Product.query.all()
    return render_template('admin.html', products=products)

@app.route('/analytics')
def analytics():
    products = Product.query.all()
    return render_template('analytics.html', products=products)

@app.route('/recommendations')
def recommendations():
    top_products = Product.query.order_by(Product.sales.desc()).limit(5).all()
    return render_template('recommendations.html', products=top_products)

# KNN-рекомендации
def train_knn_model():
    products = Product.query.all()
    data = []
    for product in products:
        data.append([product.id, product.price, product.sales])
    df = pd.DataFrame(data, columns=['id', 'price', 'sales'])
    model = NearestNeighbors(n_neighbors=3).fit(df[['price', 'sales']])
    return model

knn_model = train_knn_model()

@app.route('/recommend-by-product/<int:product_id>')
def recommend_by_product(product_id):
    product = Product.query.get(product_id)
    product_features = [[product.price, product.sales]]
    distances, indices = knn_model.kneighbors(product_features)
    recommended_products = []
    for idx in indices[0]:
        recommended_id = df.iloc[idx]['id']
        recommended_products.append(Product.query.get(recommended_id))
    return render_template('recommendations.html', products=recommended_products)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)