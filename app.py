# app.py
import os
from flask import (
    Flask, render_template, request, redirect, url_for, flash, jsonify, session
)
from flask_caching import Cache
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime, timedelta
from functools import lru_cache
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading
from typing import Dict, List, Any

from ml.train_model import load_model, predict_spoilage
from utils.data_loader import load_recipients
from services.redistribution import find_recipients
from services.routing import get_route

load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "devkey")

# Configure Flask-Caching with simple cache
cache = Cache(app, config={
    'CACHE_TYPE': 'simple',
    'CACHE_DEFAULT_TIMEOUT': 900  # 15 minutes
})

# Thread pool for async operations
executor = ThreadPoolExecutor(max_workers=4)

# Global cache for inventory data with TTL
class TTLCache:
    def __init__(self, ttl_seconds=60):
        self.cache = {}
        self.timestamps = {}
        self.ttl = ttl_seconds
        self.lock = threading.Lock()
    
    def get(self, key):
        with self.lock:
            if key in self.timestamps:
                if (datetime.now() - self.timestamps[key]).total_seconds() < self.ttl:
                    return self.cache.get(key)
                else:
                    del self.cache[key]
                    del self.timestamps[key]
            return None
    
    def set(self, key, value):
        with self.lock:
            self.cache[key] = value
            self.timestamps[key] = datetime.now()

# Initialize global caches
inventory_cache = TTLCache(ttl_seconds=60)  # 1 minute TTL
prediction_cache = TTLCache(ttl_seconds=900)  # 15 minutes TTL

# Load model & recipient data once
model = load_model()  
recipients = load_recipients("data/recipients.csv")


@app.route("/", methods=["GET"])
def home():
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = request.form.get("username")
        pwd  = request.form.get("password")
        if user and pwd:
            return redirect(url_for("dashboard"))
        flash("Username & password required", "error")
    return render_template("login.html")


@app.route("/inventory", methods=["GET", "POST"])
def inventory():
    if request.method == "POST":
        # Extract form inputs
        product     = request.form["product"]
        days_old    = int(request.form["days_old"])
        temperature = float(request.form["temperature"])
        humidity    = float(request.form["humidity"])
        location    = request.form["location"]

        # Predict spoilage probability
        prob = predict_spoilage(
            model, product, days_old, temperature, humidity
        )
        freshness = round((1 - prob) * 100, 1)

        # Store last item in session-like dict for dashboard
        item = {
            "product": product,
            "days_old": days_old,
            "temperature": temperature,
            "humidity": humidity,
            "location": location,
            "spoilage_prob": round(prob * 100, 1),
            "freshness": freshness
        }
        return render_template(
            "inventory.html",
            recent_item=item,
            show_result=True
        )

    return render_template("inventory.html")


import pandas as pd
from datetime import datetime, timedelta
from functools import lru_cache
import numpy as np

def get_emoji(product):
    emoji_map = {
        "Banana": "🍌",
        "Apple": "🍎",
        "Milk": "🥛",
        "Bread": "🍞",
        "Paneer": "🧀",
        "Spinach": "🥬",
        "Chicken": "🍗",
        "Ice Cream": "🍦"
    }
    return emoji_map.get(product, "🔸")

def get_cached_prediction(product, days_old, temperature, humidity, timestamp_15min):
    """Cache predictions based on 15-minute intervals"""
    # Create cache key
    cache_key = f"{product}:{days_old}:{temperature}:{humidity}:{timestamp_15min}"
    
    # Check cache
    cached_pred = prediction_cache.get(cache_key)
    if cached_pred is not None:
        return cached_pred
    
    # Calculate prediction
    pred = predict_spoilage(model, product, days_old, temperature, humidity)
    
    # Cache result
    prediction_cache.set(cache_key, pred)
    return pred

def calculate_risk_score(row, prob, current_date):
    """Calculate comprehensive risk score based on multiple factors"""
    
    # Base risk from ML model prediction
    risk_score = prob
    
    # Days until expiry factor (higher weight as expiry approaches)
    days_to_expiry = (row['expiry_date'] - current_date).days
    if days_to_expiry <= 2:  # Critical zone
        risk_score *= 1.5
    elif days_to_expiry <= 5:  # Warning zone
        risk_score *= 1.2
    
    # Storage condition factor
    if row['storage_type'] == 'Refrigerated':
        if row['temperature'] > 4:  # Above safe refrigeration temp
            risk_score *= 1.3
    elif row['storage_type'] == 'Frozen':
        if row['temperature'] > -15:  # Above safe freezing temp
            risk_score *= 1.4
    
    # High-risk category factor
    if row['category'] in ['Dairy', 'Meat']:
        risk_score *= 1.2
    
    # Climate factor for non-refrigerated items
    if row['storage_type'] == 'Ambient':
        if row['store_location'] in ['Chennai', 'Mumbai']:  # Hot/humid cities
            risk_score *= 1.1
    
    return min(risk_score, 1.0)  # Cap at 1.0

def load_inventory_data():
    """Load and preprocess inventory data with caching"""
    # Check cache first
    cached_df = inventory_cache.get('inventory')
    if cached_df is not None:
        return cached_df
    
    # Load and preprocess data
    try:
        df = pd.read_csv("data/mock_inventory.csv", dtype={
            'product': str,
            'category': str,
            'temperature': float,
            'humidity': float,
            'storage_type': str,
            'store_location': str,
            'stock_date': str,
            'expiry_date': str
        })
        
        # Bulk datetime conversions
        for col in ['stock_date', 'expiry_date']:
            df[col] = pd.to_datetime(df[col], format='%Y-%m-%d', errors='coerce')
        
        # Update cache
        inventory_cache.set('inventory', df)
        return df
        
    except Exception as e:
        print(f"Error loading inventory: {e}")
        return None

@app.route("/dashboard", methods=["GET"])
@cache.cached(timeout=60, query_string=True)
def dashboard():
    """Optimized dashboard route with caching and async processing"""
    try:
        df = load_inventory_data()
        if df is None:
            raise ValueError("Failed to load inventory data")
        
        current_date = datetime.now()
        
        # Vectorized calculations
        df['days_old'] = (current_date - df['stock_date']).dt.days
        df['days_to_expiry'] = (df['expiry_date'] - current_date).dt.days
        
        # Calculate risk scores in parallel
        def calculate_batch_risks(batch_df):
            timestamp_15min = (current_date - timedelta(
                minutes=current_date.minute % 15,
                seconds=current_date.second,
                microseconds=current_date.microsecond
            )).timestamp()
            
            batch_df['initial_risk'] = batch_df.apply(
                lambda row: get_cached_prediction(
                    row['product'],
                    row['days_old'],
                    row['temperature'],
                    row['humidity'],
                    timestamp_15min
                ),
                axis=1
            )
            return batch_df
        
        # Split data into chunks for parallel processing
        chunks = np.array_split(df, 4)
        with ThreadPoolExecutor(max_workers=4) as executor:
            processed_chunks = list(executor.map(calculate_batch_risks, chunks))
        
        # Combine processed chunks
        df = pd.concat(processed_chunks)
        
        # Filter high-risk items
        mask = (df['initial_risk'] > 0.6) | (df['days_to_expiry'] <= 2)
        high_risk_df = df[mask].copy()
        
        if not high_risk_df.empty:
            # Calculate detailed risk scores
            high_risk_df['risk_score'] = high_risk_df.apply(
                lambda row: calculate_risk_score(row, row['initial_risk'], current_date),
                axis=1
            )
            high_risk_df['freshness'] = (1 - high_risk_df['risk_score']) * 100
            
            # Get top 20 most critical items
            high_risk_items = (
                high_risk_df
                .nsmallest(20, 'freshness')
                .apply(lambda row: {
                    "product": row['product'],
                    "category": row['category'],
                    "emoji": get_emoji(row['product']),
                    "days_old": row['days_old'],
                    "freshness": round(row['freshness'], 1),
                    "location": row['store_location']
                }, axis=1)
                .tolist()
            )
        else:
            high_risk_items = []
        
        # Calculate metrics
        total_items = len(df)
        saved_units = total_items - len(high_risk_df)
        co2_saved = saved_units * 2.5
        
        # Update session metrics
        metrics = {
            "saved_units": saved_units,
            "co2_saved": round(co2_saved, 1),
            "green_score": min(round((saved_units / total_items * 50) + (min(co2_saved / 1000 * 100, 100) * 0.5)), 100) if total_items > 0 else 0
        }
        
        session['metrics'] = metrics
        session.modified = True
        
        # Store high-risk items in session for redistribution
        session['high_risk_items'] = high_risk_items
        
        return render_template(
            "dashboard.html",
            metrics=metrics,
            items=high_risk_items
        )
        
    except Exception as e:
        print(f"Dashboard error: {str(e)}")
        return render_template(
            "dashboard.html",
            metrics={"saved_units": 0, "co2_saved": 0, "green_score": 0},
            items=[],
            error="Unable to load dashboard data. Please try again."
        )

@app.route("/redistribute", methods=["POST"])
def redistribute():
    """Optimized redistribution endpoint with proper error handling"""
    try:
        data = request.get_json()
        if not data:
            raise ValueError("No data received")
        
        required_fields = ['product', 'location', 'freshness', 'outlet']
        if not all(field in data for field in required_fields):
            raise ValueError("Missing required fields")
        
        product = data["product"]
        location = data["location"]
        freshness = float(data["freshness"])
        outlet = data["outlet"]
        
        # Verify item exists in high-risk items
        if 'high_risk_items' not in session:
            raise ValueError("No high-risk items in session")
        
        item_exists = any(
            item["product"] == product and item["location"] == location 
            for item in session['high_risk_items']
        )
        if not item_exists:
            raise ValueError("Item not found in high-risk items")
        
        # Find recipients (cached)
        @cache.memoize(timeout=300)
        def get_cached_recipients(product, location):
            return find_recipients(recipients, product, location)
        
        candidates = get_cached_recipients(product, location)
        if not candidates:
            raise ValueError("No suitable recipients found")
        
        # Calculate routes in parallel
        def calculate_route(rec):
            route = get_route(location, rec["location"])
            return {**rec, "distance": route["distance_km"], "co2": route["emissions_kg"]}
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            routes = list(executor.map(calculate_route, candidates))
        
        total_co2 = sum(r["co2"] for r in routes)
        
        # Update metrics
        if 'metrics' not in session:
            session['metrics'] = {"saved_units": 0, "co2_saved": 0, "green_score": 0}
        
        session['metrics']["saved_units"] += round(freshness / 100, 2)
        session['metrics']["co2_saved"] += round(total_co2, 2)
        
        # Remove item from high-risk items
        session['high_risk_items'] = [
            item for item in session['high_risk_items']
            if not (item["product"] == product and item["location"] == location)
        ]
        session.modified = True
        
        return jsonify({
            "success": True,
            "metrics": session['metrics'],
            "routes": routes,
            "reload": len(session['high_risk_items']) == 0
        })
        
    except Exception as e:
        print(f"Redistribution error: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

if __name__ == "__main__":
    app.run(debug=True)
