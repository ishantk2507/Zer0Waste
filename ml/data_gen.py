import os, random, csv
from datetime import datetime, timedelta
import numpy as np
from faker import Faker

# Ensure data folder exists
os.makedirs("data", exist_ok=True)

fake = Faker("en_IN")
random.seed(42)
np.random.seed(42)

# 1. Define per-product profiles with more detailed characteristics
product_profiles = [
    {
        "product": "Apple",
        "category": "Fruit",
        "storage": "Refrigerated",
        "temp_range": (0, 4),
        "critical_temp": 10,  # Temperature above which spoilage accelerates
        "hum_range": (80, 95),
        "critical_humidity": 75,  # Humidity below which quality degrades
        "life": (20, 30),
        "sensitivity": 0.7  # How sensitive to conditions (0-1)
    },
    {
        "product": "Banana",
        "category": "Fruit",
        "storage": "Ambient",
        "temp_range": (13, 18),
        "critical_temp": 22,
        "hum_range": (60, 75),
        "critical_humidity": 80,
        "life": (3, 7),
        "sensitivity": 0.8
    },
    {
        "product": "Milk",
        "category": "Dairy",
        "storage": "Refrigerated",
        "temp_range": (1, 4),
        "critical_temp": 7,
        "hum_range": (85, 95),
        "critical_humidity": None,  # Not humidity sensitive
        "life": (10, 15),
        "sensitivity": 0.9
    },
    {
        "product": "Bread",
        "category": "Bakery",
        "storage": "Ambient",
        "temp_range": (18, 24),
        "critical_temp": 28,
        "hum_range": (50, 70),
        "critical_humidity": 75,
        "life": (3, 5),
        "sensitivity": 0.6
    },
    {
        "product": "Paneer",
        "category": "Dairy",
        "storage": "Refrigerated",
        "temp_range": (2, 5),
        "critical_temp": 8,
        "hum_range": (80, 90),
        "critical_humidity": None,
        "life": (7, 10),
        "sensitivity": 0.85
    },
    {
        "product": "Spinach",
        "category": "Vegetable",
        "storage": "Refrigerated",
        "temp_range": (1, 5),
        "critical_temp": 8,
        "hum_range": (85, 95),
        "critical_humidity": 80,
        "life": (4, 6),
        "sensitivity": 0.75
    },
    {
        "product": "Chicken",
        "category": "Meat",
        "storage": "Refrigerated",
        "temp_range": (0, 4),
        "critical_temp": 5,
        "hum_range": (85, 95),
        "critical_humidity": None,
        "life": (5, 10),
        "sensitivity": 0.95
    },
    {
        "product": "Ice Cream",
        "category": "FrozenDessert",
        "storage": "Frozen",
        "temp_range": (-20, -15),
        "critical_temp": -10,
        "hum_range": (30, 50),
        "critical_humidity": None,
        "life": (180, 365),
        "sensitivity": 0.8
    }
]

# City profiles with climate characteristics
cities = {
    "Delhi": {
        "name": "Delhi",
        "climate": "Hot",
        "avg_temp_variation": 5,
        "humidity_variation": 20,
        "infrastructure_score": 0.8  # Quality of storage infrastructure
    },
    "Mumbai": {
        "name": "Mumbai",
        "climate": "Tropical",
        "avg_temp_variation": 3,
        "humidity_variation": 15,
        "infrastructure_score": 0.85
    },
    "Bangalore": {
        "name": "Bangalore",
        "climate": "Moderate",
        "avg_temp_variation": 2,
        "humidity_variation": 10,
        "infrastructure_score": 0.9
    },
    "Kolkata": {
        "name": "Kolkata",
        "climate": "Humid",
        "avg_temp_variation": 4,
        "humidity_variation": 25,
        "infrastructure_score": 0.75
    },
    "Chennai": {
        "name": "Chennai",
        "climate": "Tropical",
        "avg_temp_variation": 3,
        "humidity_variation": 20,
        "infrastructure_score": 0.8
    }
}

def calculate_spoilage_probability(item_data, prof):
    """Calculate spoilage probability based on multiple factors"""
    
    # Base risk from time
    days_old = (datetime.now().date() - datetime.strptime(item_data['stock_date'], '%Y-%m-%d').date()).days
    max_life = prof['life'][1]
    time_risk = min(days_old / max_life, 1)
    
    # Temperature risk
    min_temp, max_temp = prof['temp_range']
    temp = item_data['temperature']
    if prof['critical_temp'] is not None:
        if prof['storage'] in ['Refrigerated', 'Frozen']:
            temp_risk = max(0, (temp - max_temp) / (prof['critical_temp'] - max_temp))
        else:
            temp_risk = max(0, abs(temp - np.mean(prof['temp_range'])) / (prof['critical_temp'] - max_temp))
    else:
        temp_risk = max(0, abs(temp - np.mean(prof['temp_range'])) / (max_temp - min_temp))

    # Humidity risk
    if prof['critical_humidity'] is not None:
        min_hum, max_hum = prof['hum_range']
        hum = item_data['humidity']
        if hum < min_hum:
            hum_risk = (min_hum - hum) / min_hum
        elif hum > max_hum:
            hum_risk = (hum - max_hum) / (100 - max_hum)
        else:
            hum_risk = 0
    else:
        hum_risk = 0

    # City infrastructure impact
    city = cities[item_data['store_location']]
    infrastructure_risk = (1 - city['infrastructure_score']) * 0.5

    # Calculate final risk score
    base_risk = (
        0.4 * time_risk +
        0.3 * temp_risk +
        0.2 * hum_risk +
        0.1 * infrastructure_risk
    )
    
    # Apply product sensitivity
    risk = base_risk * prof['sensitivity']
    
    # Add some randomness to account for unknown factors
    risk = risk * random.uniform(0.8, 1.2)
    
    return min(max(risk, 0), 1)

# Generate data
today = datetime.now().date()
batch_dates = [today - timedelta(days=7*i) for i in range(2)]  # 2 weeks of data

rows = []
for batch_date in batch_dates:
    batch_id = f"batch_{batch_date.isoformat()}"
    
    # Generate smaller number of samples
    for prof in product_profiles:
        # Generate samples based on sensitivity but with smaller base number
        n_samples = int(15 * (1 + prof['sensitivity']))  # Reduced from 75 to 15
        
        for _ in range(n_samples):
            # Get city profile
            city = random.choice(list(cities.values()))
            
            # Calculate storage conditions with city influence
            d_offset = random.randint(0, 6)
            stock = batch_date - timedelta(days=d_offset)
            life_days = random.randint(*prof['life'])
            expiry = stock + timedelta(days=life_days)
            
            # Calculate temperature with city influence
            base_temp = random.gauss(
                np.mean(prof['temp_range']),
                city['avg_temp_variation'] * (0.8 if prof['storage'] != 'Ambient' else 1.2)
            )
            temp = base_temp * (1 + random.gauss(0, 0.1))  # Add some noise
            
            # Calculate humidity with city influence
            base_hum = random.gauss(
                np.mean(prof['hum_range']),
                city['humidity_variation'] * 0.5
            )
            hum = base_hum * (1 + random.gauss(0, 0.1))  # Add some noise
            
            # Ensure values are within physical limits
            temp = max(min(temp, 50), -30)
            hum = max(min(hum, 100), 0)
            
            # Create item data
            item_data = {
                "batch_id": batch_id,
                "product": prof['product'],
                "category": prof['category'],
                "stock_date": stock.isoformat(),
                "storage_type": prof['storage'],
                "store_location": city['name'],
                "temperature": round(temp, 1),
                "humidity": round(hum, 1),
                "expiry_date": expiry.isoformat()
            }
            
            # Calculate spoilage based on all factors
            risk = calculate_spoilage_probability(item_data, prof)
            item_data["spoilage"] = 1 if random.random() < risk else 0
            
            rows.append(item_data)

# Write CSV
cols = ["batch_id", "product", "category", "stock_date", "storage_type", 
        "store_location", "temperature", "humidity", "expiry_date", "spoilage"]

with open("data/mock_inventory.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=cols)
    writer.writeheader()
    writer.writerows(rows)

print("Generated enhanced mock_inventory.csv with realistic spoilage patterns")