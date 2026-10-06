import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_traffic_dataset(output_filename="historical_traffic_data.csv", days=30):
    np.random.seed(42) # For reproducible results
    
    # Define time frame sequence targets
    start_date = datetime.now() - timedelta(days=days)
    date_range = [start_date + timedelta(hours=i) for i in range(days * 24)]
    
    data = []
    
    for dt in date_range:
        hour = dt.hour
        is_weekend = 1 if dt.weekday() >= 5 else 0
        
        # 1. Base Vehicle Flow Count Simulation Logic
        # Typical city flow baseline traffic lower at night, higher during day
        base_vehicles = np.random.randint(150, 400)
        
        # 2. Add Rush-Hour Congestion Spikes (Peak Hours)
        # Morning peak: 8:00 AM - 10:00 AM
        # Evening peak: 5:00 PM - 7:00 PM
        rush_hour_bonus = 0
        if not is_weekend:
            if 8 <= hour <= 10:
                rush_hour_bonus = np.random.randint(600, 950)
            elif 17 <= hour <= 19:
                rush_hour_bonus = np.random.randint(700, 1100)
        else:
            # Weekends have a smoother afternoon peak distribution pattern
            if 12 <= hour <= 16:
                rush_hour_bonus = np.random.randint(300, 600)
                
        total_vehicles = base_vehicles + rush_hour_bonus
        
        # 3. Simulate Travel Delay Metric (Minutes per kilometer)
        base_delay = 1.5 # Clear traffic delay factor baseline
        delay_multiplier = (total_vehicles / 300) ** 1.8
        travel_delay = round(base_delay * delay_multiplier + np.random.uniform(-0.5, 0.5), 2)
        travel_delay = max(1.0, travel_delay) # Prevent mathematical floor errors
        
        # 4. Generate Random Environmental Features Attributes
        weather = np.random.choice(["Clear Skies", "Light Monsoon Drizzle", "Heavy Storm Outage"], p=[0.75, 0.20, 0.05])
        is_holiday = np.random.choice([0, 1], p=[0.96, 0.04]) if not is_weekend else 0
        
        # Weather delay compounding multiplier factors
        if weather == "Light Monsoon Drizzle":
            total_vehicles = int(total_vehicles * 0.9) # Speeds drop, fewer entries
            travel_delay *= 1.3
        elif weather == "Heavy Storm Outage":
            total_vehicles = int(total_vehicles * 0.7)
            travel_delay *= 2.1
            
        # 5. Determine Target Congestion Level Class Label (Ground Truth Target)
        # This acts as the classification layer output variable for Machine Learning
        if travel_delay > 6.5 or total_vehicles > 1000:
            congestion_level = "High"
        elif travel_delay > 3.0 or total_vehicles > 600:
            congestion_level = "Medium"
        else:
            congestion_level = "Low"
            
        data.append({
            "Timestamp": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "Hour": hour,
            "Is_Weekend": is_weekend,
            "Is_Holiday": is_holiday,
            "Weather_Condition": weather,
            "Junction_Activity_Weight": np.random.randint(3, 12),
            "Vehicle_Count": total_vehicles,
            "Travel_Delay_Index": round(travel_delay, 2),
            "Congestion_Level": congestion_level
        })
        
    # Convert data dictionary entries stack to a clear structural Pandas DataFrame
    df = pd.DataFrame(data)
    df.to_csv(output_filename, index=False)
    print(f"📦 Dataset successfully written as: {output_filename}")
    print(f"📉 Total log records generated: {len(df)} rows.")

if __name__ == "__main__":
    generate_traffic_dataset()
