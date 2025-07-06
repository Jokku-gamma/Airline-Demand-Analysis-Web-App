from flask import Flask, render_template, request
import requests
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import json
from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()

app = Flask(__name__)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
else:
    print("OPENAI_API_KEY not found. Summary generation will not work.")
    openai_client = None
conn = sqlite3.connect('flight_data.db', check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS flights
    (date TEXT, origin TEXT, destination TEXT, data TEXT)
''')
conn.commit()
DUMMY_FLIGHT_DATA = {"flight_results": []} 
try:
    with open('flight_data.json', 'r') as f:
        DUMMY_FLIGHT_DATA = json.load(f)
    print("SUCCESS: Dummy flight data loaded successfully from flight_dummy_data.json.")
    print(f"Loaded {len(DUMMY_FLIGHT_DATA.get('flight_results', []))} dummy flight entries.")
except FileNotFoundError:
    print("ERROR: flight_dummy_data.json not found. Please ensure it's in the same directory as main.py.")
except json.JSONDecodeError as e:
    print(f"ERROR: Could not decode flight_dummy_data.json. Check JSON format. Error: {e}")
except Exception as e:
    print(f"UNEXPECTED ERROR loading dummy data: {e}")


def fetch_flight_data_from_dummy(origin, destination, date_str):
    print(f"\nDEBUG: Fetching dummy data for Request: Origin={origin}, Destination={destination}, Date={date_str}")
    
    filtered_flights = []
    dummy_results = DUMMY_FLIGHT_DATA.get("flight_results", [])
    
    if not dummy_results:
        print("DEBUG: Dummy data 'flight_results' list is empty. No data to filter.")
        return pd.DataFrame()

    for i, flight in enumerate(dummy_results):
        flight_date = flight.get('date')
        outbound_flight_data = flight.get('outbound_flight', {})
        outbound_origin_airport_full = outbound_flight_data.get('origin_airport', '')
        outbound_destination_airport_full = outbound_flight_data.get('destination_airport', '')
        outbound_origin_iata = outbound_origin_airport_full.split(' ')[0].upper() if outbound_origin_airport_full else ''
        outbound_destination_iata = outbound_destination_airport_full.split(' ')[0].upper() if outbound_destination_airport_full else ''
        if flight_date == date_str and \
           outbound_origin_iata == origin and \
           outbound_destination_iata == destination:
            adapted_flight = {
                'flight_date': flight_date,
                'airline': {'name': flight.get('booking_options', [{}])[0].get('platform', 'Unknown Airline')},
                'departure': {'scheduled': outbound_flight_data.get('departure_time', 'N/A')},
                'arrival': {'scheduled': outbound_flight_data.get('arrival_time', 'N/A')},
                'flight_status': flight.get('flight_status', 'scheduled'), 
                'flight': {'number': flight.get('flight_id', 'N/A')}
            }
            filtered_flights.append(adapted_flight)      
    print(f"DEBUG: Found {len(filtered_flights)} matching dummy flights for {origin}-{destination} on {date_str}.")
    return pd.DataFrame(filtered_flights)
def generate_summary(total_flights, flights_per_day): 
    if not openai_client:
        return "Summary generation is unavailable (OpenAI API key missing)."

    prompt = f"""
    Summarize the following flight demand data for a route:
    - Total flights found: {total_flights}
    - Flights per day: {flights_per_day}
    Highlight demand trends, high-demand periods, and any notable patterns in flight frequency.
    """
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo", 
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150 
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating summary with OpenAI: {e}")
        return "Summary generation failed."

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        origin = request.form.get('origin', '').upper()
        destination = request.form.get('destination', '').upper()
        start_date_str = request.form.get('start_date')
        end_date_str = request.form.get('end_date')
        if not all([origin, destination, start_date_str, end_date_str]):
            return render_template('index.html', error="All fields are required.")

        try:
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
        except ValueError:
            return render_template('index.html', error="Invalid date format. Please use IHDA-MM-DD.")
        if (end_date - start_date).days < 0:
            return render_template('index.html', error="End date cannot be before start date.")
        if (end_date - start_date).days > 89: 
            return render_template('index.html', error="Date range limited to 90 days for this demo.")
        all_flights = []
        for date in pd.date_range(start_date, end_date):
            date_str = date.strftime('%Y-%m-%d')
            flights_df = fetch_flight_data_from_dummy(origin, destination, date_str)
            if not flights_df.empty:
                all_flights.append(flights_df)
        
        if not all_flights:
            return render_template('index.html', error=f"No dummy flight data found for {origin} to {destination} in the selected dates. Please ensure 'flight_dummy_data.json' is correctly placed and try SYD to MEL between 2025-07-01 and 2025-09-30.")
        df = pd.concat(all_flights, ignore_index=True)
        df = df[df['airline'].apply(lambda x: isinstance(x, dict) and 'name' in x)]
        df = df[df['flight'].apply(lambda x: isinstance(x, dict) and 'number' in x)]
        total_flights = len(df)
        if total_flights == 0:
            return render_template('index.html', error="No valid flight data found after processing. Please try different dates or route (e.g., SYD to MEL between 2025-07-01 and 2025-09-30).")
        flights_per_day = df.groupby('flight_date').size().to_dict()
        summary = generate_summary(total_flights, flights_per_day)
        flights_data = []
        for _, row in df.iterrows():
            airline_name = row['airline'].get('name', 'N/A')
            departure_time = row['departure'].get('scheduled', 'N/A')
            arrival_time = row['arrival'].get('scheduled', 'N/A')
            flight_status = row.get('flight_status', 'N/A')
            booking_options = []
            original_flight_entry = next((f for f in DUMMY_FLIGHT_DATA.get("flight_results", []) if f.get('flight_id') == row['flight']['number']), None)
            if original_flight_entry and 'booking_options' in original_flight_entry:
                booking_options = original_flight_entry['booking_options']


            flights_data.append({
                'flight_date': row['flight_date'],
                'airline': airline_name,
                'departure_time': departure_time,
                'arrival_time': arrival_time,
                'flight_status': flight_status,
                'booking_options': booking_options
            })
        flights_per_day_labels = list(flights_per_day.keys())
        flights_per_day_values = list(flights_per_day.values())
        return render_template('results.html',
                               flights=flights_data,
                               total_flights=total_flights,
                               flights_per_day_labels=flights_per_day_labels, 
                               flights_per_day_values=flights_per_day_values, 
                               summary=summary,
                               origin=origin,
                               destination=destination)
    return render_template('index.html', error=None)

if __name__ == '__main__':
    app.run(debug=True)

