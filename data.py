import fastf1 as ff1
import pandas as pd
import time

combined_events = pd.DataFrame() 

ff1.Cache.enable_cache(r'C:\Users\marce\OneDrive\Documents\F1\FF1Cache')

# VARIABLES
year = int(input("What championshiop do you want to load (2018, 2024): "))
while year < 2018 or year > 2025:
    print("Invalid year")
    year = int(input("What championshiop do you want to load (2018, 2024): "))
race_index = 1
c_year = year

# List to store event details
all_events = []

# Main loop
while (c_year != year + 1):

    try: # tests to see if race exists
        # Load the race session
        session = ff1.get_session(year, race_index, 'R')
        session.load()
        track_status_data = session.track_status
        weather_data = session.weather_data
        results = session.results
            
        # Calculate track status and weather data
        yellow_count = track_status_data['Status'].str.count("2").sum()
        red_count = track_status_data['Status'].str.count("5").sum()
        safety_count = track_status_data['Status'].str.count("4").sum()
        v_safety_count = track_status_data['Status'].str.count("6").sum()
        is_rain = weather_data['Rainfall'].sum()
        is_rain = 1 if is_rain > 0 else 0 # Normalize rain

        # Set columns and values
        values = [session.event['Country'], yellow_count, red_count, safety_count, v_safety_count, is_rain]
        columns_info = ['Country', 'Yellows', 'Reds', 'Safety', 'Virtual Safety', 'Rain']

        #Driver performance

        pos = 1
        total = 0

        for grid, end in zip(results['GridPosition'], results['Position']):
            pre = (grid - end)
            total += abs(grid - end)
            values.append(pre / 20)

            columns_info.append(pos)
            pos += 1

        values.insert(0, total)
        columns_info.insert(0, 'Overtakes')

        all_events.append(values)
        race_index += 1

        while len(columns_info) < 27: # Missing drivers
            columns_info.append(0)

        # Convert the list of event details to a DataFrame
        combined_events = pd.DataFrame(all_events, columns = columns_info)
        print(combined_events)

        time.sleep(1)

    except: # end of year close csv and move to the next
        break

combined_events.to_csv(f'{year}.csv', index=True)  
combined_events.to_csv(f'{year} - Copy.csv', index=True)      
year +=1
race_index = 1
    


