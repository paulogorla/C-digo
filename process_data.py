import pandas as pd
from math import radians, sin, cos, sqrt, atan2
import openrouteservice
from openrouteservice.directions import directions
import overpy

def haversine(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Check if the coordinates are the same
    if lat1 == lat2 and lon1 == lon2:
        return 0

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = 6371000 * c  # Radius of Earth in kilometers
    return distance

def get_route_dist(coord_st, coord_end, client):
    route = client.directions(
        coordinates=[coord_st, coord_end],
        profile='driving-car',
        format='geojson'
    )
    distance = route['features'][0]['properties']['segments'][0]['distance']

    return distance

def get_maxspeed(coordinates):
    lat, lon = coordinates
    api = overpy.Overpass()

    # fetch all ways and nodes
    result = api.query("""
            way(around:""" + '500' + """,""" + lat  + """,""" + lon  + """) ["maxspeed"];
                (._;>;);
                    out body;
                        """)
    results_list = []
    for way in result.ways:
        road = {}
        road["name"] = way.tags.get("name", "n/a")
        road["speed_limit"] = way.tags.get("maxspeed", "n/a")
        nodes = []
        for node in way.nodes:
            nodes.append((node.lat, node.lon))
        road["nodes"] = nodes
        results_list.append(road)
    return results_list[0]['speed_limit']

open_route_key = "5b3ce3597851110001cf62488ef6f5755f6c42be8a56e73c87f4e501"

def get_route_data(coord_start, coord_end, route_max_speed):
    client = openrouteservice.Client(key=open_route_key)

    route_start = (coord_start[1],coord_start[0])
    route_end = (coord_end[1],coord_end[0])

    route = directions(client,(route_start,route_end),profile='driving-car',format='geojson',elevation=True)

    coordinates = route['features'][0]['geometry']['coordinates']
    duration = route['features'][0]['properties']['segments'][0]['duration']

    df = pd.DataFrame({'Latitude': [coord[1] for coord in coordinates],'Longitude': [coord[0] for coord in coordinates],'altitude': [coord[2] for coord in coordinates]})
    
    df['distance'] = 0.

    for index, row in df.iterrows():
        if index == 0: next
        else:
            df.loc[index,'distance'] = haversine(row['Latitude'],row['Longitude'],df.loc[index-1]['Latitude'],df.loc[index-1]['Longitude'])

    df['recomm_speed'] = df['distance'].sum()/duration
    df['max_speed'] = route_max_speed/3.6

    for index, row in df.iterrows():
        if index == len(df) - 1:
            next
        if index == 0:
            next
        elif df.loc[index,'distance'] > 10:
            next
        else:
            df.loc[index+1,'distance'] =  df.loc[index+1,'distance']+df.loc[index,'distance']
    
    df = df[(df['distance'] >= 10) | (df.index == len(df)-1) | (df.index == 0)]
    df = df.reset_index(drop=True)

    new_rows = []

    for index, row in df.iterrows():
        if index == 0:
            new_rows.extend([row.copy()])
            next
        else:
            duplicates = int(round(row['distance']/10,0))
            new_rows.extend([row.copy()]*duplicates)
    
    new_df = pd.DataFrame(new_rows)
    new_df = new_df.reset_index(drop=True)
    new_df.loc[1:,'distance'] = 10.
    new_df['total_distance_traveled'] = new_df['distance'].cumsum()
    new_df['distance_remaining'] = new_df.iloc[::-1].reset_index()['total_distance_traveled']
    new_df['diff_to_next_100m_altitude'] = new_df['altitude'].shift(-11)
    new_df['diff_to_next_100m_altitude'].fillna(new_df.loc[len(new_df)-1,'altitude'],inplace=True)
    new_df['diff_to_next_100m_altitude'] = new_df['diff_to_next_100m_altitude'] - new_df['altitude']
    
    df = new_df[['total_distance_traveled','distance_remaining','altitude','diff_to_next_100m_altitude','max_speed','recomm_speed']]
    df['current_speed'] = 0.

    df['total_distance_traveled_normalized'] = (df['total_distance_traveled']-df['total_distance_traveled'].min())/(df['total_distance_traveled'].max()-df['total_distance_traveled'].min())
    df['distance_remaining_normalized'] = (df['distance_remaining']-df['distance_remaining'].min())/(df['distance_remaining'].max()-df['distance_remaining'].min())
    df['altitude_normalized'] = (df['altitude']-df['altitude'].min())/(df['altitude'].max()-df['altitude'].min())
    df['diff_to_next_100m_altitude_normalized'] = df['diff_to_next_100m_altitude']/df['altitude']
    df['max_speed_normalized'] = df['max_speed']/df['max_speed']
    df['recomm_speed_normalized'] = df['recomm_speed']/df['max_speed']
    df['current_speed_normalized'] = (df['current_speed']/df['max_speed']).astype(float)
    
    return df

def calculate_stats(group):
    first_row = group.iloc[0]
    last_row = group.iloc[-1]
    avg_speed = group['speed'].mean()
    avg_consumption = group['kpl'].mean()
    return pd.Series({
        'first_latitude': first_row['latitude'],
        'first_longitude': first_row['longitude'],
        'last_latitude': last_row['latitude'],
        'last_longitude': last_row['longitude'],
        'avg_speed': avg_speed,
        'avg_consumption': avg_consumption
    })

def get_routes_coords():
    columns_rename = {
        'Id': 'id',
        'Latitude': 'latitude',
        'Longitude': 'longitude',
        'KilometersPerLitre(Instant)(kpl)': 'kpl',
        'Speed(OBD)(KM/h)': 'speed',
    }
    #read excel 
    columns_to_adjust = ['latitude','longitude','speed','kpl']
    all_trips_df = pd.read_excel("data/Dataset.xlsx",usecols=['Id','Latitude','Longitude','Speed(OBD)(KM/h)','KilometersPerLitre(Instant)(kpl)'])
    
    #rename and adjust values
    all_trips_df = all_trips_df.rename(columns=columns_rename)
    all_trips_df[columns_to_adjust] = all_trips_df[columns_to_adjust].replace('-',0)
    all_trips_df[columns_to_adjust] = all_trips_df[columns_to_adjust].fillna(0)
    all_trips_df[columns_to_adjust] = all_trips_df[columns_to_adjust].astype(float)

    grouped = all_trips_df.groupby('id')
    result = grouped.apply(calculate_stats).reset_index()
    print(result)
    result = result[(result['first_latitude'] != result['last_latitude']) & (result['first_longitude'] != result['last_longitude']) & (result['avg_speed'] >= 60)]
    result = result.reset_index(drop=True)

    client = openrouteservice.Client(key=open_route_key)
    result['distance'] = result.apply(lambda row: get_route_dist([row['first_longitude'], row['first_latitude']], [row['last_longitude'], row['last_latitude']],client), axis=1)
    
    result = result[(result['distance'] >= 10000)]
    result = result.reset_index(drop=True)
    result['first_max_speed'] = result.apply(lambda row: get_maxspeed((str(row['first_latitude']),str(row['first_longitude']))),axis=1)
    result['last_max_speed'] = result.apply(lambda row: get_maxspeed((str(row['last_latitude']),str(row['last_longitude']))),axis=1)
    result.to_excel("data/processed_data.xlsx")

# test = get_routes_coords()
# print(test)

# get_routes_coords()

# coords= get_route_data([-26.479910862474775, -49.082270932250495],[-26.480604021607586, -49.08614209120706])        

# print(coords)

# routes = pd.read_excel("data/processed_data.xlsx")
# print(routes)

# coords = get_route_data([routes.loc[3]['first_latitude'],routes.loc[3]['first_longitude']],[routes.loc[3]['last_latitude'],routes.loc[3]['last_longitude']],routes.loc[3]['first_max_speed'])
# print(coords)
# coords.to_excel("first_route_normalized_new.xlsx")