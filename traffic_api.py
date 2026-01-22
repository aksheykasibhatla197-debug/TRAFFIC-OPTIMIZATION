"""
TomTom Traffic API Integration
Fetches real-time traffic flow and incident data
"""

import requests
import os
from datetime import datetime
import pandas as pd

class TomTomTrafficAPI:
    """Real-time traffic data from TomTom API"""
    
    BASE_URL = "https://api.tomtom.com"
    
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get('TOMTOM_API_KEY', '')
        
    def is_configured(self):
        """Check if API key is configured"""
        return bool(self.api_key)
    
    def get_traffic_flow(self, lat, lon, zoom=10):
        """
        Get traffic flow data for a location
        Returns current speed, free flow speed, and congestion level
        """
        if not self.api_key:
            return None
            
        url = f"{self.BASE_URL}/traffic/services/4/flowSegmentData/absolute/{zoom}/json"
        params = {
            'point': f"{lat},{lon}",
            'key': self.api_key,
            'unit': 'KMPH'
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                flow_data = data.get('flowSegmentData', {})
                
                current_speed = flow_data.get('currentSpeed', 0)
                free_flow_speed = flow_data.get('freeFlowSpeed', 0)
                confidence = flow_data.get('confidence', 0)
                
                if free_flow_speed > 0:
                    congestion = max(0, (1 - current_speed / free_flow_speed) * 100)
                else:
                    congestion = 0
                    
                return {
                    'current_speed': current_speed,
                    'free_flow_speed': free_flow_speed,
                    'congestion_percent': round(congestion, 1),
                    'confidence': confidence,
                    'road_closure': flow_data.get('roadClosure', False),
                    'timestamp': datetime.now().isoformat()
                }
            return None
        except Exception as e:
            print(f"Traffic flow error: {e}")
            return None
    
    def get_traffic_incidents(self, bbox, categories=None):
        """
        Get traffic incidents in a bounding box
        bbox: (min_lon, min_lat, max_lon, max_lat)
        categories: list of incident types (Unknown, Accident, Fog, DangerousConditions, Rain, Ice, Jam, LaneClosed, RoadClosed, RoadWorks, Wind, Flooding, Detour)
        """
        if not self.api_key:
            return []
            
        min_lon, min_lat, max_lon, max_lat = bbox
        
        url = f"{self.BASE_URL}/traffic/services/5/incidentDetails"
        params = {
            'bbox': f"{min_lon},{min_lat},{max_lon},{max_lat}",
            'key': self.api_key,
            'fields': '{incidents{type,geometry{type,coordinates},properties{iconCategory,magnitudeOfDelay,events{description,code},startTime,endTime,from,to,length,delay,roadNumbers,aci{probabilityOfOccurrence,numberOfReports,lastReportTime}}}}',
            'language': 'en-US',
            'categoryFilter': ','.join(categories) if categories else None,
            't': 1111111111111
        }
        
        params = {k: v for k, v in params.items() if v is not None}
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                incidents = []
                
                for incident in data.get('incidents', []):
                    props = incident.get('properties', {})
                    geom = incident.get('geometry', {})
                    coords = geom.get('coordinates', [[0, 0]])
                    
                    if geom.get('type') == 'LineString' and coords:
                        lat = coords[0][1] if isinstance(coords[0], list) else coords[1]
                        lon = coords[0][0] if isinstance(coords[0], list) else coords[0]
                    else:
                        lat, lon = 0, 0
                    
                    icon_category = props.get('iconCategory', 0)
                    incident_type = self._get_incident_type(icon_category)
                    severity = self._get_severity(props.get('magnitudeOfDelay', 0))
                    
                    events = props.get('events', [])
                    description = events[0].get('description', 'Unknown incident') if events else 'Traffic incident'
                    
                    incidents.append({
                        'type': incident_type,
                        'severity': severity,
                        'description': description,
                        'latitude': lat,
                        'longitude': lon,
                        'delay_seconds': props.get('delay', 0),
                        'length_meters': props.get('length', 0),
                        'road_numbers': props.get('roadNumbers', []),
                        'from_location': props.get('from', ''),
                        'to_location': props.get('to', ''),
                        'start_time': props.get('startTime', ''),
                        'end_time': props.get('endTime', '')
                    })
                
                return incidents
            return []
        except Exception as e:
            print(f"Traffic incidents error: {e}")
            return []
    
    def _get_incident_type(self, icon_category):
        """Convert TomTom icon category to readable type"""
        types = {
            0: 'Unknown',
            1: 'Accident',
            2: 'Fog',
            3: 'Dangerous Conditions',
            4: 'Rain',
            5: 'Ice',
            6: 'Jam',
            7: 'Lane Closed',
            8: 'Road Closed',
            9: 'Road Works',
            10: 'Wind',
            11: 'Flooding',
            14: 'Broken Down Vehicle'
        }
        return types.get(icon_category, 'Unknown')
    
    def _get_severity(self, magnitude):
        """Convert magnitude of delay to severity"""
        if magnitude >= 4:
            return 'Critical'
        elif magnitude >= 3:
            return 'Major'
        elif magnitude >= 2:
            return 'Moderate'
        elif magnitude >= 1:
            return 'Minor'
        return 'Low'
    
    def get_city_traffic_summary(self, city_coords, radius_km=10):
        """
        Get comprehensive traffic summary for a city
        city_coords: (lat, lon)
        """
        lat, lon = city_coords
        
        delta = radius_km / 111
        bbox = (lon - delta, lat - delta, lon + delta, lat + delta)
        
        flow = self.get_traffic_flow(lat, lon)
        incidents = self.get_traffic_incidents(bbox)
        
        accident_count = sum(1 for i in incidents if i['type'] == 'Accident')
        jam_count = sum(1 for i in incidents if i['type'] == 'Jam')
        road_works = sum(1 for i in incidents if i['type'] == 'Road Works')
        closures = sum(1 for i in incidents if i['type'] in ['Road Closed', 'Lane Closed'])
        
        critical_count = sum(1 for i in incidents if i['severity'] == 'Critical')
        major_count = sum(1 for i in incidents if i['severity'] == 'Major')
        
        if flow:
            congestion = flow['congestion_percent']
        else:
            congestion = min(100, (critical_count * 20 + major_count * 10 + len(incidents) * 2))
        
        if congestion >= 70:
            risk_level = 'Critical'
            risk_color = '#ff0040'
        elif congestion >= 50:
            risk_level = 'High'
            risk_color = '#ff6600'
        elif congestion >= 30:
            risk_level = 'Moderate'
            risk_color = '#ffcc00'
        else:
            risk_level = 'Low'
            risk_color = '#00ff88'
        
        return {
            'congestion_percent': congestion,
            'risk_level': risk_level,
            'risk_color': risk_color,
            'total_incidents': len(incidents),
            'accidents': accident_count,
            'jams': jam_count,
            'road_works': road_works,
            'closures': closures,
            'critical_incidents': critical_count,
            'major_incidents': major_count,
            'flow_data': flow,
            'incidents': incidents[:20],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def incidents_to_dataframe(self, incidents):
        """Convert incidents list to pandas DataFrame"""
        if not incidents:
            return pd.DataFrame()
        
        df = pd.DataFrame(incidents)
        
        severity_map = {'Critical': 'Fatal', 'Major': 'Serious', 'Moderate': 'Serious', 'Minor': 'Minor', 'Low': 'Minor'}
        df['severity'] = df['severity'].map(severity_map).fillna('Minor')
        
        df['date'] = datetime.now().strftime('%Y-%m-%d')
        df['time'] = datetime.now().strftime('%H:%M')
        df['num_casualties'] = df['severity'].map({'Fatal': 2, 'Serious': 1, 'Minor': 0}).fillna(0)
        df['num_vehicles'] = 1
        df['speed_limit'] = 50
        df['weather'] = 'Clear'
        df['road_type'] = 'Urban'
        df['light_conditions'] = 'Daylight' if 6 <= datetime.now().hour <= 18 else 'Dark - lights lit'
        df['road_surface'] = 'Dry'
        df['junction_type'] = 'Not at junction'
        
        return df


MAJOR_CITIES = {
    'New York': (40.7128, -74.0060),
    'Los Angeles': (34.0522, -118.2437),
    'Chicago': (41.8781, -87.6298),
    'Houston': (29.7604, -95.3698),
    'Phoenix': (33.4484, -112.0740),
    'London': (51.5074, -0.1278),
    'Paris': (48.8566, 2.3522),
    'Berlin': (52.5200, 13.4050),
    'Tokyo': (35.6762, 139.6503),
    'Sydney': (-33.8688, 151.2093),
    'Mumbai': (19.0760, 72.8777),
    'Delhi': (28.6139, 77.2090),
    'Singapore': (1.3521, 103.8198),
    'Dubai': (25.2048, 55.2708),
    'Toronto': (43.6532, -79.3832),
    'San Francisco': (37.7749, -122.4194),
    'Seattle': (47.6062, -122.3321),
    'Boston': (42.3601, -71.0589),
    'Miami': (25.7617, -80.1918),
    'Atlanta': (33.7490, -84.3880),
}


def get_demo_traffic_data():
    """Generate demo traffic data when API key is not configured"""
    import random
    
    incidents = []
    incident_types = ['Accident', 'Jam', 'Road Works', 'Lane Closed', 'Dangerous Conditions']
    severities = ['Critical', 'Major', 'Moderate', 'Minor']
    
    base_lat, base_lon = 40.7128, -74.0060
    
    for i in range(random.randint(8, 15)):
        lat = base_lat + random.uniform(-0.1, 0.1)
        lon = base_lon + random.uniform(-0.1, 0.1)
        
        incidents.append({
            'type': random.choice(incident_types),
            'severity': random.choice(severities),
            'description': f"Traffic incident on Route {random.randint(1, 99)}",
            'latitude': lat,
            'longitude': lon,
            'delay_seconds': random.randint(60, 1800),
            'length_meters': random.randint(100, 5000),
            'road_numbers': [f"I-{random.randint(1, 99)}"],
            'from_location': 'Downtown',
            'to_location': 'Uptown'
        })
    
    congestion = random.randint(20, 75)
    
    return {
        'congestion_percent': congestion,
        'risk_level': 'High' if congestion > 50 else 'Moderate' if congestion > 30 else 'Low',
        'risk_color': '#ff6600' if congestion > 50 else '#ffcc00' if congestion > 30 else '#00ff88',
        'total_incidents': len(incidents),
        'accidents': sum(1 for i in incidents if i['type'] == 'Accident'),
        'jams': sum(1 for i in incidents if i['type'] == 'Jam'),
        'road_works': sum(1 for i in incidents if i['type'] == 'Road Works'),
        'closures': sum(1 for i in incidents if i['type'] == 'Lane Closed'),
        'critical_incidents': sum(1 for i in incidents if i['severity'] == 'Critical'),
        'major_incidents': sum(1 for i in incidents if i['severity'] == 'Major'),
        'flow_data': {
            'current_speed': random.randint(20, 60),
            'free_flow_speed': 65,
            'congestion_percent': congestion,
            'confidence': 0.85
        },
        'incidents': incidents,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'is_demo': True
    }
