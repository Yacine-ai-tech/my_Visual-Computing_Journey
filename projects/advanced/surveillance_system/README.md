# Real-Time Surveillance System

Comprehensive surveillance system with motion detection, multi-object tracking, event logging, and alert management.

## Overview

This module implements a complete surveillance solution combining multiple detection methods, intelligent tracking, restricted zone monitoring, and event management for security and monitoring applications.

## Features

### Detection Capabilities

#### Motion Detection
- Background subtraction (MOG2 algorithm)
- Adaptive sensitivity levels (low, medium, high)
- Shadow detection and removal
- Motion history tracking
- Motion heatmap generation

#### Object Detection
- Cascade-based detection (face, full body, upper body)
- Color-based object identification
- Size and shape filtering
- Confidence scoring

#### Scene Analysis
- Crowd density estimation
- Loitering detection
- Object abandonment detection
- Direction of movement analysis

### Tracking System

#### Multi-Object Tracking
- Unique ID assignment for each detected object
- Centroid-based tracking algorithm
- Track history and trajectory recording
- Occlusion handling with prediction

#### Track Management
- Track lifecycle (new, active, lost, ended)
- Confidence-based track filtering
- Re-identification after occlusion
- Track merging and splitting

### Zone Monitoring

#### Restricted Zones
- Polygonal zone definition
- Entry/exit detection
- Dwell time calculation
- Zone violation alerts

#### Zone Types
- Restricted areas (no entry)
- Monitoring zones (count entries)
- Exit-only zones
- Time-based restricted zones

### Event System

#### Event Detection
- Motion detection events
- Zone violation events
- Loitering alerts
- Crowd threshold alerts
- Object abandonment alerts

#### Event Logging
- Timestamp recording
- Event type classification
- Confidence scores
- Associated object IDs
- Video clip extraction

### Alert Management

#### Alert Types
- Real-time notifications
- Email alerts
- SMS notifications (via API)
- Sound alarms
- Visual alerts on display

#### Alert Conditions
- Motion in restricted zones
- Unauthorized entry
- Crowd density exceeded
- Suspicious behavior detected
- System health issues

### Performance Monitoring

#### System Metrics
- FPS (Frames Per Second)
- Detection rate
- Active track count
- Event frequency
- System resource usage

#### Health Monitoring
- Camera status
- Processing latency
- Storage space
- Network connectivity
- Error tracking

## Usage

```bash
cd projects/advanced/surveillance_system
python surveillance_demo.py
```

### Command-Line Arguments

```bash
# Set sensitivity level
python surveillance_demo.py --sensitivity high

# Monitor specific zones
python surveillance_demo.py --zones config/zones.json

# Enable alerts
python surveillance_demo.py --alerts email,visual

# Video source
python surveillance_demo.py --source camera.mp4

# Live camera
python surveillance_demo.py --source 0  # Webcam
```

## System Architecture

```python
class SurveillanceSystem:
    def __init__(self, sensitivity='medium'):
        # Detection components
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2()
        self.face_cascade = cv2.CascadeClassifier(...)
        self.person_cascade = cv2.CascadeClassifier(...)
        
        # Tracking system
        self.active_tracks = {}
        self.next_track_id = 0
        
        # Event system
        self.events = []
        self.alert_queue = []
        
        # Zone monitoring
        self.restricted_zones = []
        self.monitoring_zones = []
```

## Algorithm Details

### Motion Detection Pipeline

```python
def detect_motion(frame):
    # 1. Apply background subtraction
    fg_mask = bg_subtractor.apply(frame)
    
    # 2. Remove shadows (value 127 in mask)
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
    
    # 3. Clean mask
    kernel = np.ones((5, 5), np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 4. Calculate motion percentage
    motion_pixels = np.sum(fg_mask > 0)
    motion_percentage = motion_pixels / total_pixels
    
    # 5. Find moving objects
    contours = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, ...)
    
    detections = []
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            bbox = cv2.boundingRect(contour)
            detections.append(bbox)
    
    return detections, motion_percentage
```

### Multi-Object Tracking

```python
def update_tracks(detections):
    # 1. Compute centroids of new detections
    new_centroids = [compute_centroid(bbox) for bbox in detections]
    
    # 2. Match with existing tracks (minimum distance)
    matched = []
    unmatched_tracks = []
    unmatched_detections = list(range(len(new_centroids)))
    
    for track_id, track in active_tracks.items():
        distances = [distance(track.centroid, c) for c in new_centroids]
        
        if len(distances) > 0:
            min_idx = np.argmin(distances)
            min_dist = distances[min_idx]
            
            if min_dist < max_distance:
                track.update(new_centroids[min_idx], detections[min_idx])
                matched.append(min_idx)
                unmatched_detections.remove(min_idx)
            else:
                track.mark_disappeared()
                if track.disappeared_frames > max_disappeared:
                    unmatched_tracks.append(track_id)
        else:
            track.mark_disappeared()
    
    # 3. Create new tracks for unmatched detections
    for idx in unmatched_detections:
        new_track = Track(next_track_id, new_centroids[idx], detections[idx])
        active_tracks[next_track_id] = new_track
        next_track_id += 1
    
    # 4. Remove lost tracks
    for track_id in unmatched_tracks:
        del active_tracks[track_id]
```

### Zone Monitoring

```python
def check_zone_violations(tracks, zones):
    violations = []
    
    for track in tracks:
        for zone in zones:
            # Check if track is inside restricted zone
            if point_in_polygon(track.centroid, zone.polygon):
                # Calculate dwell time
                if track.id in zone.track_entry_times:
                    dwell_time = time.time() - zone.track_entry_times[track.id]
                else:
                    zone.track_entry_times[track.id] = time.time()
                    dwell_time = 0
                
                # Check violation conditions
                if zone.is_restricted or dwell_time > zone.max_dwell_time:
                    violation = {
                        'track_id': track.id,
                        'zone_name': zone.name,
                        'dwell_time': dwell_time,
                        'timestamp': datetime.now()
                    }
                    violations.append(violation)
            else:
                # Track left zone
                if track.id in zone.track_entry_times:
                    del zone.track_entry_times[track.id]
    
    return violations
```

### Event Logging

```python
from datetime import datetime

def log_event(event_type, details):
    event = {
        'timestamp': datetime.now(),
        'type': event_type,
        'details': details,
        'frame_number': current_frame,
        'screenshot': save_frame_snapshot()
    }
    
    events.append(event)
    
    # Trigger alerts if necessary
    if event_type in alert_triggers:
        send_alert(event)
    
    # Log to database
    db.insert_event(event)
```

## Sensitivity Levels

### Detection Sensitivity

| Level | Motion Threshold | Min Area | Use Case |
|-------|-----------------|----------|----------|
| Low | 0.05 | 2000 px | Outdoor, windy |
| Medium | 0.03 | 1000 px | Indoor, general |
| High | 0.01 | 500 px | Secure areas |

### Parameters by Sensitivity

```python
sensitivity_params = {
    'low': {
        'bg_threshold': 50,        # Less sensitive to changes
        'min_area': 2000,          # Larger objects only
        'motion_threshold': 0.05   # 5% motion required
    },
    'medium': {
        'bg_threshold': 30,
        'min_area': 1000,
        'motion_threshold': 0.03   # 3% motion
    },
    'high': {
        'bg_threshold': 16,        # Very sensitive
        'min_area': 500,           # Small objects detected
        'motion_threshold': 0.01   # 1% motion triggers alert
    }
}
```

## Applications

### Security & Surveillance
- Perimeter security
- Building entry monitoring
- Parking lot surveillance
- Asset protection

### Retail Analytics
- Customer traffic analysis
- Queue management
- Dwell time measurement
- Conversion rate tracking

### Industrial Safety
- Restricted area monitoring
- PPE compliance checking
- Worker safety monitoring
- Equipment area security

### Smart Buildings
- Occupancy monitoring
- Energy management
- Space utilization
- Access control

### Traffic Monitoring
- Traffic flow analysis
- Parking space detection
- Incident detection
- License plate recognition preparation

## Zone Configuration

### Zone Definition Format (JSON)

```json
{
  "zones": [
    {
      "name": "Restricted Area 1",
      "type": "restricted",
      "polygon": [[100, 100], [300, 100], [300, 300], [100, 300]],
      "max_dwell_time": 5.0,
      "alert_on_entry": true
    },
    {
      "name": "Monitoring Zone",
      "type": "monitoring",
      "polygon": [[400, 100], [600, 100], [600, 300], [400, 300]],
      "count_entries": true
    }
  ]
}
```

### Zone Types

**Restricted Zone:**
- No entry allowed
- Immediate alert on detection
- Visual/audio warning

**Monitoring Zone:**
- Count entries/exits
- Track dwell time
- Analytics only (no alerts)

**Exit-Only Zone:**
- Entry triggers alert
- Exit is normal
- One-way flow enforcement

## Event Types

### Motion Events
- Motion detected
- Motion started
- Motion stopped
- Motion threshold exceeded

### Tracking Events
- New object detected
- Object entered zone
- Object left zone
- Object loitering
- Track lost/reacquired

### Security Events
- Unauthorized entry
- Zone violation
- Object abandoned
- Crowd density alert
- Perimeter breach

### System Events
- Camera online/offline
- Processing error
- Storage warning
- Performance degradation

## Output Files

### Logs and Data
- `surveillance_log.txt`: Text-based event log
- `events.json`: Structured event data
- `tracks.csv`: Track history and statistics
- `performance_metrics.txt`: System performance data

### Media
- `motion_heatmap.png`: Motion activity visualization
- `zone_violations.png`: Zone violation screenshots
- `track_trajectories.png`: Object movement paths
- `event_snapshots/`: Folder with event screenshots

### Reports
- `daily_summary.pdf`: Daily activity summary
- `zone_statistics.csv`: Per-zone analytics
- `alert_history.csv`: Alert log

## Parameters Guide

### Background Subtraction

```python
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,          # Frames for background model
    varThreshold=16,      # Sensitivity (lower = more sensitive)
    detectShadows=True    # Shadow detection
)
```

### Tracking Parameters

```python
max_disappeared = 30      # Frames before removing track
max_distance = 50         # Max pixels for track association
min_track_length = 10     # Min frames to consider valid track
```

### Zone Monitoring

```python
max_dwell_time = 10.0     # Seconds before loitering alert
entry_cooldown = 5.0      # Seconds between zone entry alerts
crowd_threshold = 5       # Max simultaneous objects in zone
```

### Alert Settings

```python
alert_cooldown = 30       # Seconds between same-type alerts
min_confidence = 0.7      # Minimum detection confidence for alert
alert_buffer = 3          # Frames to confirm event before alert
```

## Tips for Best Results

### Camera Placement

**Height and Angle:**
- Mount 8-12 feet high
- 30-45 degree downward angle
- Avoid extreme angles
- Ensure full coverage

**Lighting:**
- Consistent, diffuse lighting
- Avoid direct sunlight
- Use IR illumination for night
- Minimize shadows

**Environment:**
- Stable mounting (minimize vibration)
- Weather protection for outdoor
- Clear field of view
- Avoid moving background elements

### Performance Optimization

**High FPS:**
- Reduce resolution (720p sufficient)
- Limit detection frequency
- Use efficient algorithms
- Optimize zone polygons

**High Accuracy:**
- Higher resolution
- Multiple detection methods
- Longer track history
- Lower sensitivity threshold

**Balanced:**
- 720p @ 20-30 FPS
- Detect every 2-3 frames
- Track every frame
- Medium sensitivity

### Reducing False Alarms

1. **Adjust Sensitivity:** Start low, increase gradually
2. **Filter by Size:** Set appropriate min/max area
3. **Use Confidence:** Require multiple confirmations
4. **Zone Design:** Define precise boundaries
5. **Environmental:** Address lighting, shadows, wind

## Advanced Features

### Crowd Density Estimation

```python
def estimate_crowd_density(detections, frame_area):
    # Method 1: Simple counting
    count = len(detections)
    density = count / frame_area
    
    # Method 2: Area coverage
    total_detection_area = sum([w * h for (x, y, w, h) in detections])
    coverage = total_detection_area / frame_area
    
    # Classify density
    if density < 0.001:
        return "Low"
    elif density < 0.005:
        return "Medium"
    else:
        return "High"
```

### Loitering Detection

```python
def detect_loitering(track, threshold_time=30.0):
    # Check if track has been stationary
    if track.lifetime > threshold_time:
        # Calculate movement variance
        positions = track.position_history
        variance = np.var(positions, axis=0)
        
        # Low variance = stationary = loitering
        if np.max(variance) < 100:  # 10 pixels squared
            return True
    
    return False
```

### Direction Analysis

```python
def analyze_direction(track):
    if len(track.position_history) < 2:
        return None
    
    # Calculate movement vector
    start = track.position_history[0]
    end = track.position_history[-1]
    vector = (end[0] - start[0], end[1] - start[1])
    
    # Calculate angle
    angle = np.arctan2(vector[1], vector[0]) * 180 / np.pi
    
    # Classify direction
    if -45 <= angle < 45:
        return "Right"
    elif 45 <= angle < 135:
        return "Down"
    elif -135 <= angle < -45:
        return "Up"
    else:
        return "Left"
```

## Integration Options

### Database Integration

```python
import sqlite3

# Connect to database
conn = sqlite3.connect('surveillance.db')

# Log event
def log_to_database(event):
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO events (timestamp, type, details, confidence)
        VALUES (?, ?, ?, ?)
    ''', (event['timestamp'], event['type'], 
          json.dumps(event['details']), event['confidence']))
    conn.commit()
```

### Email Alerts

```python
import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart

def send_email_alert(event):
    msg = MIMEMultipart()
    msg['Subject'] = f"Surveillance Alert: {event['type']}"
    msg['From'] = "surveillance@example.com"
    msg['To'] = "security@example.com"
    
    # Attach event details
    text = MIMEText(f"Event: {event['type']}\n"
                   f"Time: {event['timestamp']}\n"
                   f"Details: {event['details']}")
    msg.attach(text)
    
    # Attach screenshot
    with open(event['screenshot'], 'rb') as f:
        img = MIMEImage(f.read())
        msg.attach(img)
    
    # Send email
    smtp = smtplib.SMTP('smtp.example.com', 587)
    smtp.send_message(msg)
    smtp.quit()
```

### Web Dashboard

```python
from flask import Flask, render_template, jsonify

app = Flask(__name__)

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/status')
def get_status():
    return jsonify({
        'fps': current_fps,
        'active_tracks': len(active_tracks),
        'recent_events': events[-10:],
        'system_health': 'OK'
    })

@app.route('/api/live_feed')
def live_feed():
    # Stream video with annotations
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')
```

## Common Issues

**Problem**: Too many false motion alerts
**Solution**: Increase `varThreshold`, adjust `min_area`, use morphological filtering

**Problem**: Missing detections in crowded scenes
**Solution**: Lower `min_area`, use person detector, improve lighting

**Problem**: Track ID switching
**Solution**: Reduce `max_distance`, increase `max_disappeared`, improve detection consistency

**Problem**: System too slow
**Solution**: Reduce resolution, skip frames, limit detection frequency, optimize zones

**Problem**: Nighttime poor performance
**Solution**: Use IR camera, adjust exposure, use motion-only detection

## System Requirements

### Minimum
- CPU: Dual-core 2.5 GHz
- RAM: 4 GB
- Camera: 720p @ 15 FPS
- Storage: 100 GB

### Recommended
- CPU: Quad-core 3.0 GHz
- RAM: 8 GB
- Camera: 1080p @ 30 FPS
- Storage: 500 GB (for video retention)

### High-End
- CPU: 8-core 3.5+ GHz
- RAM: 16 GB
- GPU: NVIDIA with CUDA
- Camera: 4K @ 30 FPS or multiple 1080p
- Storage: 2+ TB with RAID

## Extensions

The code can be extended with:
- Deep learning person detection (YOLO, SSD)
- Face recognition for identification
- Action recognition (fighting, falling)
- License plate recognition
- Weapon detection
- Fire and smoke detection
- Multi-camera tracking
- 3D tracking with stereo cameras
- Predictive analytics
- Cloud storage and processing

## Requirements

- opencv-python >= 4.5.0
- numpy >= 1.19.0
- matplotlib >= 3.3.0
- scipy >= 1.5.0

Optional:
- smtplib (email alerts)
- flask (web dashboard)
- sqlite3 (database)

## References

- Yilmaz, A., Javed, O., & Shah, M. (2006). Object tracking: A survey
- Zivkovic, Z. (2004). Improved adaptive Gaussian mixture model
- Bewley, A., et al. (2016). Simple Online and Realtime Tracking
- OpenCV Documentation: Video Analysis and Object Tracking
