import network
import socket
import time
import camera
import machine
import ubinascii
import gc
from umqtt.simple import MQTTClient

# ===== Configuration =====
SSID = "Thiab2"
PASSWORD = "Kareem77186"
MQTT_BROKER = "192.168.8.101"
PORT = 8080

# ===== MQTT Topics =====
TOPIC_TRIGGER = b"car/trigger"
TOPIC_SUMMARY = b"car/summary"
TOPIC_GATE = b"car/gate"
TOPIC_SNAP = b"car/snap"  # New topic for snap commands

# ===== GPIO Setup =====
# Removed the problematic button pin
flash_led = machine.Pin(4, machine.Pin.OUT)
flash_led.value(0)

# ===== Global Variables =====
last_snapshot = None
mqtt_connected = False
wifi_connected = False
camera_initialized = False
client = None
s = None

# ===== WiFi Connection =====
def connect_wifi():
    global wifi_connected
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    
    if not wlan.isconnected():
        print('Connecting to WiFi...')
        wlan.connect(SSID, PASSWORD)
        
        max_attempts = 20
        for attempt in range(max_attempts):
            if wlan.isconnected():
                break
            print('Waiting for connection... ({}/{})'.format(attempt + 1, max_attempts))
            time.sleep(1)
    
    if wlan.isconnected():
        status = wlan.ifconfig()
        print('WiFi connected, IP:', status[0])
        wifi_connected = True
        return wlan
    else:
        print('WiFi connection failed')
        wifi_connected = False
        return None

# ===== MQTT Connection =====
def connect_mqtt():
    global client, mqtt_connected
    try:
        client_id = ubinascii.hexlify(machine.unique_id())
        client = MQTTClient(client_id, MQTT_BROKER, keepalive=60)
        client.set_callback(mqtt_callback)
        client.connect()
        # Subscribe to both gate control and snap commands
        client.subscribe(TOPIC_GATE)
        client.subscribe(TOPIC_SNAP)
        print("Connected to MQTT broker and subscribed to topics")
        mqtt_connected = True
        return True
    except Exception as e:
        print("MQTT connection failed:", e)
        mqtt_connected = False
        return False

# ===== MQTT Callback =====
def mqtt_callback(topic, msg):
    print("MQTT message received:", topic, msg)
    try:
        if topic == TOPIC_GATE:
            if msg == b"open":
                print("Opening gate - flash ON for 3 seconds")
                flash_led.value(1)
                time.sleep(3)
                flash_led.value(0)
                print("Flash turned OFF")
            elif msg == b"deny":
                print("Denying access - triple blink")
                for i in range(3):
                    flash_led.value(1)
                    time.sleep(0.2)
                    flash_led.value(0)
                    time.sleep(0.2)
                print("Triple blink completed")
        
        elif topic == TOPIC_SNAP:
            if msg == b"capture":
                print("MQTT snap command received - capturing image")
                capture_and_publish()
                
    except Exception as e:
        print("Error in MQTT callback:", e)

# ===== Camera Initialization =====
def init_camera():
    global camera_initialized
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            # Try to deinitialize first
            try:
                camera.deinit()
                time.sleep(0.5)
            except:
                pass
                
            # Initialize camera
            camera.init(0, format=camera.JPEG)
            camera.framesize(camera.FRAME_QVGA)
            camera.quality(12)
            print("Camera initialized successfully")
            camera_initialized = True
            return True
        except Exception as e:
            print("Camera initialization failed (attempt {}): {}".format(attempt + 1, e))
            time.sleep(1)
    
    print("All camera initialization attempts failed")
    camera_initialized = False
    return False

# ===== Capture and Publish Function =====
def capture_and_publish():
    global last_snapshot
    try:
        print("Free memory before capture:", gc.mem_free())
        
        # Turn on flash for better image quality
        flash_led.value(1)
        time.sleep(0.1)
        
        # Capture image
        buf = camera.capture()
        
        # Turn off flash immediately
        flash_led.value(0)
        
        if buf:
            print("Snapshot taken, size:", len(buf))
            
            # Free previous snapshot memory
            if last_snapshot:
                del last_snapshot
                gc.collect()
            
            # Store new snapshot
            last_snapshot = buf
            
            # Force garbage collection
            gc.collect()
            print("Free memory after GC:", gc.mem_free())
            
            # Publish trigger to MQTT
            if mqtt_connected:
                try:
                    client.publish(TOPIC_TRIGGER, b"car_detected")
                    print("Trigger published to MQTT")
                    return True
                except Exception as e:
                    print("MQTT publish error:", e)
                    mqtt_connected = False
                    return False
            else:
                print("MQTT not connected, can't publish trigger")
                return False
        else:
            print("Failed to capture image")
            return False
            
    except Exception as e:
        print("Camera capture error:", e)
        return False

# ===== HTTP Server Functions =====
def serve_single_image(cl):
    global last_snapshot
    if last_snapshot:
        cl.send("HTTP/1.1 200 OK\r\n")
        cl.send("Content-Type: image/jpeg\r\n")
        cl.send("Content-Length: {}\r\n\r\n".format(len(last_snapshot)))
        cl.send(last_snapshot)
    else:
        cl.send("HTTP/1.1 204 No Content\r\n\r\n")
    cl.close()

def serve_mjpeg(cl):
    cl.send("HTTP/1.1 200 OK\r\n")
    cl.send("Content-Type: multipart/x-mixed-replace; boundary=frame\r\n")
    cl.send("\r\n")
    try:
        while True:
            buf = camera.capture()
            if buf:
                cl.send("--frame\r\n")
                cl.send("Content-Type: image/jpeg\r\n")
                cl.send("Content-Length: {}\r\n\r\n".format(len(buf)))
                cl.send(buf)
                cl.send("\r\n")
            time.sleep(0.1)
            gc.collect()
    except Exception as e:
        print("Stream error:", e)
    finally:
        cl.close()

# ===== Setup HTTP Server =====
def setup_server():
    global s
    try:
        addr = socket.getaddrinfo("0.0.0.0", PORT)[0][-1]
        s = socket.socket()
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(addr)
        s.listen(2)
        s.setblocking(False)
        print("Server ready on port", PORT)
        return True
    except Exception as e:
        print("HTTP server setup failed:", e)
        return False

# ===== Main Initialization =====
print("Starting ESP32-CAM Car Plate Recognition System...")
print("Using MQTT snap commands instead of GPIO button")

# Initialize components
wlan = connect_wifi()
if wlan:
    wifi_connected = True
    setup_server()

camera_initialized = init_camera()

if wifi_connected:
    mqtt_connected = connect_mqtt()

# ===== Main Loop =====
print("Starting main loop...")
print("Send 'capture' to car/snap topic to trigger image capture")

last_mqtt_reconnect = time.time()
last_wifi_reconnect = time.time()
last_mqtt_maintenance = time.time()

while True:
    try:
        # Regular MQTT maintenance
        if mqtt_connected and time.time() - last_mqtt_maintenance > 5:
            try:
                # Check if client is still connected
                client.publish(b"heartbeat", b"alive")
                last_mqtt_maintenance = time.time()
            except Exception as e:
                print("MQTT maintenance failed:", e)
                mqtt_connected = False
        
        # Handle MQTT messages (including snap commands)
        if mqtt_connected:
            try:
                client.check_msg()
            except Exception as e:
                print("MQTT check error:", e)
                mqtt_connected = False
        
        # Reconnect WiFi if needed
        if not wifi_connected and time.time() - last_wifi_reconnect > 30:
            print("Attempting WiFi reconnection...")
            wlan = connect_wifi()
            if wlan:
                wifi_connected = True
                setup_server()
            last_wifi_reconnect = time.time()
        
        # Reconnect MQTT if needed
        if wifi_connected and (not mqtt_connected or time.time() - last_mqtt_reconnect > 30):
            print("Attempting MQTT reconnection...")
            mqtt_connected = connect_mqtt()
            last_mqtt_reconnect = time.time()
        
        # Reinitialize camera if needed
        if not camera_initialized and time.time() - last_wifi_reconnect > 10:
            print("Attempting camera reinitialization...")
            camera_initialized = init_camera()
        
        # Handle HTTP clients
        if wifi_connected and s:
            try:
                cl, addr = s.accept()
                cl.setblocking(True)
                req = cl.recv(1024).decode()
                
                if "GET /video" in req:
                    print("Starting video stream for", addr)
                    serve_mjpeg(cl)
                elif "GET /pic" in req:
                    print("Serving single image to", addr)
                    serve_single_image(cl)
                else:
                    cl.send("HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n")
                    html = """<html>
                        <head>
                            <title>ESP32-CAM Car Plate Recognition</title>
                            <meta name="viewport" content="width=device-width, initial-scale=1">
                        </head>
                        <body>
                            <h1>ESP32-CAM Car Plate Recognition</h1>
                            <p><a href='/video'>Live Video Stream</a></p>
                            <p><a href='/pic'>Latest Snapshot</a></p>
                            <p>Use MQTT to control the system:</p>
                            <ul>
                                <li>Send 'capture' to <b>car/snap</b> to take a picture</li>
                                <li>Send 'open' to <b>car/gate</b> to open gate (flash on for 3s)</li>
                                <li>Send 'deny' to <b>car/gate</b> to deny access (triple flash)</li>
                            </ul>
                        </body>
                    </html>"""
                    cl.send(html)
                    cl.close()
            except OSError as e:
                if e.args[0] != 11:  # Ignore EAGAIN/EWOULDBLOCK errors
                    print("HTTP accept error:", e)
            except Exception as e:
                print("HTTP error:", e)
                try:
                    cl.close()
                except:
                    pass
        
        # Small delay to prevent busy waiting
        time.sleep(0.1)
            
    except Exception as e:
        print("Main loop error:", e)
        time.sleep(5)  # Longer sleep on critical errors
