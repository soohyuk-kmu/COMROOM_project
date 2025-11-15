import time
import board
import adafruit_dht
import RPi.GPIO as GPIO

LED_PIN = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(LED_PIN, GPIO.OUT)


dht_sensor = adafruit_dht.DHT22(board.D4, use_pulseio=False)

while True:
    try:
        temp = dht_sensor.temperature
        hum = dht_sensor.humidity
        
        if hum is not None and temp is not None:
            print(f"Temp: {temp:.1f}, Hum:{hum:.1f}%")

            if hum >= 70:
                GPIO.output(LED_PIN, GPIO.HIGH)
                print("high huminity!! LED ON!!")
            else:
                GPIO.output(LED_PIN, GPIO.LOW)


        time.sleep(2)
    except RuntimeError as e:
        print(f"Error: {e}")
    
    except KeyboardInterrupt:
        GPIO.output(LED_PIN, GPIO.LOW)
        GPIO.cleanup()
        break
