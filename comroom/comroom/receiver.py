import socket
import time
import RPi.GPIO as GPIO

PORT = 8888
LED_PIN = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(LED_PIN, GPIO.OUT)




sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", PORT))

print(f"receiving (port {PORT})")



while True:
    try:
        data, addr = sock.recvfrom(1024)
        recv_data = data.decode()


        if recv_data == "ON":
            GPIO.output(LED_PIN, GPIO.HIGH)
            print("LED ON!")
        elif recv_data == "OFF":
            GPIO.output(LED_PIN, GPIO.LOW)
            print("LED OFF!")

        
        time.sleep(1)
    
    except RuntimeError as e:
        print(f"error: {e}")
    except KeyboardInterrupt:
        GPIO.output(LED_PIN, GPIO.LOW)
        GPIO.cleanup()
        break