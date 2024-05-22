import cv2 
import numpy as np
import urllib.request

url = "http://192.168.184.214/cam-mid.jpg"

template = cv2.imread('1t.png', 0)
w, h = template.shape[::-1]
tech = cv2.TM_CCOEFF_NORMED

img_resp = urllib.request.urlopen(url)
imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
img_rgb = cv2.imdecode(imgnp, -1)

while True:
    img_resp = urllib.request.urlopen(url)
    imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
    img_rgb = cv2.imdecode(imgnp, -1)

    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(img_gray, template, tech)
    threshold = 0.75
    
    loc = np.where(res >= threshold)
   
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
        cv2.putText(img_rgb, 'Leaf Disease Detected', (pt[0], pt[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow('res.png', img_rgb)
    k = cv2.waitKey(100)
    
    if k == ord('t'):
        cv2.imwrite("2k.jpeg", img_rgb)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
