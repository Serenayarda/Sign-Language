import cv2
import os
import time
IMG_SIZE=96
top, right, bottom, left = 100, 150, 400, 450
exit_con='**'
a=''

dir0=input('Veri setini kaydetmek istediğiniz klasörün dizinini giriniz : ')
try:
    os.mkdir(dir0)
except:
    print(' ')
camera = cv2.VideoCapture(0)



(t, frame) = camera.read()  
frame = cv2.flip(frame, 1)
roi = frame[top:bottom, right:left]
 
gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)
gray = cv2.resize(gray, (IMG_SIZE,IMG_SIZE))
cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
cv2.imshow("Video Feed", frame)

keypress = cv2.waitKey(5)

while(True):
    
    a=input('Veri setini oluşturmak istediğiniz harfin etiketini girin (Çıkmak için: ** tuşlayın):')

    if a==exit_con:
        break
    dir1=str(dir0)+'/'+str(a)
    print(dir1)
    try:
        os.mkdir(dir1)
    except:
        print('Görüntüler alınıyor')
    i=0
    time.sleep(3)
    while(True):
        
        (t, frame) = camera.read()  
        frame = cv2.flip(frame, 1)
        roi = frame[top:bottom, right:left]
     
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        gray = cv2.resize(gray, (IMG_SIZE,IMG_SIZE))
        cv2.imwrite("%s/%s/%d.jpg"%(dir0,a,i),gray)
        i+=1
        print(i)
        
        if i>1000:
            break
        cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
        cv2.imshow("Video Feed", frame)
        
        keypress = cv2.waitKey(5)
        if keypress == 'q':
            break


camera.release()
cv2.destroyAllWindows()
