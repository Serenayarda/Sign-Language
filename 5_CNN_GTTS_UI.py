
import tkinter as tk
from PIL import ImageTk, Image

def anaprogram():
    import cv2
    import numpy as np         
    import os                 
    import cnn_sgn
    import imutils
    import time
    from gtts import gTTS 
    import playsound
    
    IMG_SIZE = 96
    LR = 1e-3 #10^-3
    nb_classes=27
    MODEL_NAME = 'handsign.model'
    model=cnn_sgn.cnn_model()
    
    if os.path.exists('{}.meta'.format(MODEL_NAME)):
        model.load(MODEL_NAME)
        print('model loaded!')
    
               #0   1    2     3       4    5    6    7    8    9    10   11   12   13   14   15   16   17   18   19   20   21    22   23   24   25   26   
    out_label=['A', 'B', '<', 'BLANK', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'OK', 'P', 'R', 'S', ' ', 'T', 'U', 'V', 'Y', 'Z', ]
    pre=[]
    
    aWeight = 0.5
    camera = cv2.VideoCapture(0)
    top, right, bottom, left = 150, 150, 405, 450
    num_frames = 0
    harf=""
    
    while(True):
        
        (grabbed, frame) = camera.read() #anlık kare alınıyor
        frame = imutils.resize(frame, width=700) #çerçeveyi yeniden boyutlandır
        frame = cv2.flip(frame, 1) #kamera görüntü alırken ayna görüntüsüne çeviriliyor
        clone = frame.copy()  #Ekrana harfleri yazdırmak için gerekli olan img için o an kopyalanan frame'i clone değişkeni olarak alıyoruz
        (height, width) = frame.shape[:2] #çerçevenin yüksekliğini ve genişliğini alın
        roi = frame[top:bottom, right:left]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        img=gray
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        test_data =img
        orig = img
        data = img.reshape(IMG_SIZE,IMG_SIZE,1)
        model_out = model.predict([data])[0]
        pnb=np.argmax(model_out)
        print(str(np.argmax(model_out))+" "+str(out_label[pnb]))
        pre.append(out_label[pnb]) 
        #yeşil çerçeveni köşesine yazılan harf
        cv2.putText(clone, (str(out_label[pnb])),
               (450, 150), cv2.FONT_HERSHEY_PLAIN,4,(0, 0, 255),2)      
        #cv.putText(çizim yapılacak image, ekrana yazılacak metin, yazılacak metnin başlayacağı koordinat, font, font büyüklüğü,yazı rengi,yazı kalınlığı )
        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)
        # ekrana yazılan harflerin yeri
        cv2.putText(clone, (harf),
                       (10, 80), cv2.FONT_HERSHEY_DUPLEX,2,(0,255,255), 3)
    
        num_frames += 1
        cv2.imshow("Video feed", clone)
        keypress = cv2.waitKey(1) & 0xFF  
        if str(out_label[pnb])== 'BLANK' :  
            keypress = cv2.waitKey(1) & 0xFF  
            BLANK = ""
            time.sleep(3)
            (grabbed, frame) = camera.read()
            frame = imutils.resize(frame, width=700)        
            frame = cv2.flip(frame, 1)        
            clone = frame.copy()    
            (height, width) = frame.shape[:2]       
            roi = frame[top:bottom, right:left]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (7, 7), 0)       
            img=gray       
            img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
            test_data =img
            orig = img
            data = img.reshape(IMG_SIZE,IMG_SIZE,1)       
            model_out = model.predict([data])[0]    
            pnb=np.argmax(model_out)
            print(str(out_label[pnb]))
            
            if not (pnb == 2 or pnb == 3 or pnb == 17 or pnb == 21):
                harf=harf + out_label[pnb] 
                
            num_frames += 1
    
            cv2.imshow("Video feed", clone)
    
            if pnb==21: harf += ' '
            if str(out_label[pnb])=='<':
                harf = harf[:-1]
                
            if str(out_label[pnb])== 'OK' :
                
                text = str(harf)
                mytext = text
                
                language = 'tr'
                
                myobj = gTTS(text=mytext, lang=language, slow=False) 
                
                myobj.save("welcome.mp3") 
                
                playsound.playsound('welcome.mp3')
                
                os.remove("welcome.mp3")
                
                harf=' '
    
    camera.release()
    cv2.destroyAllWindows()
    
#User Interface
root = tk.Tk()
root.title("ENGELSİZ İLETİŞİM")
root.geometry("700x500+500+250")
root.resizable(False, False)
#root.config(bg='pink')
#root.iconbitmap('C:/Users/ACER/Desktop/sign_language/train_data')

my_image = ImageTk.PhotoImage(Image.open("Resim1.jpg"))
L2 = tk.Label(image = my_image)
L2.pack()

L1 = tk.Label(root, text="Engelsiz İletişim Arayüzüne Hoşgeldiniz" , fg= 'black', font='calibri 15 bold italic')
L1.pack()
L1.place(x=190, y=75)

def myclick():
    root2= tk.Tk()   
    root2.title("Engelsiz İletişim Nedir?")
    root2.geometry("550x300+590+370")
    root2.config(bg='white')
    L2 = tk.Label(root2, text =
                  """               Engelsiz iletişim, işitme engelli bireylerin çevresindeki diğer
                 
                    insanlarla iletişimlerini kolaylaştırmak amacıyla düşünülmüş
                  
                    bir projedir. El işaretleri ile konuşan işitme engelli bireyin
                  
                    söyledikleri, kamera aracılığıyla yazıya çevrilecek ve bu yazı
                  
                    da metin seslendirme yöntemi ile sese dönüştürülecektir.
                  
                    Böylelikle işitme engelli birey ile karşısındaki kişi 
                  
                    arasında 'engelsiz bir iletişim' kurulması amaçlanmıştır.""",bg= 'white',wraplength=500, padx=50, pady=100)
    L2.pack()
    
def myclick2():
    root3= tk.Tk()
    root3.title("Nasıl Kullanılır?")
    root3.geometry("700x400+500+300")
    root3.config(bg='white')
    L3= tk.Label(root3, text= """Engelsiz iletişim ara yüzünde uygulama çok basittir. Sen de aşağıdaki adımları takip ederek işaretlerini
                sese çevirebilir ve ‘Engelsiz İletişim’in bir parçası olabilirsin. 
                
                1. Açılan kamera penceresindeki yeşil çerçevenin tamamı siyah olacak şekilde konumunu ayarla.
                
                2. Kamera siyah ekranı algıladıktan sonraki 3 saniye içerisinde yapacağın harfi yeşil çerçeve içinde kalacak şekilde yap.
               
                3. Harfin ekrana doğru bir şekilde yazıldıysa madde 2’ye geri dön ve bunu sıradaki harf için tekrarla.
                
                4. Harfin ekrana doğru bir şekilde yazılmadıysa madde 2’ye dön ve ardından ‘SİLME’ karakterini kameraya okut.
                
                5. Oluşturmak istediğin kelime bittiyse boşluk bırakmak için ‘BOŞLUK’ karakterini kameraya okut ve kelimeni oluştur.
                
                6. Oluşturmak istediğin cümleyi tamamladıysan madde 2’den sonra ‘OKUT’ karakterini 
                
                kameraya okut ve oluşturduğun cümlenin ses karşılığını dinle.
                
                TEBRİKLER ARTIK SEN DE “ENGELSİZ İLETİŞİM” İN BİR PARÇASISIN :)""" , bg= 'white', wraplength=700, padx= 50, pady= 50)
    L3.pack()
    
    
    
    
    
b1 = tk.Button(root, text= "Engelsiz İletişim Nedir?", font='calibri 10 bold italic',padx=19, command= myclick)
b2 = tk.Button(root, text= "Nasıl Kullanılır?", font='calibri 10 bold italic',padx=40, command= myclick2)
b3 = tk.Button(root, text= "Uygulamayı Başlat", font='calibri 10 bold italic', command = anaprogram,padx=30)
b1.pack()
b2.pack()
b3.pack()
b1.place(x=260, y=175)
b2.place(x=260,y= 215)
b3.place(x=260, y=255)






root.mainloop()







