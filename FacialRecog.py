import cv2
import os
import numpy as np
from PIL import Image

#顔認識クラス（撮影/訓練/認識）
class FaceRecog:
    
    def __init__(self,haar_dir='./'):        
        self.face_detector = cv2.CascadeClassifier(haar_dir+'haarcascade_frontalface_default.xml')
        self.ids = []#selfにする必要は低い→グローバル変数にした方がいいかも
    # 顔写真撮影クラス　引数　

    #cap_dir:撮影写真保存用
    #take_pic:撮影写真の枚数 defo=30
    #scaleFactor=1.2
    #minNeighbors=5
    #minSize=(100, 100)
    
    def capture(self,cap_dir='./',take_pic=30,scaleFactor=1.2, minNeighbors=5, minSize=(100, 100)):
        cam =  cv2.VideoCapture(0)   
        cam.set(3, 640) # set video width
        cam.set(4, 480) # set video height
        # For each person, enter one numeric face id
        face_id = input('\n enter user id end press <return> ==>  ')
        name = input('\n enter user name end press <return> ==>')
        print("\n [INFO] Initializing face capture. Look the camera and wait ...")
        # Initialize individual sampling face count
        count = 0

        while(True):
            ret, img = cam.read()
            if not ret:
                break
            img = cv2.flip(img, -1) # flip video image vertically
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(gray, 1.1)

            for (x,y,w,h) in faces:
                cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
                count += 1

                # Save the captured image into the datasets folder
                cv2.imwrite(cap_dir+'/' +str(name)+'.'+ str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

            cv2.imshow('image', img)

            k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
            if k == 27:
                break
            elif count >= take_pic: # Take 30 face sample and stop video
                break

        # Do a bit of cleanup
        print("\n [INFO] Exiting Program and cleanup stuff")
        cam.release()
        cv2.destroyAllWindows()

    #顔画像の訓練をするメソッド


    #引数
    #img_path:顔写真の保存されているディレクトリ（デフォルトはカレント）
    #train_dir:訓練後のデータを保存するディレクトリ（デフォルトはカレント）
    
    
    def facetrain(self,img_path='./',train_dir='./'):#訓練させる画像のあるディレクトリ,訓練させた後のディレクトリ
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        # function to get the images and label data
        imagePaths = [os.path.join(img_path,f) for f in os.listdir(img_path)]  
        print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")    
        faceSamples=[]
        for imagePath in imagePaths:
            PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
            img_numpy = np.array(PIL_img,'uint8')
            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = self.face_detector.detectMultiScale(img_numpy)
            for (x,y,w,h) in faces:
                faceSamples.append(img_numpy[y:y+h,x:x+w])
                self.ids.append(id)#self
            faces,ids = faceSamples,self.ids#self
            recognizer.train(faces, np.array(self.ids))#self

        # Save the model into trainer/trainer.yml
        recognizer.write(train_dir+'/trainer.yml') # recognizer.save() worked on Mac, but not on Pi

        # Print the numer of faces trained and end program
        print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(self.ids))))  
    #顔認識をするメソッド

    #引数
    #train_dir:訓練データのあるディレクトリ
    #names:idに対応した名前のリスト
    #scaleFactor=1.2
    #minNeighbors=5
    #minSize=(100, 100)


    def facerecog(self,train_dir='./',names=['none'],scaleFactor=1.2, minNeighbors=5, minSize=(100, 100)):
        print(minSize)
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(train_dir+'/trainer.yml')
        font = cv2.FONT_HERSHEY_SIMPLEX
        #iniciate id counter
        id = 0
        
        # Initialize and start realtime video capture
        cam = cv2.VideoCapture(0)
        cam.set(3, 640) # set video widht
        cam.set(4, 480) # set video height

        while True:
            ret, img =cam.read()
            if not ret:
                break
            img = cv2.flip(img, -1) # Flip vertically
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
            faces = self.face_detector.detectMultiScale( 
                gray,
                scaleFactor,
                minNeighbors,
                minSize=minSize,
              )

            for(x,y,w,h) in faces:
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
                id, confidence = recognizer.predict(gray[y:y+h,x:x+w])#戻り値はidと確信度（0に近いほど正確）

                # Check if confidence is less them 100 ==> "0" is perfect match 
                if (confidence < 100):
                    id = names[id]
                    confidence = "  {0}%".format(round(100 - confidence))
                else:
                    id = "unknown"
                    confidence = "  {0}%".format(round(100 - confidence))
        
                cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
                cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
    
            cv2.imshow('camera',img) 

            k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
            if k == 27:
                break

        # Do a bit of cleanup
        print("\n [INFO] Exiting Program and cleanup stuff")
        cam.release()
        cv2.destroyAllWindows()
