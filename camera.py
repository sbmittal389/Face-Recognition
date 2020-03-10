import cv2
import numpy as np

#read a video stream and display it

#camera object
cam = cv2.VideoCapture(0)
face_cascade= cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
face_data=[]
cnt=0

user_name=input("enter your name:")

while True:
    ret,frame = cam.read()

    if ret:
        #cv2.imshow("hello", frame)
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        """bright_image=frame+100
        bright_image[bright_image>255] =255
        cv2.imshow("bright_image", bright_image)"""

        faces = face_cascade.detectMultiScale(frame,1.3,5)
        #print(faces)

        if(len(faces)==0):
            #cv2.imshow("video",frame)
            continue
        
        for face in faces:
            x,y,w,h = face
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
            cv2.rectangle(gray,(x,y),(x+w,y+h),(0,255,255),2)

            face_section = frame[y-10:y+h+10, x-10:x+w+10]
            face_section = cv2.resize(face_section,(100,100))

            if cnt%10 ==0:
                print("taking picture")
                face_data.append(face_section)
                cnt +=1
            
        cv2.imshow("RGB title", frame)
        #cv2.imshow("GRAY title", gray)
        cv2.imshow("face_section", face_section)

        #new_img = np.zeros((*gray.shape,3))
        #new_img[:,:,0] = gray
        #new_img[:,:,1] = gray
        #new_img[:,:,2] =gray
        #combined=np.hstack((frame,new_img))
        #cv2.imshow("viode",frame)
        #cv2.imshow("video2",new_gray)

    else:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#save the face data in a numpy file
print("total number of faces:",cnt)

face_data=np.array(face_data)
face_data=face_data.reshape((face_data.shape[0], -1))
print(face_data.shape)

np.save("./"+user_name+".npy",face_data)

cam.release()
cv2.destroyAllWindows()
