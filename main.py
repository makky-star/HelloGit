from FacialRecog import FaceRecog
names = ['makky','none']

if __name__ == "__main__":
    face = FaceRecog()
    #face.capture('image',50)
    #face.facetrain('image',"train")
    face.facerecog('train',names)
    
