from PIL import Image
from PIL import ImageFilter
import numpy as np
from sklearn import datasets,svm, metrics
from sklearn.datasets import load_digits

count = -1
digits = datasets.load_digits()
images_and_labels = list(zip(digits.images, digits.target))
n_samples = len(digits.images)
train_data = digits.images.reshape((n_samples, -1))
classifier = svm.SVC(gamma=0.001)
#classifier.fit(train_data[:n_samples],digits.target[:n_samples])
imagelist=[]
while (count < 9):
    xlist=[]
    ylist=[]
    count=count+1
    img=Image.open(str(count)+'.png')
    pixelmap=img.load()
    imglist=[]
    output=[]
    pixellist=[]

    for x in range(img.size[0]):
        for y in range(img.size[1]):
            pixellist=[]
            pixellist.append(pixelmap[x,y])
          
            
            if pixellist[0][0] + 15 <= pixellist[0][2] & pixellist[0][1] + 15 <= pixellist[0][2]:
                pixelmap[x,y]=(0,0,0)
                xlist.append(x)
                ylist.append(y)
                
            else:
               
                pixelmap[x,y]=(255,255,255)

   
    
    img1 = img.crop((min(xlist),min(ylist),max(xlist),max(ylist))) #left,upper,right,lower
    temp= Image.new( 'L', (max(xlist)-min(xlist),max(ylist)-min(ylist)), "white") # create a new white image
    temp.paste(img1)
    resized_img = temp.resize((16,16))
    converted_img=resized_img.convert('L')
    filtered_img = converted_img.filter(ImageFilter.Kernel((3,3), [0.0625, 0.125, 0.0625, 0.125, 0.25, 0.125, 0.0625, 0.125, 0.0625]))
    p = filtered_img.load()     #step 4 
    x = [[imglist for i in range(8)] for j in range(8)]  #8x8 two dimension list
  
    for i in range(0,len(x)):
        
        for j in range(0,len(x)):
                     
                       x[j][i]=((int(p[2*i,2*j])+int(p[2*i+1,2*j])+int(p[2*i,2*j+1])+int(p[2*i+1,2*j+1]))/4)
    
    #step 6

    for i in range(0,len(x)):
        for j in range(0,len(x)):
            x[j][i] = (256 - x[j][i])/16
        
    
    #Now convert the 2-Dimensional list into 1-Dimensional list
    
    data  = np.reshape(x,-1)
    imagelist.append(data)
a=[]
for i in range(10):
    a.append(i)

classifier.fit(imagelist,a)
print(classifier.predict(imagelist))
    

    
