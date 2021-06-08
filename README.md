# License-Plate-Number-Detection
A project where the license plate number is extracted from image of a vehicle using Object detection and Character recognition techniques.

## Introduction
Automatic license plate detection has the ability to automatically identify the vehicle by capturing and recognizing the number plates of any vehicle with the help of an image, provided by video surveillance cameras.It has many practical applications like noting vehicle numbers at toll gate operation, tracing cars, finding stolen cars from CCTVs, etc.

## Approach
To predict the license plate number, the following things need to be done:
1. The license plate needs to be detected from the overall image. This can be done using object detection methods like finding contours, using You-Only-Look-Once (YOLO), etc.
2. After extracting the license plate, individual characters need to be seperated and segregated using character segmentation techniques like finding rectangular contours.
3. The last phase is performing character recognition, where the segmented characters are recognized using deep learning classifiers. We used CNN in this project as CNNs work the best with images.

## Dataset
The following datasets have been used for different purposes:
1. **For license plate detection (YOLO)**: The dataset contains approximately 4000 annotated images of cars with license plates. The dataset can be found and downloaded from here https://data.mendeley.com/datasets/nx9xbs4rgx/2
2. **For character recognition**: The dataset has about 1000 images of digits from 0-9 and alphabets from A-Z. You can find the dataset here [Character Dataset](data.zip)
3. **For testing the whole model**: The dataset contains about 200 images of cars with license plates. You can find the dataset here https://drive.google.com/file/d/1QAFdt5Mq8X6fZud7kdsjaJbJfSXrsFse/view?usp=sharing

## Technologies/Languages Used
- **Python**: This is the most sought language for implementing AI projects. The version used is python 3.6 here.
- **IDE**: We used Jupter Notebook for this project.
- **OpenCV**: OpenCV is a library of programming functions mainly aimed at real-time computer vision. It eases the work when projects are based primarily on images or videos.
- **Tensorflow**: TensorFlow is a free and open-source software library for machine learning. It can be used across a range of tasks but has a particular focus on training and inference of deep neural networks.
- **Keras**: Keras is an open-source software library that provides a Python interface for artificial neural networks. Keras acts as an interface for the TensorFlow library.
- **YOLOv3**: YOLOv3 (You Only Look Once, Version 3) is a real-time object detection algorithm that identifies specific objects in videos, live feeds, or images.
- **Scikit-Learn**: It is a free software machine learning library for the Python programming language.
- **Matplotlib**: Matplotlib is a plotting library for the Python programming language and its numerical mathematics extension NumPy.
- **Imutils**: A series of convenience functions to make basic image processing functions such as translation, rotation, resizing, skeletonization, displaying Matplotlib images, sorting contours, detecting edges, and much more easier with OpenCV.

## Methodology
### License plate Detection
We used two methods for license plate detection:
#### 1. Plate detection by Finding Contours
Firstly, the image needs to be imported and preprocesses before appyling contours:
```
image = imutils.resize(image, width=500)
img=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display the original image
fig, ax = plt.subplots(2, 2, figsize=(10,7))
ax[0,0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
ax[0,0].set_title('Original Image')

# RGB to Gray scale conversion
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ax[0,1].imshow(gray, cmap='gray')
ax[0,1].set_title('Grayscale Conversion')

# Noise removal with iterative bilateral filter(removes noise while preserving edges)
gray = cv2.bilateralFilter(gray, 11, 17, 17)
ax[1,0].imshow(gray, cmap='gray')
ax[1,0].set_title('Bilateral Filter')

# Find Edges of the grayscale image
edged = cv2.Canny(gray, 170, 200)
ax[1,1].imshow(edged, cmap='gray')
ax[1,1].set_title('Canny Edges')

fig.tight_layout()
plt.show()
```
The above code performs the following functions:
- **Resizing the image**: Each image is resized to 500px width, to ease processing in the later stages.
- **Grayscale Conversion**: The input image is in RGB format. Main purpose of this conversion is to reduce the number of colors.
- **Noise Removal**: Image noises are distortion in the image that arises due to fault in camera or result of poor visibility due to changing weather conditions. Noises are also the random variation in the intensity levels of the pixels. Noise can be of various types like Gaussian noise, Salt and pepper noise. We used iterative bilateral filter for noise removal. It provides the mechanism for noise reduction while preserving edges more effectively than median filter.
- **Binarization**: Binarization is the process of converting an image into an image with two pixels value only i.e. containing white and black pixels. Performing binarization process before detecting and extracting license plate from the image will make the task of detecting license plate easier as edges will be more clearly in binary image.

![image](https://user-images.githubusercontent.com/85444229/121041204-2acaf080-c7d0-11eb-97fd-02cf776069cf.png)

After preprocessing, our image is ready to find contours.
```
# Find contours based on Edges
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:30] #sort contours based on their area keeping minimum required area as '30' (anything smaller than this will not be considered)
NumberPlateCnt = None #we currently have no Number plate contour

# loop over our contours to find the best possible approximate contour of number plate
count = 0
for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:  # Select the contour with 4 corners
            NumberPlateCnt = approx #This is our approx Number Plate Contour
            x,y,w,h = cv2.boundingRect(c)
            ROI = img[y:y+h, x:x+w]
            break

if NumberPlateCnt is not None:
    # Drawing the selected contour on the original image
    cv2.drawContours(image, [NumberPlateCnt], -1, (0,255,0), 3)
```
Initially, all contours are found by using ```cv2.findContours()``` methods. All the contours with area less than 30 are discarded and the remaining contours are send for further processing. Each contour is approximated to form a polygon and if a contour is quadrilateral in shape (has 4 sides), then it is predicted to be the number plate and the contours are drawn using ```cv2.drawContours()``` method.

![image](https://user-images.githubusercontent.com/85444229/121042882-bbee9700-c7d1-11eb-8105-1a351f9b5b8e.png)

If the extracted license plate is tilted, it might face problems in the image segmentation phase. So, we need to straighten the image, if titled.

![image](https://user-images.githubusercontent.com/85444229/121043704-7ed6d480-c7d2-11eb-9b0b-446063ab8e19.png)

Let (left_x, left_y) and (right_x, right_y) be the bottom-left and bottom-right coordinates of the predicted license plate respectively. Then the image rotation can be performed as:
```
import math

opp=right_y-left_y
hyp=((left_x-right_x)**2+(left_y-right_y)**2)**0.5
sin=opp/hyp
theta=math.asin(sin)*57.2958

image_center = tuple(np.array(ROI.shape[1::-1]) / 2)
rot_mat = cv2.getRotationMatrix2D(image_center, theta, 1.0)
result = cv2.warpAffine(ROI, rot_mat, ROI.shape[1::-1], flags=cv2.INTER_LINEAR)

if opp>0:
    h=result.shape[0]-opp//2
else:
    h=result.shape[0]+opp//2

result=result[0:h, :]
plt.imshow(result)
plt.show()
```
In this, the angle of rotation is found by finding the sin of theta, from which the angle can be found easily. After that, the image is rotated according the the angle obtained.

![image](https://user-images.githubusercontent.com/85444229/121045588-6b783900-c7d3-11eb-91b8-3c66719dfe40.png)

#### 2. Plate detection using YOLOv3
We trained YOLOv3 on custom dataset for detection of license plate as mentioned in the 'Dataset' section.
For this, darknet was installed and set up in the system. Using YOLOv3 config files, we trained our dataset on Git which returned ‘plates.weights’ file containing the weights obtained after training. The .weights file is then imported in the program and used to detect plates. The repository for installing and using darknet can be found here https://github.com/pjreddie/darknet.
Due to constraints on size of file on github, we uploaded the ```lapi.weights``` file along with ```classes.names``` and ```darknet-yolov3.cfg``` files on the following link: https://drive.google.com/file/d/1cktcL1TXXRJ5o6CxzIuR08hPEWbb8Kkx/view?usp=sharing

Initially, import all the necessary files and set up the model in the following way:
```
# Load names of classes
classesFile = "classes.names";

# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = "darknet-yolov3.cfg";
modelWeights = "lapi.weights";

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
```

The following function ```drawPred()``` draws a bounding box in an image when attributes like class name, confidence, coordinates and the image itself is passed though it.
```
# Draw the predicted bounding box
def drawPred(classId, conf, left, top, right, bottom, frame):
    # Draw a bounding box.
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)

    label = '%.2f' % conf

    # Get the label for the class name and its confidence
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    #Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (0, 0, 255), cv2.FILLED)
    #cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine),    (255, 255, 255), cv.FILLED)
    cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2)
```
The following function ```postprocess()``` removes the bounding boxes with low confidence when the image and ouput are passed as attributes.
```
# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    
    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        #print("out.shape : ", out.shape)
        for detection in out:
            #if detection[4]>0.001:
            scores = detection[5:]
            classId = np.argmax(scores)
            #if scores[classId]>confThreshold:
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        
        # calculate bottom and right
        bottom = top + height
        right = left + width
        
        #crop the plate out
        cropped = frame[top:bottom, left:right].copy()
        # drawPred
        drawPred(classIds[i], confidences[i], left, top, right, bottom, frame)
        
    return cropped
```
The following code snippet first creates a 4D blob from the image, sets input to the network, runs the forward pass to get output of the output layers and sends the image to the function ```postprocess()``` for removing low confidence boxes and drawing the predicted box.
```
# Create a 4D blob from a frame.
blob = cv2.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

# Sets the input to the network
net.setInput(blob)

# Runs the forward pass to get output of the output layers
outs = net.forward(getOutputsNames(net))

# Remove the bounding boxes with low confidence
cropped = postprocess(frame, outs)
```
![image](https://user-images.githubusercontent.com/85444229/121055679-193b1600-c7db-11eb-9f1b-512e3883ee0f.png)

In the above image, bounding box is drawn along with the probability/confidence of the object being a license plate. In this case, the confidence is 0.98.

### Character Segmentation
This phase contains the use of two functions: ```segment_characters()``` and ```find_contours()```.
```
# Find characters in the resulting images
def segment_characters(image) :
    # Preprocess cropped license plate image
    img_lp = cv2.resize(image, (333, 75))
    img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
    _, img_binary_lp = cv2.threshold(img_gray_lp, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_binary_lp = cv2.erode(img_binary_lp, (3,3))
    img_binary_lp = cv2.dilate(img_binary_lp, (3,3))

    LP_WIDTH = img_binary_lp.shape[0]
    LP_HEIGHT = img_binary_lp.shape[1]

    # Make borders white
    img_binary_lp[0:3,:] = 255
    img_binary_lp[:,0:3] = 255
    img_binary_lp[72:75,:] = 255
    img_binary_lp[:,330:333] = 255

    # Estimations of character contours sizes of cropped license plates
    dimensions = [LP_WIDTH/6,
                       LP_WIDTH/2,
                       LP_HEIGHT/10,
                       2*LP_HEIGHT/3]
    plt.imshow(img_binary_lp, cmap='gray')
    plt.title('Contour')
    plt.show()
    cv2.imwrite('contour.jpg',img_binary_lp)

    # Get contours within cropped license plate
    char_list = find_contours(dimensions, img_binary_lp)

    return char_list
```
The above function takes in the image as input and performs the following operation on it:
- Resizes it to a dimension such that all characters seem distinct and clear.
- Convert the colored image to a gray scaled image. We do this to prepare the image for the next process.
- Now the threshold function converts the grey scaled image to a binary image i.e each pixel will now have a value of 0 or 1 where 0 corresponds to black and 1 corresponds to white. It is done by applying a threshold that has a value between 0 and 255, here the value is 200 which means in the grayscaled image for pixels having a value above 200, in the new binary image that pixel will be given a value of 1. And for pixels having value below 200, in the new binary image that pixel will be given a value of 0.
- The image is now in binary form and ready for the next process Eroding. Eroding is a simple process used for removing unwanted pixels from the object’s boundary meaning pixels that should have a value of 0 but are having a value of 1.
- The image is now clean and free of boundary noise, we will now dilate the image to fill up the absent pixels meaning pixels that should have a value of 1 but are having value 0.
- The next step now is to make the boundaries of the image white. This is to remove any out of the frame pixel in case it is present.
- Next, we define a list of dimensions that contains 4 values with which we’ll be comparing the character’s dimensions for filtering out the required characters.
- Through the above processes, we have reduced our image to a processed binary image and we are ready to pass this image for character extraction.

```
# Match contours to license plate or character template
def find_contours(dimensions, img) :

    # Find all contours in the image
    cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Retrieve potential dimensions
    lower_width = dimensions[0]
    upper_width = dimensions[1]
    lower_height = dimensions[2]
    upper_height = dimensions[3]
    
    # Check largest 5 or  15 contours for license plate or character respectively
    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]
    
    ii = cv2.imread('contour.jpg')
    
    x_cntr_list = []
    target_contours = []
    img_res = []
    for cntr in cntrs :
        # detects contour in binary image and returns the coordinates of rectangle enclosing it
        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
        
        # checking the dimensions of the contour to filter out the characters by contour's size
        if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height :
            x_cntr_list.append(intX) #stores the x coordinate of the character's contour, to used later for indexing the contours

            char_copy = np.zeros((44,24))
            # extracting each character using the enclosing rectangle's coordinates.
            char = img[intY:intY+intHeight, intX:intX+intWidth]
            char = cv2.resize(char, (20, 40))
            
            cv2.rectangle(ii, (intX,intY), (intWidth+intX, intY+intHeight), (50,21,200), 2)
            plt.imshow(ii, cmap='gray')
            plt.title('Predict Segments')

            # Make result formatted for classification: invert colors
            char = cv2.subtract(255, char)

            # Resize the image to 24x44 with black border
            char_copy[2:42, 2:22] = char
            char_copy[0:2, :] = 0
            char_copy[:, 0:2] = 0
            char_copy[42:44, :] = 0
            char_copy[:, 22:24] = 0

            img_res.append(char_copy) # List that stores the character's binary image (unsorted)
            
    # Return characters on ascending order with respect to the x-coordinate (most-left character first)
            
    plt.show()
    # arbitrary function that stores sorted list of character indeces
    indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
    img_res_copy = []
    for idx in indices:
        img_res_copy.append(img_res[idx])# stores character images according to their index
    img_res = np.array(img_res_copy)

    return img_res
```
In the above function, we will be applying some more image processing to extract the individual characters from the license plate. The steps involved will be:
- Finding all the contours in the input image. The function cv2.findContours returns all the contours it finds in the image.
- After finding all the contours we consider them one by one and calculate the dimension of their respective bounding rectangle. Now consider bounding rectangle is the smallest rectangle possible that contains the contour. All we need to do is do some parameter tuning and filter out the required rectangle containing required characters. For this, we will be performing some dimension comparison by accepting only those rectangle that have:
  1. Width in the range 0, (length of the pic)/(number of characters) and,
  2. Length in a range of (width of the pic)/2, 4*(width of the pic)/5. After this step, we should have all the characters extracted as binary images.

![image](https://user-images.githubusercontent.com/85444229/121058623-2ad1ed00-c7de-11eb-9fbc-df103300bfa8.png)

![image](https://user-images.githubusercontent.com/85444229/121058686-40dfad80-c7de-11eb-9242-800cd47b3eb0.png)

### Character Recognition
Since the data is all clean and ready, now it’s time do create a Neural Network that will be intelligent enough to recognize the characters after training. In this project, we used CNN model for character recognition.
```
model = Sequential()
model.add(Conv2D(16, (22,22), input_shape=(28, 28, 3), activation='relu', padding='same'))
model.add(Conv2D(32, (16,16), input_shape=(28, 28, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (8,8), input_shape=(28, 28, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (4,4), input_shape=(28, 28, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(36, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adam(lr=0.0001), metrics=[custom_f1score])
```
![image](https://user-images.githubusercontent.com/85444229/121059410-117d7080-c7df-11eb-9ff5-b1249c23f8e9.png)

- To keep the model simple, we’ll start by creating a sequential object.
- We will use 4 convolutional layers with 'Relu' as the activation function.
- Next, we’ll be adding a max-pooling layer with a window size of (4,4). Max pooling is a sample-based discretization process. The objective is to down-sample an input representation, reducing its dimensionality and allowing for assumptions to be made about features contained in the sub-regions binned.
- Now, we will be adding some dropout rate to take care of overfitting. Dropout is a regularization hyperparameter initialized to prevent Neural Networks from Overfitting. We have chosen a dropout rate of 0.4 meaning 60% of the node will be retained.
- Now it’s time to flatten the node data so we add a flatten layer for that. The flatten layer takes data from the previous layer and represents it in a single dimension.
The last Dense layer has 36 outputs because we have 26 alphabets(A-Z) and 10 digits(0-9) for classification. The activation used in this layer is 'softmax' because it is a multi-class classification problem.
- Finally, we will be adding 2 dense layers, one with the dimensionality of the output space as 128, activation function='ReLU' and other, our final layer with 36 outputs for categorizing the 26 alphabets (A-Z) + 10 digits (0–9) and activation function= 'softmax'

All the above parameters used in the model have been already tuned with hyperparameter tuning using Grid Search.

- For training the model, we’ll be using ImageDataGenerator class available in keras to generate some more data using image augmentation techniques like width shift, height shift.
- Width shift: Accepts a float value denoting by what fraction the image will be shifted left and right.
- Height shift: Accepts a float value denoting by what fraction the image will be shifted up and down.

```
train_datagen = ImageDataGenerator(rescale=1./255, width_shift_range=0.1, height_shift_range=0.1)
path = 'data/data'
train_generator = train_datagen.flow_from_directory(
        path+'/train',  # this is the target directory
        target_size=(28,28),  # all images will be resized to 28x28
        batch_size=1,
        class_mode='sparse')

validation_generator = train_datagen.flow_from_directory(
        path+'/val',  # this is the target directory
        target_size=(28,28),  # all images will be resized to 28x28 batch_size=1,
        class_mode='sparse')
```
After training the model using model.fit() method, the training accuracy obtained after 18 epochs is 98.43%.
![image](https://user-images.githubusercontent.com/85444229/121062049-3e7f5280-c7e2-11eb-9ce7-80cf293726fb.png)

#### Predicting the plate number
```
# Predicting the output
def fix_dimension(img): 
  new_img = np.zeros((28,28,3))
  for i in range(3):
    new_img[:,:,i] = img
  return new_img
  
def show_results():
    dic = {}
    characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i,c in enumerate(characters):
        dic[i] = c

    output = []
    for i,ch in enumerate(char): #iterating over the characters
        img_ = cv2.resize(ch, (28,28), interpolation=cv2.INTER_AREA)
        img = fix_dimension(img_)
        img = img.reshape(1,28,28,3) #preparing image for the model
        y_ = loaded_model.predict_classes(img)[0] #predicting the class
        character = dic[y_]
        output.append(character) #storing the result in a list
        
    plate_number = ''.join(output)
    
    return plate_number
```

![image](https://user-images.githubusercontent.com/85444229/121062438-b3eb2300-c7e2-11eb-9ac9-bc2dffc0c6be.png)

## Hyperparameter Tuning
Hyperparameter optimization or tuning is the problem of choosing a set of optimal hyperparameters for a learning algorithm. A hyperparameter is a parameter whose value is used to control the learning process.
We performed hyperparameter tuning in the CNN model on the parameters dropout rate, optimizer and learning rate using Grid Search.

![image](https://user-images.githubusercontent.com/85444229/121062753-1e03c800-c7e3-11eb-89c3-d1453b65869c.png)

Optimal parameters:
1. Dropout rate = 0.4
2. Learning rate = 0.0001
3. Optimizer = Adam

## Result
- Accuracy obtained using Contour method is about 60.24%.
- Accuracy obtained by using YOLOv3 is about 74.10%.
### Optimizing the result
- Since the individual accuracies are not that great, we created a hybrid model of both the license detection methods.
- The idea behind this is to send the image initially to YOLOv3 method. If the method does not return any image as license plate, the image is passed into the Contour method. The accuracy obtained using this hybrid approach is about ```90.96%```.
- Note that the accuracy obtained will be less when implemented in the opposite direction (Contour method -> YOLOv3), because the Contour method often returns any rectangular object found in the image. So, it is always better to use YOLO first.

## References
- Shrutika Saunshi, Vishal Sahani, Juhi Patil, Abhishek Yadav, Dr. Sheetal Rathi, "License Plate Recognition Using Convolutional Neural Network" in IOSR Journal of Computer Engineering (IOSR-JCE), e-ISSN: 2278-0661,p-ISSN: 2278-8727, PP 28-33
- Prof. Rupali Hande, Simran Pandita, Gaurav Marwal, Gaurav Marwal, Sivanta Beera, "Automatic Number Plate Detection System and Automating the Fine
Generation Using YOLO-v3." in International Journal of Future Generation Communication and Networking, Vol. 13, No. 1s, (2020), pp. 406-- 413
