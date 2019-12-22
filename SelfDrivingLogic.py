




import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
casc_path = 'CarCascade.xml'
car_cascade = cv2.CascadeClassifier(casc_path)



def make_coord(frame, line_parameters):
    slope, intercept = line_parameters       
    y1 = frame.shape[0]
    y2 = int(y1*(4.5/10))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])
    
def display_lines(frame, lines):    
    line_image = np.zeros_like(frame)
    
    if lines is not None:
         for  x1, y1, x2, y2 in lines:
             pts = np.array([[x1, y1 ], [x2 , y2]], np.int32)
             cv2.polylines(line_image, [pts], True, (0,0,255), 10)
    return line_image
           

def canny(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny
            
def roi(frame):
    height = frame.shape[0]
    polygons = np.array([[(0, 500 ), (0, height), (1500, height), (1000, 250), (700, 250)]])    
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, polygons, (255, 255, 255))
    masked_image = cv2.bitwise_and(frame, mask)
    return masked_image
    
    
def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)                
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        elif slope > 0:
            right_fit.append((slope, intercept))
    if left_fit is not None:
        left_fit_average = np.average(left_fit, axis=0)
        right_fit_average = np.average(right_fit, axis=0)
        left_line = make_coord(image, left_fit_average)
        right_line = make_coord(image, right_fit_average)
        return np.array([left_line, right_line])
    

device = 0
try:
    device = int(sys.argv[1])
except IndexError:
    pass
    
cap = cv2.VideoCapture('GTATestVideo.mov')
while cap.isOpened():
    ret, frame = cap.read()
    h, w, c = frame.shape

    canny_image = canny(frame)
    cropped_image = roi(canny_image)
    retr, thresh = cv2.threshold(cropped_image,127,255,0)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    copy = np.zeros_like(frame)
    cntImage = np.zeros_like(frame)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        approx = cv2.approxPolyDP(cnt, 0.001*cv2.arcLength(cnt, True), True)
        x = approx.ravel()[0]
        y = approx.ravel()[1]
        cv2.polylines(cntImage, [approx], True, (0, 0, 255), 2)
    cntFinal = copy | cntImage
    cannyCntFinal = cv2.Canny(cntFinal, 50, 150)
    lines = cv2.HoughLinesP(cannyCntFinal,cv2.HOUGH_PROBABILISTIC, np.pi/180, 200, minLineLength=50,maxLineGap=45)
    try:
        averaged_lines = average_slope_intercept(frame, lines)
    except TypeError:
        continue
    
    line_image = display_lines(frame, averaged_lines)
    comboImage = frame | line_image

    for leftLine in averaged_lines:
        left_lane = []
        x1L, y1L, x2L, y2L = leftLine.reshape(4)

    for line in averaged_lines:
        x1, y1, x2, y2 = line.reshape(4)
        slope = (y2-y1)/(x2-x1)
                    
        left_lane = []
        right_lane = []
        
        if x1 < 400:
            left_lane.append(line)
            
        if x1 > 400:
            right_lane.append(line) 
                          
        for x1R, y1R, x2R, y2R in right_lane:
            right_P = x1R, y1R, x2R, y2R = line.reshape(4)
                        
        for x1L, y1L, x2L, y2L in left_lane:
            left_P = x1L, y1L, x2L, y2L = line.reshape(4)
            
            
    try: cv2.line(comboImage, (x1R, y1R), (x2R, y2R), (0, 0, 255), 5)
    except OverflowError:
        contiune
    try:
        cv2.line(comboImage, (x1L, y1L), (x2L, y2L), (0, 0, 255), 5)
    except OverflowError:
        continue
        
    try:
        bottom_guide = cv2.line(comboImage, (x1R, y1R), (x1L, y1L), (0, 0, 255), 5)
    except OverflowError:
        continue
        
    try:
        top_guide= cv2.line(comboImage, (x2R, y2R), (x2L, y2L), (0, 0, 255), 5)
    except OverflowError:
        continue    
        
        
    font = cv2.FONT_HERSHEY_COMPLEX
    SteeringLine = cv2.line(comboImage, (int(w*.5), int(h)), (int(w*.5), int(h*.5)), (0, 255, 0), 5)
    RoadPts = np.array([[x1L, y1L], [x2L , y2L], [x2R, y2R], [x1R, y1R]], np.int32)
    lane = cv2.polylines(comboImage, [RoadPts], True, (255, 0, 0), 10)
    bottomLength = x1R - x1L
    B = ((x2R-x2L)/2)+x2L
    Subtract = B - 640
    Add = 640 - B
    if B > 640:
        cv2.putText(comboImage, "Turn Right " + str(Subtract) + " degrees", (150, 600), font, 1, (0, 255, 0), 4)

    elif B < 640:
        cv2.putText(comboImage, "Turn Left " + str(Add) + " degrees", (800, 600), font, 1, (0, 255, 0), 4)

   
    GrayFrame = np.zeros_like(frame)
    GrayFrame = cv2.cvtColor(GrayFrame, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(
        frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100),
        flags=cv2.CASCADE_SCALE_IMAGE)        
    for (x,y,w,h) in cars:
        BrakingDifference = y-400
        BrakingPercentage = ((BrakingDifference/400)*100)*-1
        cv2.rectangle(comboImage,(x,y),(x+w,y+h),(0,0,255),5)
        
        if y < 600:
            DistanceInFeet = (600 - y)*0.25            
        VehicleText = cv2.putText(comboImage, "VEHICLE " + str(DistanceInFeet) + "Ft.", (x, y), font, 1, (0, 0, 255), 3)        
        if DistanceInFeet < 100:
            BrakingPercentage = 100 - DistanceInFeet
            cv2.putText(comboImage, "Brake " + str(BrakingPercentage) + "%", (500, 100), font, 2, (0, 0, 255), 3)
            
                                                        
    
    
    
    
    
    
    
    
    
    
    
    cv2.imshow('TEST', comboImage)
  
