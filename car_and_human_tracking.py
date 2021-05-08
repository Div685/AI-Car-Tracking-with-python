import cv2

# Our image
img_file = 'car.jpg'
# video = cv2.VideoCapture('cars1.mp4')
video = cv2.VideoCapture('ped_car_Trim1.mp4')


# Create open cv image
# img = cv2.imread(img_file)

# our pre-trained 
classifier_file = 'cars_casscade.xml'

car_tracker = cv2.CascadeClassifier(classifier_file)


# pedestrian tracker
pedestrian_file = 'haarcascade_fullbody.xml'

pedestrian_tracker = cv2.CascadeClassifier(pedestrian_file)




# Run forever until car stops
while True:
	(read_successful, frame) = video.read()

	if read_successful:
		grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	else:
		break



	cars = car_tracker.detectMultiScale(grayscaled_frame)
	pedestrian = pedestrian_tracker.detectMultiScale(grayscaled_frame) 


	for(x, y, w,h) in cars:
		cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)


	for(x, y, w,h) in pedestrian:
		cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 255), 2)


	# print(cars)

	cv2.imshow('Car Detector', frame)

	key = cv2.waitKey(1)

	#Stop If q is pressed
	if key==81 or key==113:
		break


# Release the videoCapture object
video.release()




  


# # convert to grey scale 
# black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



# # Create Class classifier
# car_tracker = cv2.CascadeClassifier(classifier_file)

# # detect cars 
# cars = car_tracker.detectMultiScale(black_n_white)


# # Draw rectangle around the car 
# # x , y , width, height (Blue, Green, Red), thickness

# for(x, y, w,h) in cars:
# 	cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)

# # car1 = cars[0]
# # (x, y, w, h) = car1
# # cv2.rectangle(img, (x, y), (x+w, y+h), (0,255, 0), 2)




# # Display image with face doted
# cv2.imshow('Car Detector', img)

# # Dont autoclose (wait here in the code and listen  a key press)
# cv2.waitKey()

print("Code Completed!!")