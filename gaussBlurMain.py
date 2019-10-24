from PIL import Image
import numpy as np
import cv2 as cv

def writeMatToFile(img, outputFile):
	cv_file = cv.FileStorage(outputFile, cv.FILE_STORAGE_WRITE)
	print("write img\n")
	cv_file.write("my_matrix", img)
	# note you *release* you don't close() a FileStorage object
	cv_file.release()

def generateGaussianKernel():
	gaussList = []
	gaussArr = np.array([[1,4,7,4,1],[4,16,26,16,4],[7,26,41,26,7],[4,16,26,16,4],[1,4,7,4,1]])
	gaussDescend = 273
	gaussList.append(gaussArr)
	gaussList.append(gaussDescend)
	return gaussList

def setValueInPixelOfImageWithKernel(img, x, y, gaussKernel):
	val:int = 1

	return val 
def blurImageWithGaussianKernelWithType(img, gaussianKernel, gaussType):
	# Create an white image
	height, width = img.shape
	print img.shape
	#print(s+"{}".format(i))
	print("width = " + "{}".format(width) + " , height = " + "{}".format(height))
	imgTemp = np.zeros([height,width,3],dtype=np.uint8)
	imgTemp.fill(255) # or img[:] = 255
	if gaussType == 1:
		# Do blur
		print "type = " + "{}".format(gaussType)
		
		setValueInPixelOfImageWithKernel(img, x, y, gaussKernel);
	else: 
		print "Not my type"
	return imgTemp

def main(argv):
	fileName = 'flower_500x500.jpg'
	# Load a color image in grayscale
	img = cv.imread(fileName,0)
	#write Image matric to file
	writeMatToFile(img, "flowerImg.xml")
	
	gaussKernel = generateGaussianKernel()
	
	print gaussKernel
	
	blurImageWithGaussianKernelWithType(img, gaussKernel, 1)
	
	cv.imshow("Image", img)

	cv.waitKey(0)
	cv.destroyAllWindows()
	print "main Done!"


main("test")

