from PIL import Image
import numpy as np
import cv2 as cv
import scipy.stats as st
# ~ import pyplot as plt
import threading as td

class BlurThread (td.Thread):
	def __init__ (self, threadID, name, inputImg, imgHeight, imgWidth, kernelArr, kernelX, outputImg):
		td.Thread.__init__(self)
		self.inputImg = inputImg
		self.imgHeight = imgHeight
		self.imgWidth = imgWidth
		self.kernelArr = kernelArr
		self.outputImg = outputImg
		self.name = name
		self.threadID = threadID
		self.kernelX = kernelX

	def run(self):
		print "Starting" + self.name
		
		for i in range(0, imgHeight):
			for j in range(0, imgWidth):
				val = pixelOfImageWithKernel(img, imgHeight, imgWidth, i, j, kernelArr, kernelX)
				outputImg[i][j] += val
				
	def pixelOfImageWithKernel(img, height, width, x, y, kernelArr, kernelX):
		total = 0;
		for j in range(-2, 3):
			a = x + kernelX
			if a < 0:
				a = 0
			elif a > height - 1:
				a = height - 1
			b = y + j
			if b < 0:
				b = 0
			elif b > width - 1:
				b = width - 1
			tmp = img[a][b]
			total += tmp * kernelArr [kernelX + 2][j + 2]

def startBlurThreadWithKernel(img, kernel):
	height, width = img.shape
	print img.shape
	#print(s+"{}".format(i))
	print("width = " + "{}".format(width) + " , height = " + "{}".format(height))
	dst = np.zeros([imgHeight,imgWidth,1],dtype=np.uint8)
	dst.fill(255) # or img[:] = 255
	threads = []
	thread1 = BlurThread(1, "blur-1", img, height, width, kernel, -2, dst)
	thread2 = BlurThread(2, "blur-2", img, height, width, kernel, -1, dst)
	thread3 = BlurThread(3, "blur-3", img, height, width, kernel,  0, dst)
	thread4 = BlurThread(4, "blur-4", img, height, width, kernel,  1, dst)
	thread5 = BlurThread(5, "blur-5", img, height, width, kernel,  2, dst)
	
		

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

def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel."""

    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()
    
def setValueInPixelOfImageWithKernel(img, height, width, x, y, gaussKernel, gaussType):
	val = 0
	total = 0
	for i in range(-2,3):
		for j in range(-2,3):
			a = x + i
			if a < 0:
				a = 0
			elif a > height - 1:
				a = height - 1
			b = y + j
			if b < 0:
				b = 0
			elif b > width - 1:
				b = width - 1
			tmp = img[a][b]
			if gaussType == 1:
				total += tmp * gaussKernel[0][i+2][j+2]
			elif gaussType == 2:
				total += tmp * gaussKernel[i+2][j+2]

	before = total
	if gaussType == 1:
		total /= gaussKernel[1]
	elif gaussType == 2:
		total = before
	# ~ if total < 255:
		# ~ print "total before " + "{}".format(before)
		# ~ print "total after " + "{}".format(total)
	return total
def blurImageWithGaussianKernelWithType(img, gaussianKernel, gaussType):
	# Create an white image
	if img is None:
		print "Wrong input Image"
		return None
	else: 
		height, width = img.shape
		print img.shape
		#print(s+"{}".format(i))
		print("width = " + "{}".format(width) + " , height = " + "{}".format(height))
		imgTemp = np.zeros([height,width,1],dtype=np.uint8)
		imgTemp.fill(255) # or img[:] = 255
		if gaussType == 1:
			# Do blur
			print "type = " + "{}".format(gaussType)
			for i in range(0, width - 1):
				for j in range(0, height - 1):
					val = setValueInPixelOfImageWithKernel(img, height, width, i, j, gaussianKernel,1);
					imgTemp[i][j] = val
					# ~ print "val = " + "{}".format(val)
					# ~ print "img2["+"{}".format(i)+"]["+"{}".format(j)+ "] = " + "{}".format(imgTemp[i][j])
		elif gaussType == 2:
			# Do blur
			print "type = " + "{}".format(gaussType)
			for i in range(0, width - 1):
				for j in range(0, height - 1):
					val = setValueInPixelOfImageWithKernel(img, height, width, i, j, gaussianKernel,2);
					imgTemp[i][j] = val
					# ~ print "val = " + "{}".format(val)
					# ~ print "img2["+"{}".format(i)+"]["+"{}".format(j)+ "] = " + "{}".format(imgTemp[i][j])
		else:
			print "Not my type"
		return imgTemp

def test():
	img = cv.imread('CCTV-icon.png',0)
	kernel = np.ones((5,5),np.float32)/25
	gaussKernel_1 = np.array([[1,4,7,4,1],[4,16,26,16,4],[7,26,41,26,7],[4,16,26,16,4],[1,4,7,4,1]]) / 273
	dst = cv.filter2D(img,-1,gaussKernel_1)
	cv.imshow("Image", img)
	cv.imshow("dst", dst)
	cv.waitKey(0)
	cv.destroyAllWindows()
	# ~ plt.subplot(121),plt.imshow(img),plt.title('Original')
	# ~ plt.xticks([]), plt.yticks([])
	# ~ plt.subplot(122),plt.imshow(dst),plt.title('Averaging')
	# ~ plt.xticks([]), plt.yticks([])
	# ~ plt.show()

def main(argv):
	fileName = 'flower_500x500.jpg'
	# Load a color image in grayscale
	img = cv.imread(fileName,0)
	if img is None:
		print "Cannot open Image file!"
	else:
		#write Image matric to file
		# ~ writeMatToFile(img, "flowerImg.xml")
		gaussKernel_1 = np.array([[1,4,7,4,1],[4,16,26,16,4],[7,26,41,26,7],[4,16,26,16,4],[1,4,7,4,1]]) / 273
		gauss2 = gkern(5, 2.5)
		print gauss2
		gaussKernel = generateGaussianKernel()
		
		print gaussKernel
		
		img2 = blurImageWithGaussianKernelWithType(img, gauss2, 2)
		img3 = blurImageWithGaussianKernelWithType(img, gaussKernel, 1)
		# ~ dst = cv.GaussianBlur(img,(5,5),cv.BORDER_DEFAULT)
		# ~ print "img2 = " + "{}".format(img2)
		cv.imshow("Image", img)
		cv.imshow("Image2", img2)
		cv.imshow("Image3", img3)
		# ~ cv.imshow("DST", dst)
		cv.waitKey(0)
		cv.destroyAllWindows()
		# ~ writeMatToFile(img, "img_Input.xml")
		# ~ writeMatToFile(img2, "img_Output.xml")
		# ~ writeMatToFile(dst, "img_OutputGauss.xml")
		print "main Done!"

#test()
main("test")

