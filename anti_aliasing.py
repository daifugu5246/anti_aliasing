import cv2
import numpy as np

a = cv2.imread('barbara.tif', cv2.IMREAD_GRAYSCALE)

# Resize Image
b = cv2.resize(a, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
c = cv2.resize(b, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)

cv2.imwrite('resized_image.png', c)
cv2.imshow('Resized Image', c / np.max(c))

H = np.zeros((512, 512))
H[256-64:256+64, 256-64:256+64] = 1

# FFT
Da = np.fft.fft2(a)
Da = np.fft.fftshift(Da)
Dd = Da * H

# inverse FFT
Dd = np.fft.ifftshift(Dd)
d = np.real(np.fft.ifft2(Dd))

# result
cv2.imwrite('filtered_image.png', d)
cv2.imshow('Filtered Image', d / np.max(d))
cv2.waitKey(0)
cv2.destroyAllWindows()