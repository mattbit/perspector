import cv2
import glob
import logging
import numpy as np
import matplotlib.pyplot as plt


def preview(img, cmap=None):
    """Preview the image using pyplot."""
    if not cmap:
        # Convert BGR to RGB
        img = img[:,:,::-1]

    plt.imshow(img, cmap=cmap)
    plt.xticks([])
    plt.yticks([])
    plt.show()


class Perspector(object):
    def __init__(self, image):
        self.original = cv2.imread(image)
        self.scale = self.original.shape[0] / 600

        img = self.original.copy()
        self.img = cv2.resize(img, (img.shape[1]/self.scale, img.shape[0]/self.scale))


    def _detect_edges(self):
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (7, 7), 0)
        cv2.convertScaleAbs(img, img, 2.5, -200)
        kernel = np.ones((5, 5), np.uint8)
        img = cv2.dilate(img, kernel, iterations=2)
        img = cv2.erode(img, kernel, iterations=2)

        return cv2.Canny(img, 0, 200)


    def _find_outline(self):
        edges = self._detect_edges()


        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)  # sort by area

        min_contour_area = edges.shape[0]*edges.shape[1]/5

        for c in contours:
            p = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02*p, True)

            if len(approx) == 4 and cv2.contourArea(approx) > min_contour_area:
                return approx

        raise Exception("Unable to find the outline.")


    def outline(self):
        contour = self._find_outline()
        corners = contour.reshape(4, 2)

        img = self.img.copy()
        cv2.drawContours(img, [contour], -1, (0, 0, 255), 5)

        return img


    def transform(self, adjust=True):
        contour = self._find_outline()
        corners = contour.reshape(4, 2)

        transformed = self._perspective_transform(self.original, corners*self.scale)

        if adjust:
            return self._adjust(transformed)

        return transformed



    def _adjust(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist, bins = np.histogram(img.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf = cdf * hist.max()/cdf.max()

        lb = np.argmax(cdf >= np.percentile(cdf, 15))
        ub = np.argmax(cdf >= np.percentile(cdf, 75))

        alpha = 255./(ub-lb)
        beta = -alpha*np.min(img)

        cv2.convertScaleAbs(img, img, alpha, beta)

        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


    def _sort_corners(self, points):
        """Sort the corners from top left in clockwise direction.
        """
    	corners = np.zeros((4, 2), dtype='float32')

    	s = np.sum(points, axis=1)
    	corners[0] = points[np.argmin(s)]
    	corners[2] = points[np.argmax(s)]

    	diff = np.diff(points, axis=1)
    	corners[1] = points[np.argmin(diff)]
    	corners[3] = points[np.argmax(diff)]

    	return corners


    def _perspective_transform(self, img, corners):
        corners = self._sort_corners(corners)
        tl, tr, br, bl = corners

        width = int(min(self._distance(tl, tr), self._distance(bl, br)))
        height = int(min(self._distance(tl, bl), self._distance(tr, br)))

        dest = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]],
                        dtype='float32')

        M = cv2.getPerspectiveTransform(corners, dest)

        return cv2.warpPerspective(img, M, (width, height))


    def _distance(self, pointA, pointB):
        dx = pointA[0] - pointB[0]
        dy = pointA[1] - pointB[1]

        return np.sqrt(dx**2 + dy**2)


    def write(self, dest):
        cv2.imwrite(dest, self.transform())


    def write_original(self, dest):
        img = self._adjust(self.original)
        cv2.imwrite(dest, img)
