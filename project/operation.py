import cv2
import numpy as np
from skimage.feature import hog

def extract_sift(img):
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    return kp, des

def extract_orb(img):
    orb = cv2.ORB_create(nfeatures=1000)
    kp, des = orb.detectAndCompute(img, None)
    return kp, des


def extract_hog(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features, _ = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=True
    )
    return features

def selective_search(img):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchFast()
    return ss.process()[:200]

def dpm_detection(img):
    hog_d = cv2.HOGDescriptor()
    hog_d.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    rects, _ = hog_d.detectMultiScale(img, winStride=(8, 8))
    return rects

def process_image(input_path, output_path):
    img = cv2.imread(input_path)

    # Feature extraction
    sift_kp, _ = extract_sift(img)
    orb_kp, _ = extract_orb(img)
    hog_feat = extract_hog(img)

    print("SIFT keypoints:", len(sift_kp))
    print("ORB keypoints :", len(orb_kp))
    print("HOG feature length:", len(hog_feat))


    # Region proposals
    regions = selective_search(img)
    print("Selective Search regions:", len(regions))

    # DPM detection
    detections = dpm_detection(img)

    for (x, y, w, h) in detections:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imwrite(output_path, img)
