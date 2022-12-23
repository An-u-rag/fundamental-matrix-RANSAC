import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
import math

def find_matching_keypoints(image1, image2):
    #Input: two images (numpy arrays)
    #Output: two lists of corresponding keypoints (numpy arrays of shape (N, 2))
    sift = cv2.SIFT_create()
    kp1, desc1 = sift.detectAndCompute(image1, None)
    kp2, desc2 = sift.detectAndCompute(image2, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc1, desc2, k=2)

    good = []
    pts1 = []
    pts2 = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    return pts1, pts2

def drawlines(img1,img2,lines,pts1,pts2):
    #img1: image on which we draw the epilines for the points in img2
    #lines: corresponding epilines
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

def Construct_Transformation_Matrix(mean_dist, tx, ty):
    # Construct translation matrix
    # Translate first and then scale
    s = np.sqrt(2) / mean_dist
    # Translate_mat = np.array([[1, 0, 0],
    #                           [0, 1, 0],
    #                           [tx, ty, 1]])
    #
    # Scale_mat = np.array([[s, 0, 0],
    #                       [0, s, 0],
    #                       [0, 0, 1]])

    # Combined Transformation Matrix
    T_mat = np.array([[s, 0, 0],
                      [0, s, 0],
                      [s*tx, s*ty, 1]])
    return T_mat.T

def FindFundamentalMatrix(pts1, pts2):
    #Input: two lists of corresponding keypoints (numpy arrays of shape (N, 2))
    #Output: fundamental matrix (numpy array of shape (3, 3))

    ctd_1 = np.mean(pts1, axis=0)
    ctd_2 = np.mean(pts2, axis=0)
    # Compute mean distance of all points to the centroids.
    distances_1 = []
    distances_2 = []
    for i in range(pts1.shape[0]):
        distances_1.append(math.dist(pts1[i], ctd_1))
        distances_2.append(math.dist(pts2[i], ctd_2))

    distances_1 = np.array(distances_1)
    distances_2 = np.array(distances_2)
    mean_dist_1 = np.mean(distances_1, axis=0)
    mean_dist_2 = np.mean(distances_2, axis=0)

    # Apply translation to the points to bring centroid to origin.
    ones = np.ones((pts1.shape[0], 1), dtype=np.int32)
    pts1 = np.hstack((pts1, ones))
    pts2 = np.hstack((pts2, ones))
    T1 = Construct_Transformation_Matrix(mean_dist_1, -ctd_1[0], -ctd_1[1])
    T2 = Construct_Transformation_Matrix(mean_dist_2, -ctd_2[0], -ctd_2[1])
    pts1_new = (T1 @ pts1.T) # 3 x N
    pts2_new = (T2 @ pts2.T) # 3 x N

    A = np.zeros((pts1_new.shape[1], 9))
    for i in range(pts1_new.shape[1]):
        x1 = np.array([pts1_new[:, i]]).reshape((3, 1))
        x2 = np.array([pts2_new[:, i]]).reshape((3, 1))
        row = x2 @ x1.T
        row = np.reshape(row, (1, 9))
        A[i] = row

    u, sig, vh = np.linalg.svd(A)
    v = vh.T
    F_est = v[:, -1]
    F_est = np.reshape(F_est, (3, 3))
    # Now find SVD for F_est and make it a rank 2 matrix by setting 3rd eigen value to 0 and recalculate F.
    u, sig, vh = np.linalg.svd(F_est)
    sig[2] = 0
    F = u @ np.diag(sig) @ vh
    # Denormalize the matrix F with the translation matrix that we applied for 2 images' points.
    F_mat = T2.T @ F @ T1
    # Since the last coordinate can be voided as a scaling factor, we can normalize the fundamental matrix with it.
    # (Eight point Algorithm)
    F_mat = F_mat * (1 / F_mat[2, 2])
    return F_mat

def FindFundamentalMatrixRansac(pts1, pts2, num_trials = 1000, threshold = 0.01):
    #Input: two lists of corresponding keypoints (numpy arrays of shape (N, 2)).
    #Output: fundamental matrix (numpy array of shape (3, 3)).

    # Set parameters and initialize variables.
    adaptive = False
    pts1 = pts1  # First set of points
    pts2 = pts2  # Second set of points
    size = pts1.shape[0]
    s = 8  # no of point pairs required to compute fundamental matrix
    t = threshold  # Epsilon threshold
    N = num_trials

    e = 0.5  # Assumption Outlier Ratio => no. of outliers / no. of points
    # Adaptively determining value of trials/ iterations - N
    if adaptive:
        p = 0.95  # Required probability of Success
        N = np.log(1-p) / np.log(1 - (pow(1-e, s)))

    F_arr = []  # Array to store Fundamental matrices for each set of sample points.
    inliers_count_arr = []  # Array to store no. of inliers for corresponding Fundamental Matrix.

    # RANSAC Loop
    for i in range(N):
        # Sample 8 points from correspondences pts1 and pts2.
        ## Generate 8 random unique integers between 0 and no. of correspondences for indices.
        indices = sorted(np.random.choice(size, s, replace=False))
        train_pts1 = pts1[indices]
        train_pts2 = pts2[indices]
        ## Compute Fundamental Matrix using the function written above
        F = FindFundamentalMatrix(train_pts1, train_pts2)
        F_arr.append(F)  # Append to list of fundamental matrices

        ## Calculate number of inliers and outliers using Fundamental matrix.
        ### Remember: pts2.T @ F @ pts1 < t ~ 0 # Where t is the threshold.
        test_pts1 = np.delete(pts1, indices, axis=0)
        test_pts2 = np.delete(pts2, indices, axis=0)
        ones = np.ones((test_pts1.shape[0], 1), dtype=np.int32)
        test_pts1 = np.hstack((test_pts1, ones))
        test_pts2 = np.hstack((test_pts2, ones))
        # ones = np.ones((pts1.shape[0], 1), dtype=np.int32)
        # test_pts1 = np.hstack((pts1, ones))
        # test_pts2 = np.hstack((pts2, ones))

        ### Find number of inliers in pts2_c => How many lie inside the threshold t.
        inliers_count = 8

        #### Run a loop against all the entries of pts2 and compare each point with pts2_c and check if it lies in the
        #### desired range of pts2 +- t (threshold).
        for j in range(test_pts1.shape[0]):
            #### loss error
            error = abs(test_pts2[j] @ F @ test_pts1[j])
            # e1 = test_pts2[j] @ F
            #
            # a = e1[0] * test_pts1[j, 0]
            # b = e1[1] * test_pts1[j, 1]
            # c = e1[2] * test_pts1[j, 2]
            #
            # error = np.sqrt(a*a + b*b + c*c)

            if error <= t:
                inliers_count += 1
        # Save the number of inliers in this model based on the loss and threshold above.
        inliers_count_arr.append(inliers_count)

    # Return the best fundamental matrix which fit most number of points as inliers.
    F_best = F_arr[np.argmax(inliers_count_arr)]
    return F_best

if __name__ == '__main__':
    #Set parameters
    data_path = './data'
    use_ransac = True

    #Load images
    image1_path = os.path.join(data_path, 'notredam_1.jpg')
    image2_path = os.path.join(data_path, 'notredam2.jpg')
    image1 = np.array(Image.open(image1_path).convert('L'))
    image2 = np.array(Image.open(image2_path).convert('L'))


    #Find matching keypoints
    pts1, pts2 = find_matching_keypoints(image1, image2)
    #print(pts1[:5])
    #print(pts2[:5])

    #Builtin opencv function for comparison
    F_true = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)[0]

    if use_ransac:
        F = FindFundamentalMatrix(pts1, pts2)
        F_ransac = FindFundamentalMatrixRansac(pts1, pts2, num_trials=1200, threshold=0.001)
        F_ransac_true = cv2.findFundamentalMat(pts1, pts2, method=cv2.FM_RANSAC, ransacReprojThreshold=0.01, confidence=0.5, maxIters=1200)[0]
    else:
        F = FindFundamentalMatrix(pts1, pts2)

    print("CV2 Estimated Fundamental Matrix: \n", F_true)
    print("Manually Estimated Fundamental Matrix: \n", F)
    print("CV2 Estimated Fundamental Matrix with RANSAC: \n", F_ransac_true)
    print("Manually Estimated Fundamental Matrix with RANSAC: \n", F_ransac)

    arr = [F_true, F, F_ransac_true, F_ransac]

    # Loop to draw images
    for f in arr:
        # Find epilines corresponding to points in second image,  and draw the lines on first image
        lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, f)
        lines1 = lines1.reshape(-1, 3)
        img1, img2 = drawlines(image1, image2, lines1, pts1, pts2)
        fig, axis = plt.subplots(2, 2)

        # Find epilines corresponding to points in first image, and draw the lines on second image
        lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, f)
        lines2 = lines2.reshape(-1, 3)
        img3, img4 = drawlines(image2, image1, lines2, pts2, pts1)

        axis[0, 0].imshow(img1)
        axis[0, 0].set_title('Image 1')
        axis[0, 0].axis('off')

        axis[0, 1].imshow(img2)
        axis[0, 1].set_title('Image 2')
        axis[0, 1].axis('off')

        axis[1, 0].imshow(img3)
        axis[1, 0].set_title('Image 3')
        axis[1, 0].axis('off')

        axis[1, 1].imshow(img4)
        axis[1, 1].set_title('Image 4')
        axis[1, 1].axis('off')

        plt.show()





