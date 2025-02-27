import os
import numpy as np
import cv2
import glob
import argparse

h_size = 480
w_size = 640

def extract_frame_for_metric(video_path,output_path):
  cap = cv2.VideoCapture(video_path)
  if os.path.exists(output_path):
    os.rmdir(output_path)
  os.mkdir(output_path)
  count=0
  while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
      cv2.imwrite(output_path+str(count).zfill(5)+'.png',frame)
      count+=1
    else: 
      break
  cap.release()  

# Part of code reference from https://github.com/jinsc37/DIFRINT/blob/master/metrics.py
def metrics(in_src, out_src):
    dic = {
        'M': None,
        'CR_seq': [],
        'DV_seq': [],
        'SS_t': None,
        'SS_r': None,
        'w_crop':[],
        'h_crop':[],
        'distortion': [],
        'count': 0,
        }

    frameList_in = sorted(glob.glob(in_src+'/*.png'))
    frameList = sorted(glob.glob(out_src+'/*.png'))
    all=min(len(frameList_in),len(frameList))
    frameList = frameList[:all]
    frameList_in=frameList_in[:all]

    orb_detector = cv2.ORB_create(5000)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    
    for i in range(all):
        print("Computing cropping and distortion: %d/%d"%(i,all))
        img1 = cv2.imread(frameList_in[i], 0)
        img1 = cv2.resize(img1, (w_size,h_size), interpolation = cv2.INTER_LINEAR)

        img1o = cv2.imread(frameList[i], 0)
        img1o = cv2.resize(img1o, (w_size,h_size), interpolation = cv2.INTER_LINEAR)
        
        try:
            kp1, d1 = orb_detector.detectAndCompute(img1, None) 
            kp2, d2 = orb_detector.detectAndCompute(img1o, None)
            
            matches = matcher.match(d1, d2) 
    
            # Sort matches on the basis of their Hamming distance. 
            matches = sorted(matches, key = lambda x: x.distance)
            
            # Take the top 90 % matches forward. 
            matches = matches[:int(len(matches)*0.9)] 
            no_of_matches = len(matches) 
            
            # Define empty matrices of shape no_of_matches * 2. 
            p1 = np.zeros((no_of_matches, 2)) 
            p2 = np.zeros((no_of_matches, 2)) 
            
            for j in range(len(matches)): 
                p1[j, :] = kp1[matches[j].queryIdx].pt 
                p2[j, :] = kp2[matches[j].trainIdx].pt 
            
            M, _ = cv2.findHomography(p1, p2, cv2.RANSAC)
            
            scaleRecovered = np.sqrt(M[0,1]**2 + M[0,0]**2)

            _, w, _ = np.linalg.svd(M[:2,:2])
            w = np.sort(np.abs(w))[::-1]
            DV = w[1]/w[0]

            dic["CR_seq"].append(1.0/scaleRecovered)
            dic["DV_seq"].append(DV)
        except:
            print("Failed: %d"%(i))
            dic["CR_seq"].append(0)
            dic["DV_seq"].append(0)
        
    P_seq = []

    # params for corner detection 
    feature_params = dict( maxCorners = 5000, 
                        qualityLevel = 0.3, 
                        minDistance = 7, 
                        blockSize = 17 ) 
    
    # Parameters for lucas kanade optical flow 
    lk_params = dict( winSize = (15, 15), 
                    maxLevel = 4, 
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
                                10, 0.03))

    for i in range(all-1):
        print("Computing stability: %d/%d"%(i,all-1))
        try:
            img1 = cv2.imread(frameList[i],0)
            img1 = cv2.resize(img1, (w_size,h_size), interpolation = cv2.INTER_LINEAR)
            img1o = cv2.imread(frameList[i+1],0)
            img1o = cv2.resize(img1o, (w_size,h_size), interpolation = cv2.INTER_LINEAR)
            p0 = cv2.goodFeaturesToTrack(img1, mask = None, **feature_params)
            
            # calculate optical flow 
            p1, st, err = cv2.calcOpticalFlowPyrLK(img1, 
                                                img1o, 
                                                p0, None, 
                                                **lk_params) 
        
            # Select good points 
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            M, mask = cv2.findHomography(good_old, good_new, method=cv2.RANSAC, ransacReprojThreshold=1)
            if M is None:
                new_mat=np.eye(3)
            else:
                new_mat=M
        except:
            print("Failed at frame: %d"%(i))
            new_mat=np.eye(3)
           
        P_seq.append(new_mat)
    
    # Make 1D temporal signals
    P_seq_tx = np.asarray([1])
    P_seq_ty = np.asarray([1])
    P_seq_r = np.asarray([1])

    for Mp in P_seq:
        transRecoveredx = Mp[0, 2]
        transRecoveredy = Mp[1, 2]
		# Based on https://math.stackexchange.com/questions/78137/decomposition-of-a-nonsquare-affine-matrix
        thetaRecovered = np.arctan2(Mp[0, 1], Mp[0, 0]) * 180 / np.pi
        P_seq_tx = np.concatenate((P_seq_tx, [transRecoveredx]), axis=0)
        P_seq_ty = np.concatenate((P_seq_ty, [transRecoveredy]), axis=0)
        P_seq_r = np.concatenate((P_seq_r, [thetaRecovered]), axis=0)

    P_seq_tx = np.delete(P_seq_tx, 0)
    P_seq_ty = np.delete(P_seq_ty, 0)
    P_seq_r = np.delete(P_seq_r, 0)

    # FFT
    fft_tx = np.fft.fft(P_seq_tx)
    fft_ty = np.fft.fft(P_seq_ty)
    fft_r = np.fft.fft(P_seq_r)
    fft_tx = fft_tx[0:int(len(fft_tx)/2)]
    fft_ty = fft_ty[0:int(len(fft_ty)/2)]
    fft_r = fft_r[0:int(len(fft_r)/2)]
    
    fft_tx = abs(fft_tx)**2
    fft_ty = abs(fft_ty)**2
    fft_r = abs(fft_r)**2

    SS_tx = np.sum(fft_tx[:15])/np.sum(fft_tx)
    SS_ty = np.sum(fft_ty[:15])/np.sum(fft_ty)  
    SS_r = np.sum(fft_r[:15])/np.sum(fft_r)

    dic["SS_tx"] = SS_tx
    dic["SS_ty"] = SS_ty
    dic["SS_r"] = SS_r

    return dic


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video stabilization evaluation metrics.")
    parser.add_argument("--input_video", type=str, help="The input unstable video path.")
    parser.add_argument("--stable_video", type=str, help="The stabilized video path.")
    args = parser.parse_args()
    
    extract_frame_for_metric(args.input_video,'./input_images/')
    extract_frame_for_metric(args.stable_video,'./target_images/')
        
    result=metrics(in_src='./input_images/', out_src='./target_images/')
    stability=(result["SS_tx"]+result["SS_ty"]+result["SS_r"])/3
    cropping=np.array(result["CR_seq"])
    distortion=np.absolute(result["DV_seq"])
    #Filter out abnormal values due to bad homography estimation.
    valid=np.where((distortion > 0.3) & (cropping>0) & (cropping<1))
    cropping=cropping[valid]
    distortion = distortion[valid]
    cropping=np.nanmean(cropping)
    distortion=np.nanmin(distortion)
    print("Evaluation Done.")
    print("Stability: %f, Distortion: %f, Cropping: %f"%(stability, distortion, cropping))