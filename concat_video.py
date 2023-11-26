import cv2
import os 
import numpy as np

def merge_vid(vidname1, vidname2, outputname):
    print(f"Vertically stack {vidname1} and {vidname2}, save as {outputname}")
    #os.makedirs(os.path.dirname(outputname), exist_ok=True)

    # Get input video capture
    cap1 = cv2.VideoCapture(vidname1)
    cap2 = cv2.VideoCapture(vidname2)

    # Default resolutions of the frame are obtained.The default resolutions
    # are system dependent.
    # We convert the resolutions from float to integer.
    # https://docs.opencv.org/2.4/modules/highgui/doc
    # /reading_and_writing_images_and_video.html#videocapture-get
    frame_width = int(cap1.get(3))
    frame_height = int(cap1.get(4))
    fps1 = cap1.get(10)
    FOURCC = int(cap1.get(6))
    num_frames = int(cap1.get(7))

    frame_width2 = int(cap2.get(3))
    frame_height2 = int(cap2.get(4))
    fps2 = cap2.get(10)
    num_frames2 = int(cap2.get(7))

    print(fps1, fps2)
    if fps1 > fps2:
        fps = fps1
    else:
        fps = fps2

    # assert frame_height == frame_height2, \
    #     f"Height of frames are not equal. {frame_height} vs. {frame_height2}"
    assert num_frames == num_frames2, \
        f"Number of frames are not equal. {num_frames} vs. {num_frames2}"
    assert frame_width == frame_width2 and frame_height == frame_height2 
    
    #size = (shape[1], shape[0])

    #print(f"video size: {size}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    #output_path = 'output.mp4'
    fps = 10

    #videoWriter = cv2.VideoWriter(output_path, fourcc, fps, size, True)

    # Set output videowriter
    vidsize = (frame_width, frame_height * 2)
    out = cv2.VideoWriter(outputname, fourcc, fps, vidsize, True)
    print(outputname, FOURCC, fps, vidsize)
    # Loop over and save
    print(f"Total {num_frames} frames. Now saving...")
    idx = 0
    while (cap1.isOpened() and cap2.isOpened() and idx < num_frames):
        ret1 = None
        ret2 = None 

        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not (ret1 and ret2):
            break

        #out_frame = np.hstack([frame1, frame2])
        out_frame = np.zeros((frame1.shape[0]*2, frame1.shape[1], 3), dtype=frame1.dtype)
        out_frame[:frame1.shape[0], :, :] = frame1
        out_frame[frame1.shape[0]:, :, :] = frame2
        out.write(out_frame)
        
        #print(f"idx={idx}, frame1.shape={frame1.shape}, frame2.shape={frame2.shape}, out_frame.shape={out_frame.shape}")
        idx += 1

        '''
        ret1 = ret2 = False
        frame1 = frame2 = None
        if idx % (fps / fps1) == 0.0:
            # print(idx, fps/fps2, "1")
            ret1, frame1 = cap1.read()
        if idx % (fps / fps2) == 0.0:
            # print(idx, fps/fps1, "2")
            ret2, frame2 = cap2.read()
            if frame_height != frame_height2:
                frame2 = cv2.resize(frame2, (frame_height, frame_height))
        # print(ret1, ret2)
        if ret1 and ret2:
            out_frame = np.hstack([frame1, frame2])
            out.write(out_frame)
        idx += 1
        '''
    print(f"num_frames={num_frames}, idx={idx}")
    out.release()
    cap1.release()
    cap2.release()
    print(f'{outputname} Done!')

if __name__ == '__main__':
    video1 = 'output_2.mp4'
    video2 = 'output_3.mp4'
    output = 'final.mp4'
    
    merge_vid(video1, video2, output)
