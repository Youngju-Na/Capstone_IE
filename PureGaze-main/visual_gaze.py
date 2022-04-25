import cv2
import numpy as np
import h5py
import os

def pitchyaw_to_vector(pitchyaws):
    r"""Convert given yaw (:math:`\theta`) and pitch (:math:`\phi`) angles to unit gaze vectors.

    Args:
        pitchyaws (:obj:`numpy.array`): yaw and pitch angles :math:`(n\times 2)` in radians.

    Returns:
        :obj:`numpy.array` of shape :math:`(n\times 3)` with 3D vectors per row.
    """
    n = pitchyaws.shape[0]
    sin = np.sin(pitchyaws)
    cos = np.cos(pitchyaws)
    out = np.empty((n, 3))
    out[:, 0] = np.multiply(cos[:, 0], sin[:, 1])
    out[:, 1] = sin[:, 0]
    out[:, 2] = np.multiply(cos[:, 0], cos[:, 1])
    return out

def draw_gaze(image_in, pitchyaw, thickness=2, color=(0, 0, 255)):
    """Draw gaze angle on given image with a given eye positions."""
    image_out = image_in
    (h, w) = image_in.shape[:2]
    length = w / 2.0
    pos = (int(h / 2.0), int(w / 2.0))
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
    dx = -length * np.sin(pitchyaw[1]) * np.cos(pitchyaw[0])
    dy = -length * np.sin(pitchyaw[0])
    cv2.arrowedLine(image_out, tuple(np.round(pos).astype(np.int32)),
                   tuple(np.round([pos[0] + dx, pos[1] + dy]).astype(int)), color,
                   thickness, cv2.LINE_AA, tipLength=0.2)
    return image_out

if __name__ == '__main__':
    sub_id = 0   # subject id number you would like to see
    input_file = './train/subject' + str(sub_id).zfill(4) + '.h5'

    fid = h5py.File(input_file, 'r')
    img_size = 256
    num_data = fid["face_patch"].shape[0]   # get the total number of samples inside the h5 file
    print('num_data: ', num_data)

    img_show = np.zeros((img_size*3, img_size*6, 3), dtype=np.uint8)  # initial a empty image

    cv2.namedWindow("image")
    gaze = []

    num_i = 0
    # while True:
    for num_r in range(0, 3):   # we show them in 3 rows
        for num_c in range(0, 6):   # we show them in 6 columns
            face_patch = fid['face_patch'][num_i, :]  # the face patch
            if 'face_gaze' in fid.keys():
                gaze = fid['face_gaze'][num_i, :]   # the normalized gaze direction with size of 2 dimensions as horizontal and vertical gaze directions.
            frame_index = fid['frame_index'][num_i, 0]  # the frame index
            cam_index = fid['cam_index'][num_i, 0]   # the camera index
            face_mat_norm = fid['face_mat_norm'][num_i, 0]   # the rotation matrix during data normalization
            face_head_pose = fid['face_head_pose'][num_i, 0]  # the normalized head pose with size of 2 dimensions horizontal and vertical head rotations.

            face_patch = cv2.resize(face_patch, (img_size, img_size))
            cv2.imwrite('./image'+str(num_c)+'.jpg', face_patch)
                # if frame_index > 524:  # if the image is captured under low lighting conditions, we do histogram equalization
                #     img_yuv = cv2.cvtColor(face_patch, cv2.COLOR_BGR2YUV)
                #     img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
                #     face_patch = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

                # if 'face_gaze' in fid.keys():
                #     face_patch = draw_gaze(face_patch, gaze)  # draw gaze direction on the face patch image

                # img_show[img_size*num_r:img_size*(num_r+1), img_size*num_c:img_size*(num_c+1)] = face_patch
                # num_i = num_i + 1
                # if num_i >= num_data:
                #     break

        # font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(img_show, 'Please press J to the previous sample, L to the next sample, and ESC to exit', (10, 30),
        #             font, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

        # cv2.imshow('image', img_show)
        # input_key = cv2.waitKey(0)
        # if input_key == 27:  # ESC key to exit
        #     break
        # elif input_key == 106:  # j key to previous
        #     num_i = num_i - 18*2
        #     if num_i < 0:
        #         num_i = 0
        # elif input_key == 108:  # l key to the next
        #     num_i = num_i + 18
        # else:
        #     continue

    # cv2.destroyAllWindows()
    # fid.close()