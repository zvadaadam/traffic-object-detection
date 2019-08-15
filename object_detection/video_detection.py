import cv2
import tensorflow as tf
import datetime
from object_detection.detector import Detector
from object_detection.config.config_reader import ConfigReader
from object_detection.utils import image_utils

class VideoDetection(object):

    def __init__(self):
        pass

    def detect(self, video_path, detection_config_path ):

        detection_config = ConfigReader(detection_config_path)

        video_cap = cv2.VideoCapture(video_path)

        if (video_cap.isOpened() == False):
            raise Exception("Error opening video stream or file")

        #frame_width, frame_height = int(video_cap[3]), int(video_cap[4])
        video_out = cv2.VideoWriter('dji_detection.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 10,
                                    (detection_config.image_width(), detection_config.image_height()))

        with tf.Session() as session:
            # dataset = UdacityObjectDataset(config)
            # test_df = dataset.test_dataset()
            # images = np.array(test_df['image'].values.tolist()[0:90], dtype=np.float32)
            # label = np.array(test_df['label'].values.tolist()[0:1], dtype=np.float32)[0]

            # init YOLO detector
            detector = Detector(session, config=detection_config)
            detector.init_prediction()

            video_out = self.apply_detection_on_video(detector, video_cap, video_out, detection_config)

        # When everything done, release the video capture object
        video_cap.release()
        video_out.release()

        # Closes all the frames
        cv2.destroyAllWindows()

    def apply_detection_on_video(self, detector, video_capture, video_writer, detection_config: ConfigReader):

        while video_capture.isOpened():

            # Capture frame-by-frame
            success_flag, frame = video_capture.read()

            if success_flag:
                resized_frame = image_utils.letterbox_image_2(frame, (detection_config.image_width(), detection_config.image_height()))
                # resized_frame = cv2.resize(frame, (detection_config.image_width(), detection_config.image_height()),
                #                            interpolation=cv2.INTER_LINEAR)
#                resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)  # changed colors
                #resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2BGR)

                now = datetime.datetime.now()

                # apply YOLO prediction
                cv2.imshow('test', resized_frame)

                output = detector.predict(resized_frame)
                # draw bounding boxes on image
                detected_img = image_utils.draw_boxes_PIL(resized_frame, output[0], output[1], output[2])
                # Display the resulting frame
                cv2.imshow('detection_frame', detected_img)

                after = datetime.datetime.now()
                print(f'Inference Duration: {after - now}')

                # Write the frame into the file 'output.avi'
                video_writer.write(detected_img)

                # Press Q on keyboard to  exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Break the loop
            else:
                print('Not Success Flag!')
                break


if __name__ == '__main__':
    video_path = '/Users/adam.zvada/Documents/Dev/object-detection/videos/dji.MP4'
    #video_path = '/Users/adam.zvada/Documents/Dev/object-detection/videos/bali.MP4'
    #video_path = '/Users/adam.zvada/Documents/Dev/object-detection/videos/bcn.MP4'

    #detection_config_path = '/Users/adam.zvada/Documents/Dev/object-detection/config/test.yml'
    detection_config_path = '/Users/adam.zvada/Documents/Dev/object-detection/config/yolo.yml'

    print('RUNNING!')

    detection = VideoDetection()

    detection.detect(video_path, detection_config_path )
