import sys, time, os, tqdm, torch, argparse, glob, subprocess, warnings, cv2, pickle, numpy, pdb, math, python_speech_features
import numpy as np
from array import array


sys.path.append("~/youngju/Capstone_IE/TalkNet-ASD")

from retinaface_tf2.modules.models import RetinaFaceModel


from scipy import signal
from shutil import rmtree
from scipy.io import wavfile
from scipy.interpolate import interp1d
from sklearn.metrics import accuracy_score, f1_score

from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector

from model.faceDetector.s3fd import S3FD
from talkNet import talkNet
import pyaudio
import wave

# const values for mic streaming
CHUNK = 1024
BUFF = CHUNK * 10
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000


parser = argparse.ArgumentParser(description = "TalkNet Demo or Columnbia ASD Evaluation")

parser.add_argument('--videoName',             type=str, default="001",   help='Demo video name')
parser.add_argument('--videoFolder',           type=str, default="demo",  help='Path for inputs, tmps and outputs')
parser.add_argument('--pretrainModel',         type=str, default="pretrain_TalkSet.model",   help='Path for the pretrained TalkNet model')

parser.add_argument('--nDataLoaderThread',     type=int,   default=10,   help='Number of workers')
parser.add_argument('--facedetScale',          type=float, default=0.25, help='Scale factor for face detection, the frames will be scale to 0.25 orig')
parser.add_argument('--minTrack',              type=int,   default=10,   help='Number of min frames for each shot')
parser.add_argument('--numFailedDet',          type=int,   default=10,   help='Number of missed detections allowed before tracking is stopped')
parser.add_argument('--minFaceSize',           type=int,   default=1,    help='Minimum face size in pixels')
parser.add_argument('--cropScale',             type=float, default=0.40, help='Scale bounding box')

parser.add_argument('--start',                 type=int, default=0,   help='The start time of the video')
parser.add_argument('--duration',              type=int, default=0,  help='The duration of the video, when set as 0, will extract the whole video')

parser.add_argument('--evalCol',               dest='evalCol', action='store_true', help='Evaluate on Columnbia dataset')
parser.add_argument('--colSavePath',           type=str, default="/data08/col",  help='Path for inputs+, tmps and outputs')

args = parser.parse_args()


class RealTimeDetection:
    """
    real time inference for Talk Detection
    """
    def __init__(self):
        # 모델 호출
        # GPU: active speaker detection by pretrained TalkNet
        s = talkNet() # TalkNet 모델 호출
        s.loadParameters(args.pretrainModel) # 사전학습된 모델 불러오고 argument 호출
        sys.stderr.write("Model %s loaded from previous state! \r\n" % args.pretrainModel)
        s.eval()

        # audio setting
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        self.video = cv2.VideoCapture(0)
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        # print("fps: ", self.fps)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.DET = S3FD(device='cuda')

        # 저장 장소 세팅
        # 참고할 프레임 구간 설정 (e.g., ~3초)

    def __call__(self):
        # 비디오 인풋으로 받기
        # 오디오 인풋으로 받기

        # 두 인풋의 싱크 맞추기
        # 두 인풋을 모델에 넣기

        while True:
            start_time = time.time()
            ret, frame = self.video.read()

            frame_height, frame_width, _ = frame.shape
            img = np.float32(frame.copy())

            if ret == True:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224))
                bboxes = self.DET.detect_faces(img, conf_th=0.9, scales=[args.facedetScale])

                # print(bboxes)
                # self.visualize(bboxes)
                # audio
                data = self.stream.read(CHUNK, exception_on_overflow=False)
                # print("Listening")
                vol = max(array('h', data))


                audioFeature = python_speech_features.mfcc(audio, 16000, numcep=13, winlen=0.025, winstep=0.010)
                videoFeature = numpy.array(videoFeature)
                length = min((audioFeature.shape[0] - audioFeature.shape[0] % 4) / 100, videoFeature.shape[0])
                batchSize = int(math.ceil(length / duration))
                scores = []
                with torch.no_grad():
                    for i in range(batchSize):
                        inputA = torch.FloatTensor(
                            audioFeature[i * duration * 100:(i + 1) * duration * 100, :]).unsqueeze(0).cuda()
                        inputV = torch.FloatTensor(
                            videoFeature[i * duration * 25: (i + 1) * duration * 25, :, :]).unsqueeze(0).cuda()



            else:
                break

            # show frame
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) == ord('q'):
                exit()

        # self.evalulate_network(video, audio)
        # # 모델이 해당 프레임에 대해 추론한 결과를 화면에 띄우기
        #
        # # 프레임 저장소 업데이트하기
        # pass

    def visualize(bboxes):
        pass

    def evaluate_network(self, video, audio, files, args):
        # GPU: active speaker detection by pretrained TalkNet

        allScores = []
        # durationSet = {1,2,4,6} # To make the result more reliable
        durationSet = {1,1,1,2,2,2,3,3,4,5,6} # Use this line can get more reliable result
        fileName = os.path.splitext(file.split('/')[-1])[0]  # Load audio and video
        _, audio = wavfile.read(os.path.join(args.pycropPath, fileName + '.wav'))
        audioFeature = python_speech_features.mfcc(audio, 16000, numcep = 13, winlen = 0.025, winstep = 0.010)

        videoFeature = []

        assert video.isOpened()

        while(True):

            ret, frames = video.read()
            if ret == True:
                face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
                face = cv2.resize(face, (224,224))

                face = face[int(112-(112/2)):int(112+(112/2)), int(112-(112/2)):int(112+(112/2))]
                videoFeature.append(face)
            else:
                break

            video.release()
            videoFeature = numpy.array(videoFeature)
            length = min((audioFeature.shape[0] - audioFeature.shape[0] % 4) / 100, videoFeature.shape[0])
            audioFeature = audioFeature[:int(round(length * 100)),:]
            videoFeature = videoFeature[:int(round(length * 25)),:,:]
            allScore = [] # Evaluation use TalkNet
            for duration in durationSet:
                batchSize = int(math.ceil(length / duration))
                scores = []
                with torch.no_grad():
                    for i in range(batchSize):
                        inputA = torch.FloatTensor(audioFeature[i * duration * 100:(i+1) * duration * 100,:]).unsqueeze(0).cuda()
                        inputV = torch.FloatTensor(videoFeature[i * duration * 25: (i+1) * duration * 25,:,:]).unsqueeze(0).cuda()
                        embedA = s.model.forward_audio_frontend(inputA)
                        embedV = s.model.forward_visual_frontend(inputV)
                        embedA, embedV = s.model.forward_cross_attention(embedA, embedV)
                        out = s.model.forward_audio_visual_backend(embedA, embedV)
                        score = s.lossAV.forward(out, labels = None)
                        scores.extend(score)
                allScore.append(scores)
            allScore = numpy.round((numpy.mean(numpy.array(allScore), axis = 0)), 1).astype(float)
            allScores.append(allScore)

        return allScores

    def inference_video(self, args):
        # GPU: Face detection, output is the list contains the face location and score in this frame
        DET = S3FD(device='cuda')
        flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg'))
        flist.sort()
        dets = []
        for fidx, fname in enumerate(flist):
            image = cv2.imread(fname)
            imageNumpy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            bboxes = DET.detect_faces(imageNumpy, conf_th=0.9, scales=[args.facedetScale])
            dets.append([])
            for bbox in bboxes:
                dets[-1].append({'frame': fidx, 'bbox': (bbox[:-1]).tolist(),
                                 'conf': bbox[-1]})  # dets has the frames info, bbox info, conf info
            sys.stderr.write('%s-%05d; %d dets\r' % (args.videoFilePath, fidx, len(dets[-1])))
        savePath = os.path.join(args.pyworkPath, 'faces.pckl')
        with open(savePath, 'wb') as fil:
            pickle.dump(dets, fil)
        return dets



if __name__ == "__main__":
    realtime_detector = RealTimeDetection()
    realtime_detector()

