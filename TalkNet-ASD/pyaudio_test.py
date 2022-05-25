import pyaudio
import wave



p = pyaudio.PyAudio()


CHUNK = 1024 # 1초에 몇개의 청크를 담을 것인가?
FORMAT = pyaudio.paInt16 # 소리 입력 format
CHANNELS = 1 # 마이크 수?
RATE = 44100 # 모르겠음
RECORD_SECONDS = 5  # 녹화할 시간
WAVE_OUTPUT_FILENAME = "output.wav" # 저장할 파일 이름

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("Start to record the audio.")

frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("Recording is finished.")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()