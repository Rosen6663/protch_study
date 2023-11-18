import wave
def pcm2wav(pcm_file, wav_file, channels=1, bits=16, sample_rate=80000):
    pcmf = open(pcm_file, 'rb')
    pcmdata = pcmf.read()
    pcmf.close()

    if bits % 8 != 0:
        raise ValueError("bits % 8 must == 0. now bits:" + str(bits))

    wavfile = wave.open(wav_file, 'wb')
    wavfile.setnchannels(channels)
    wavfile.setsampwidth(bits // 8)
    wavfile.setframerate(sample_rate)
    wavfile.writeframes(pcmdata)
    wavfile.close()
pcm2wav('test.pcm','succeed.wav')

