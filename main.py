from preprocess import PreprocessData


def main():
    dicts = {'01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad', '05': 'angry', '06': 'fearful',
             '07': 'disgust', '08': 'surprised'}

    path_main = '/home/igkinis/projects/datasets/Audio_Speech_Actors_01-24'
    outfolder = '/home/igkinis/projects/datasets/Audio_Speech_Actors_01-24_mels'
    cl_instance = PreprocessData(dicts, path_main, outfolder)

    cl_instance.process_audios()

if __name__ == '__main__':
    main()

