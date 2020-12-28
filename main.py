from Dataloader.Dataloader import Dataloader


def main():
    dicts = {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fearful',
             7: 'disgust', 8: 'surprised'}

    path_main = '/home/igkinis/projects/datasets/subset_RAVDESS'
    outfolder = '/home/igkinis/projects/datasets/subset_RAVDESS_mels'
    cl_instance = Dataloader(dicts, path_main, outfolder)
    cl_instance.load_data(save2disk=True)
    cl_instance.feature_extraction()
    cl_instance.preprocess_data()
    print("here")


if __name__ == '__main__':
    main()
