from Dataloader.Dataloader import Dataloader


def main():
    dicts = {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fearful',
             7: 'disgust', 8: 'surprised'}

    path_main = '/home/igkinis/projects/datasets/subset_RAVDESS'
    outfolder = '/home/igkinis/projects/datasets/subset_RAVDESS_mels'
    cl_instance = Dataloader(dicts, path_main, outfolder)
    cl_instance.load_data(save2disk=True)
    cl_instance.feature_extraction(feature_type="mfcc", pooling=True)
    x_train, x_test, y_train, y_test = cl_instance.preprocess_data()
    print(cl_instance.train.shape)


if __name__ == '__main__':
    main()
