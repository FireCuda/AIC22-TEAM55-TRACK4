import argparse
import os.path
import time
import sys 
from transformers import DeiTFeatureExtractor, DeiTForImageClassification

from hugsvision.inference.VisionClassifierInference import VisionClassifierInference

# parser = argparse.ArgumentParser(description='Image classifier')
# parser.add_argument('--path', type=str, default="./out/KVASIR_V2_MODEL_DEIT/20_2021-08-20-01-46-44/model/", help='The model path')
# parser.add_argument('--img', type=str, default="../../../samples/kvasir_v2/dyed-lifted-polyps.jpg", help='The input image')
# args = parser.parse_args() 

# print("Process the image: " + args.img)
classifier = VisionClassifierInference(
            feature_extractor = DeiTFeatureExtractor.from_pretrained('./out/KVASIR_V2/5_2022-03-24-00-14-23/model'),
            model = DeiTForImageClassification.from_pretrained('./out/KVASIR_V2/5_2022-03-24-00-14-23/model'),
        )
def inference(img):
    # try:

    
        # start = time.time()
        score, label = classifier.predict(img)
        # print("Predicted class:", label)
        # print('total time: ',time.time()-start)
        # probas = classifier.predict(img_path=args.img, return_str=False)
        # print("Probabilities:", probas)
        return score, label
    # except Exception as e:
    #     if "containing a preprocessor_config.json file" in str(e) and os.path.isfile(args.path + "config.json") == True:
    #         print("\033[91m\033[4mError:\033[0m")
    #         print("\033[91mRename the config.json file into \033[4mpreprocessor_config.json\033[0m")
    #     else:
    #         print(str(e))
