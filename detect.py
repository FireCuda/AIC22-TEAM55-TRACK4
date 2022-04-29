import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import sys
import torch
import cv2
import torch.backends.cudnn as cudnn
from numpy import random
from predict_deit import inference
import ast
from utils.google_utils import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer)
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
sys.path.append(os.path.join(os.getcwd(),'yolov5'))
print(sys.path)
from models.models import *
from utils.datasets import *
from utils.general import *
import pandas as pd
def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)

def detect(save_img=False):
    df =  pd.DataFrame(columns=['video_id','class_id', 'bbox', 'frame'])

    out, source, weights, view_img, save_txt, imgsz, cfg, names = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.cfg, opt.names
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')
    # Initialize
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = Darknet(cfg, imgsz).cuda()
    model.load_state_dict(torch.load(weights[0], map_location=device)['model'])
    #model = attempt_load(weights, map_location=device)  # load FP32 model
    #imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz, auto_size=64)
  

    # Get names and colors
    names = load_classes(names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    detect_tray = 1
    path_tmp = ''
    count_frame_detect_tray = 0
    for path, img, im0s, vid_cap in dataset:
        # print('-------',path)
        if path_tmp!= path:
            count_frame_detect_tray = 0
        count_frame_detect_tray += 1 
        if detect_tray == 1 and count_frame_detect_tray<2:
            path_tmp = path
            vidcap = cv2.VideoCapture(path)
            frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = int(vidcap.get(cv2.CAP_PROP_FPS))
            seconds = int(frames / fps)
            success,image = vidcap.read()
            count = 0
            fps_to_second_vid = seconds/frames
            while success:
                print('1')
                cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file      
                print('Read a new frame: ', success)
                count += 1
                if count == 1:
                    break
            model_tray = torch.hub.load('./yolov5/', 'custom', path='tray_detect_mosaicless_2303.pt', source='local',force_reload=True)
            # print(model_tray)
            results = model_tray('frame0.jpg')
            xmin_tray, ymin_tray, xmax_tray, ymax_tray = int(results.pandas().xyxy[0]['xmin'].values), int(results.pandas().xyxy[0]['ymin'].values),int(results.pandas().xyxy[0]['xmax'].values), int(results.pandas().xyxy[0]['ymax'].values)
            print(xmin_tray,ymin_tray,xmax_tray,ymax_tray)
            with open('bounding_box_tray.txt','a') as f:
                f.write(str(path.split('/')[-1].split('.')[0])+'\t'+str([xmin_tray,ymin_tray,xmax_tray,ymax_tray])+'\t'+str(fps_to_second_vid)+'\n')
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s
            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy()
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    bbox=torch.tensor(xyxy).view(1, 4).tolist()[0]
                    print(bbox)
                    croped = imc[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
                    croped = cv2.cvtColor(croped, cv2.COLOR_BGR2RGB)
                    score, label_cls = inference(Image.fromarray(croped))
                    print(label_cls)
                    if (label_cls!='None'):
                        df = df.append({
                            'video_id':Path(p).stem,
                            'frame':dataset.frame,
                            'bbox':[int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                            'class_id':label_cls,
                            'score_cls' : score
                        },ignore_index=True)
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label_cls, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)
        df.to_csv('./output_yolo/result.csv')

    if save_txt or save_img:
        print('Results saved to %s' % Path(out))
        if platform == 'darwin' and not opt.update:  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    if os.path.exists('bounding_box_tray.txt'):
        os.remove('bounding_box_tray.txt')
    if os.path.exists('track4.txt'):
        os.remove('track4.txt')
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolor_p6.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='TestA/', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=1280, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--cfg', type=str, default='cfg/yolor_p6_custom.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/custom.names', help='*.cfg path')
    opt = parser.parse_args()
    # print(opt.source)
    # vidcap = cv2.VideoCapture(video_path)
    # count = 0
    # while success:
    #     print('1')
    #     cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file      
    #     success,image = vidcap.read()
    #     print('Read a new frame: ', success)
    #     count += 1
    #     if count == 1:
    #         break
    # model_tray = torch.hub.load('.', 'custom', path='tray_detect_mosaicless_2303.pt', source='local') 
    # # print(video_path[0])
    # results = model_tray('frame0.jpg')
    # # print(results.pandas().xyxy[0])
    # xmin_tray, ymin_tray, xmax_tray, ymax_tray = int(results.pandas().xyxy[0]['xmin'].values), int(results.pandas().xyxy[0]['ymin'].values),int(results.pandas().xyxy[0]['xmax'].values), int(results.pandas().xyxy[0]['ymax'].values)

    
    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['']:
                detect()
                strip_optimizer(opt.weights)
        else:
            # print(opt.source)
            detect()
    with open('bounding_box_tray.txt','r') as f:
        bbox_ls = f.readlines()
    video_info = []
    bbox_info = []
    fps_to_second_vid = []
    for i in bbox_ls:
        i = i.split('\t')
        video_info.append(i[0])
        bbox_info.append(ast.literal_eval(i[1]))
        fps_to_second_vid.append(i[-1])
    with open('labels.txt','r',encoding='utf-8') as f:
        label = f.readlines()
    for i in range(len(label)):
        label[i] = label[i].replace('\n','')
        label[i] = label[i].split(',')
    dictionary = {}
    for i in range(len(label)):
        dictionary.update({'{}'.format(label[i][0]):'{}'.format(label[i][1])})
    result = pd.read_csv('./output_yolo/result.csv')
    cnt = 0
    res = []
    video_id = list(result.video_id.unique())
    for i in range(len(video_id)):
        cnt+=1
        res_vid = result[result['video_id']==video_id[i]]
        fps_to_second = float(fps_to_second_vid[i])
        bbox = bbox_info[i]
        for m in range(len(res_vid['video_id'])):
            obj_bbox = ast.literal_eval(str(res_vid['bbox'][m]))
            if (obj_bbox[0] < bbox[0] and obj_bbox[1] < bbox[1] or obj_bbox[2] < bbox[2] and obj_bbox[3] < bbox[3]):
                if res_vid["score_cls"][m]>8.3:
                # print(1)
                    class_obj = res_vid["class_id"].values[m]
                    class_id = dictionary.get('{}'.format(class_obj))
                    if class_id in res:
                        continue
                    res.append(class_id)
                    with open('track4.txt','a',encoding='utf-8') as f:
                        f.writelines(str(cnt)+ " " + class_id + " "+ str(int(res_vid['frame'][m]*fps_to_second))+'\n')
        res = []
    # for m in range(len(result['video_id'])):
    #     if str(result['video_id'][m]) == "testA_1":
    #         fps_to_second = 30/1799
    #         bbox = [484, 250, 1250, 847]
    #         cnt=1
    #         obj_bbox = ast.literal_eval(str(result['bbox'][m]))
    #         if (obj_bbox[0] < bbox[0] and obj_bbox[1] < bbox[1] or obj_bbox[2] < bbox[2] and obj_bbox[3] < bbox[3]):
    #             if result["score_cls"][m]>6.3:
    #             # print(1)
    #                 class_obj = result["class_id"].values[m]
    #                 class_id = dictionary.get('{}'.format(class_obj))
    #                 if class_id in res1:
    #                     continue
    #                 res1.append(class_id)
    #                 with open('track4.txt','a',encoding='utf-8') as f:
    #                     f.writelines(str(cnt)+ " " + class_id + " "+ str(int(result['frame'][m]*fps_to_second))+'\n')
    #     elif str(result['video_id'][m]) == "testA_2":
    #         fps_to_second = 25/1499
    #         bbox = [589, 298, 1359, 904]
    #         cnt=2
    #         obj_bbox = ast.literal_eval(str(result['bbox'][m]))
    #         if (obj_bbox[0] < bbox[0] and obj_bbox[1] < bbox[1] or obj_bbox[2] < bbox[2] and obj_bbox[3] < bbox[3]):
    #             if result["score_cls"][m]>9:
    #             # print(1)
    #                 class_obj = result["class_id"].values[m]
    #                 class_id = dictionary.get('{}'.format(class_obj))
    #                 if class_id in res2:
    #                     continue
    #                 res2.append(class_id)
    #                 with open('track4.txt','a',encoding='utf-8') as f:
    #                     f.writelines(str(cnt)+ " " + class_id + " "+ str(int(result['frame'][m]*fps_to_second))+'\n')
    #     elif str(result['video_id'][m]) == "testA_3":
    #         fps_to_second = 60/2642
    #         bbox =[586, 302, 1354, 905]
    #         cnt=3
    #         obj_bbox = ast.literal_eval(str(result['bbox'][m]))
    #         if (obj_bbox[0] < bbox[0] and obj_bbox[1] < bbox[1] or obj_bbox[2] < bbox[2] and obj_bbox[3] < bbox[3]):
    #             if result["score_cls"][m]>9.7:
    #             # print(1)
    #                 class_obj = result["class_id"].values[m]
    #                 class_id = dictionary.get('{}'.format(class_obj))
    #                 if class_id in res3:
    #                     continue
    #                 res3.append(class_id)
    #                 with open('track4.txt','a',encoding='utf-8') as f:
    #                     f.writelines(str(cnt)+ " " + class_id + " "+ str(int(result['frame'][m]*fps_to_second))+'\n')
    #     elif str(result['video_id'][m]) == "testA_4":
    #         fps_to_second = 35/2123
    #         bbox = [582, 303, 1354, 910]
    #         cnt=4
    #         obj_bbox = ast.literal_eval(str(result['bbox'][m]))
    #         if (obj_bbox[0] < bbox[0] and obj_bbox[1] < bbox[1] or obj_bbox[2] < bbox[2] and obj_bbox[3] < bbox[3]):
    #             if result["score_cls"][m]>8.3:
    #             # print(1)
    #                 class_obj = result["class_id"].values[m]
    #                 class_id = dictionary.get('{}'.format(class_obj))
    #                 if class_id in res4:
    #                     continue
    #                 res4.append(class_id)
    #                 with open('track4.txt','a',encoding='utf-8') as f:
    #                     f.writelines(str(cnt)+ " " + class_id + " "+ str(int(result['frame'][m]*fps_to_second))+'\n')
    #     elif str(result['video_id'][m]) == "testA_5":
    #         fps_to_second = 25/1547
    #         bbox = [592, 304, 1363, 913]
    #         cnt=5
    #         obj_bbox = ast.literal_eval(str(result['bbox'][m]))
    #         if (obj_bbox[0] < bbox[0] and obj_bbox[1] < bbox[1] or obj_bbox[2] < bbox[2] and obj_bbox[3] < bbox[3]):
    #             if result["score_cls"][m]>9:
    #             # print(1)
    #                 class_obj = result["class_id"].values[m]
    #                 class_id = dictionary.get('{}'.format(class_obj))
    #                 if class_id in res5:
    #                     continue
    #                 res5.append(class_id)
    #                 with open('track4.txt','a',encoding='utf-8') as f:
    #                     f.writelines(str(cnt)+ " " + class_id + " "+ str(int(result['frame'][m]*fps_to_second))+'\n')
    #     else:
    #         fps_to_second = 25/1547
    #         bbox = [592, 304, 1363, 913]
        # print(video)
       
        
        # print(calculate_iou(bbox,obj_bbox))
        
          
