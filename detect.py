# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse  # python的命令行解析的标准模块  可以让我们直接在命令行中就可以向程序中传入参数并让程序运行
import csv
import os
import platform
import sys  # sys系统模块 包含了与Python解释器和它的环境有关的函数
from pathlib import Path  # Path将str转换为Path对象 使字符串路径易于操作的模块

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # object置信度阈值 默认0.25  用在nms中
        iou_thres=0.45,  # 做nms的iou阈值 默认0.45   用在nms中
        max_det=1000,  # 每张图片最多的目标数量  用在nms中
        device='',  # 设置代码执行的设备 cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # 是否展示预测之后的图片或视频 默认False
        save_txt=False,  # 是否将预测的框坐标以txt文件格式保存 默认True 会在runs/detect/expn/labels下生成每张图片预测的txt文件
        save_csv=False,  # save results in CSV format
        save_conf=False,  # 是否保存预测每个目标的置信度到预测tx文件中 默认True
        save_crop=False,  # 是否需要将预测到的目标从原图中扣出来 剪切好 并保存 会在runs/detect/expn下生成crops文件，将剪切的图片保存在里面  默认False
        nosave=False,  # 是否不要保存预测后的图片  默认False 就是默认要保存预测后的图片
        classes=None,  # 在nms中是否是只保留某些特定的类 默认是None 就是所有类只要满足条件都可以保留
        agnostic_nms=False,  # 进行nms是否也除去不同类别之间的框 默认False
        augment=False,  # 预测是否也要采用数据增强 TTA 默认False
        visualize=False,  # visualize features
        update=False,  # 是否将optimizer从ckpt中删除  更新模型  默认False
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # 当前测试结果放在run/detect下的文件名  默认是exp  =>  run/detect/exp
        exist_ok=False,  # 是否存在当前文件 默认False 一般是 no exist-ok 连用  所以一般都要重新创建文件夹
        line_thickness=3,  # bounding box thickness (pixels)   画框的框框的线宽  默认是 3
        hide_labels=False,  # 画出的框框是否需要隐藏label信息 默认False
        hide_conf=False,  # 画出的框框是否需要隐藏conf信息 默认False
        half=False,  # 是否使用半精度 Float16 推理 可以缩短推理时间 但是默认是False
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    # 是否保存预测后的图片 默认nosave=False 所以只要传入的文件地址不是以.txt结尾 就都是要保存预测后的图片的
    save_img = not nosave and not source.endswith('.txt')  # save inference images   True
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # 检查当前Path(project) / name是否存在 如果存在就新建新的save_dir 默认exist_ok=False 需要重建
    # 将原先传入的名字扩展成新的save_dir 如runs/detect/exp存在 就扩展成 runs/detect/exp1
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    # 如果需要save txt就新建save_dir / 'labels' 否则就新建save_dir
    # 默认save_txt=False 所以这里一般都是新建一个 save_dir(runs/detect/expn)
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # 获取当前主机可用的设备
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)  # batch_size
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # 5.3、nms除去多余的框
        # Apply NMS  进行NMS
        # conf_thres: 置信度阈值
        # iou_thres: iou阈值
        # classes: 是否只保留特定的类别 默认为None
        # agnostic_nms: 进行nms是否也去除不同类别之间的框 默认False
        # max_det: 每张图片的最大目标个数 默认1000
        # pred: [num_obj, 6] = [5, 6]   这里的预测信息pred还是相对于 img_size(640) 的
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Define the path for the CSV file
        csv_path = save_dir / 'predictions.csv'

        # Create or append to the CSV file
        def write_to_csv(image_name, prediction, confidence):
            data = {'Image Name': image_name, 'Prediction': prediction, 'Confidence': confidence}
            with open(csv_path, mode='a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not csv_path.is_file():
                    writer.writeheader()
                writer.writerow(data)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            # 当前图片路径 如 F:\yolo_v5\yolov5-U\data\images\bus.jpg
            p = Path(p)  # to Path
            # 图片/视频的保存路径save_path 如 runs\\detect\\exp8\\bus.jpg
            save_path = str(save_dir / p.name)  # im.jpg
            # txt文件(保存预测框坐标)保存路径 如 runs\\detect\\exp8\\labels\\bus
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            # print string  输出信息  图片shape (w, h)
            s += '%gx%g ' % im.shape[2:]  # print string
            #  normalization gain gn = [w, h, w, h]  用于后面的归一化
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            # imc: for save_crop 在save_crop中使用
            imc = im0.copy() if save_crop else im0  # for save_crop
            # 这是一个与画检测框相关的类
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                # 将预测信息（相对img_size 640）映射回原图
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                # 输出信息s + 检测到的各个类别的目标个数
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                # 保存预测信息: txt、img0上画框、crop_img
                for *xyxy, conf, cls in reversed(det):
                    # 将每个图片的预测信息分别存入save_dir/labels下的xxx.txt中 每行: class_id+score+xywh
                    c = int(cls)  # integer class
                    label = names[c] if hide_conf else f'{names[c]}'
                    confidence = float(conf)
                    confidence_str = f'{confidence:.2f}'

                    if save_csv:
                        write_to_csv(p.name, label, confidence_str)
                    if save_txt:  # Write to file
                        # 将xyxy(左上角 + 右下角)格式转换为xywh(中心的 + 宽高)格式 并除以gn(whwh)做归一化 转为list再保存
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    # 在原图上画框 + 将预测到的目标剪切出来 保存成图片 保存在save_dir/crops下
                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        # 如果需要就将预测到的目标剪切出来 保存成图片 保存在save_dir/crops下
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            # 是否需要保存图片或视频（检测后的图片/视频 里面已经被我们画好了框的） img0
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    # 权重路径
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    # 测试数据文件(图片或视频)的保存路径
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    # 在检测模块中没用到
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    # 网络输入图片的大小
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    # 目标检测置信度阈值,值越大,可信度更高,同时获得检测目标/检测框越少
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    # 做nms的iou阈值,若两个检测框iou大于该值,其中conf-thres小的框被舍弃. 即该值越大,获得检测框越多
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    # 每张图片最大的目标个数
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    # 设置代码执行的设备,gpu填数字,无gpu可以填cpu, 若什么都不写, 程序会gpu->cpu,依次检测可用设备进行推理
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # 是否展示预测之后的图片或视频 默认False
    parser.add_argument('--view-img', action='store_true', help='show results')
    # 是否将预测的框坐标以txt文件格式保存 默认False
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    # 是否将预测的框坐标以csv文件格式保存 默认False
    parser.add_argument('--save-csv', action='store_true', help='save results in CSV format')
    # 是否保存预测每个目标的置信度到预测tx文件中 默认False
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    # 是否需要将预测到的目标从原图中扣出来 剪切好, 默认False
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    # 是否不要保存预测后的图片  默认False 就是默认要保存预测后的图片
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # 在nms中是否是只保留某些特定的类 默认是None 就是所有类只要满足条件都可以保留
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    # 进行nms是否也除去不同类别之间的框 默认False
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    # 预测是否也要采用数据增强 TTA
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    # 可视化特征?
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    # 是否将optimizer从ckpt中删除  更新模型  默认False
    parser.add_argument('--update', action='store_true', help='update all models')
    # 当前测试结果放在哪个主文件夹下 默认runs/detect
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    # 在上面runs/detect中,每次运行子文件夹名称
    parser.add_argument('--name', default='exp', help='save results to project/name')
    # 是否存在当前文件 默认False 一般是 no exist-ok 连用  所以一般都要重新创建文件夹
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    # 画框的线宽
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    # 画出的框框是否需要隐藏label信息 默认False, 即显示框的标签
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    # 画出的框框是否需要隐藏conf信息 默认False, 即显示框的置信度
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    # 是否使用半精度 Float16 推理 可以缩短推理时间 但是默认是False
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    # 对于到处的onnx文件, 用opencv自带的dnn模块进行推理
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    # 每个多少帧选一个视频帧图片
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))

    return opt


def main(opt):
    # 检查已经安装的包是否满足requirements对应txt文件的要求
    check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    # 执行run 开始推理
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
