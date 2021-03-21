import argparse
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import cv2
import sys
import numpy
from socket import *
#socket.socket
#from socket import *
#socket
#import socket
#socket.socket

from core.utils import load_class_names, load_image, draw_boxes, draw_boxes_frame
from core.yolo_tiny import YOLOv3_tiny
from core.yolo import YOLOv3

#ソケット通信用
# 送信側アドレスの設定
# 送信側IP(AWS)
#SrcIP = "127.0.0.1"
#SrcIP = "54.248.64.253"
SrcIP = "0.0.0.0"
# 送信側ポート番号
SrcPort = 8888
# 送信側アドレスをtupleに格納
SrcAddr = (SrcIP,SrcPort)
BUFSIZE = 1024

# ソケット作成
udpServSock = socket(AF_INET, SOCK_DGRAM)
# 受信側アドレスでソケットを設定
udpServSock.bind(SrcAddr)
print('bind:clear')

def receive():
    # 送信データの作成
  print('ready')
  data, addr = udpServSock.recvfrom(BUFSIZE*60) 
  print(addr)
  # バイナリを逆変換
  narray = numpy.frombuffer(data, dtype='uint8')
  decode = cv2.imdecode(narray,1)
  return decode, addr

def main(mode, tiny, iou_threshold, confidence_threshold, path):
  class_names, n_classes = load_class_names()
  if tiny:
    model = YOLOv3_tiny(n_classes=n_classes,
                        iou_threshold=iou_threshold,
                        confidence_threshold=confidence_threshold)
  else:
    model = YOLOv3(n_classes=n_classes,
                   iou_threshold=iou_threshold,
                   confidence_threshold=confidence_threshold)
  inputs = tf.placeholder(tf.float32, [1, *model.input_size, 3])
  detections = model(inputs)
  saver = tf.train.Saver(tf.global_variables(scope=model.scope))

  with tf.Session() as sess:
    saver.restore(sess, './weights/model-tiny.ckpt' if tiny else './weights/model.ckpt')

    if mode == 'image':
      image = load_image(path, input_size=model.input_size)
      result = sess.run(detections, feed_dict={inputs: image})
      draw_boxes(path, boxes_dict=result[0], class_names=class_names, input_size=model.input_size)
      return

    elif mode == 'video':
      cv2.namedWindow("Detections")
      video = cv2.VideoCapture(path)
      fourcc = int(video.get(cv2.CAP_PROP_FOURCC))
      fps = video.get(cv2.CAP_PROP_FPS)
      frame_size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
      out = cv2.VideoWriter('./detections/video_output.mp4', fourcc, fps, frame_size)
      print("Video being saved at \"" + './detections/video_output.mp4' + "\"")
      print("Press 'q' to quit")
      while True:
        retval, frame = video.read()
        if not retval:
          break
        resized_frame = cv2.resize(frame, dsize=tuple((x) for x in model.input_size[::-1]), interpolation=cv2.INTER_NEAREST)
        result = sess.run(detections, feed_dict={inputs: [resized_frame]})
        draw_boxes_frame(frame, frame_size, result, class_names, model.input_size)
        cv2.imshow("Detections", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        out.write(frame)
      cv2.destroyAllWindows()
      video.release()
      return

    elif mode == 'webcam':
#      cap = cv2.VideoCapture(0)
      while True:
#        ret, frame = cap.read()
        frame, addr = receive()
        frame_size = (frame.shape[1], frame.shape[0])
        resized_frame = cv2.resize(frame, dsize=tuple((x) for x in model.input_size[::-1]), interpolation=cv2.INTER_NEAREST)
        result = sess.run(detections, feed_dict={inputs: [resized_frame]})
        draw_boxes_frame(frame, frame_size, result, class_names, model.input_size)
        jpgstring = cv2.imencode(".jpg", frame)
        packet = jpgstring[1].tostring()
        udpServSock.sendto(packet, addr)
#        cv2.imshow('frame', frame)
#        if cv2.waitKey(1) & 0xFF == ord('q'):
#          break
#      cap.release()
#      cv2.destroyAllWindows()
      return

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--tiny", action="store_true", help="enable tiny model")
  parser.add_argument("mode", choices=["video", "image", "webcam"], help="detection mode")
  parser.add_argument("iou", metavar="iou", type=float, help="IoU threshold [0.0, 1.0]")
  parser.add_argument("confidence", metavar="confidence", type=float, help="confidence threshold [0.0, 1.0]")
  if 'video' in sys.argv or 'image' in sys.argv:
    parser.add_argument("path", type=str, help="path to file")

  args = parser.parse_args()
  if args.mode == 'video' or args.mode == 'image':
    main(args.mode, args.tiny, args.iou, args.confidence, args.path)
  else:
    main(args.mode, args.tiny, args.iou, args.confidence, '')
