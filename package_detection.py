import argparse
import time, cv2

from PIL import Image
import collections

import classify
import tflite_runtime.interpreter as tflite
import os
from google.cloud import storage

from imutils import keyclipwriter
import datetime

buf_len = 50

client = storage.Client()
bucket = client.get_bucket("nomi-kb.appspot.com")

EDGETPU_SHARED_LIB = "libedgetpu.so.1"


def load_labels(path, encoding="utf-8"):
    """Loads labels from file (with or without index numbers).

  Args:
    path: path to label file.
    encoding: label file encoding.
  Returns:
    Dictionary mapping indices to labels.
  """
    with open(path, "r", encoding=encoding) as f:
        lines = f.readlines()
        if not lines:
            return {}

        if lines[0].split(" ", maxsplit=1)[0].isdigit():
            pairs = [line.split(" ", maxsplit=1) for line in lines]
            return {int(index): label.strip() for index, label in pairs}
        else:
            return {index: line.strip() for index, line in enumerate(lines)}


def check_buf(b):
    # pass
    if b.count(b[0]) > buf_len * 0.7:
        return b[0]


def make_interpreter(model_file):
    model_file, *device = model_file.split("@")
    return tflite.Interpreter(
        model_path=model_file,
        experimental_delegates=[
            tflite.load_delegate(
                EDGETPU_SHARED_LIB, {"device": device[0]} if device else {}
            )
        ],
    )


def make_interpreter(model_file):
    model_file, *device = model_file.split("@")
    return tflite.Interpreter(
        model_path=model_file,
        experimental_delegates=[
            tflite.load_delegate(
                EDGETPU_SHARED_LIB, {"device": device[0]} if device else {}
            )
        ],
    )


def main():
    dir_loc = "/home/pi/Documents/samsung/package_detection/test_model/"
    img_loc = "/home/pi/Documents/samsung/package_detection/test_model/image.jpg"
    vid_loc = "/home/pi/Documents/samsung/package_detection/test_model/test.mp4"
    prediction = "na"

    frames_record = 64
    fourcc = cv2.VideoWriter_fourcc(*"MP4V")

    kcw = keyclipwriter.KeyClipWriter(frames_record)
    consecFrames = 0

    buffer_prediction = collections.deque(maxlen=buf_len)
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-m", "--model", required=True, help="File path of .tflite file."
    )
    parser.add_argument("-i", "--input", required=True, help="Image to be classified.")
    parser.add_argument("-l", "--labels", help="File path of labels file.")
    parser.add_argument(
        "-k",
        "--top_k",
        type=int,
        default=1,
        help="Max number of classification results",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.0,
        help="Classification score threshold",
    )
    parser.add_argument(
        "-c", "--count", type=int, default=5, help="Number of times to run inference"
    )
    args = parser.parse_args()

    labels = load_labels(args.labels) if args.labels else {}

    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()

    size = classify.input_size(interpreter)

    # while True:
    cap = cv2.VideoCapture(0)
    count_pred = ""
    while True:
        updateConsecFrames = True
        ret, image_webcam = cap.read()
        image_orig = image_webcam

        frame = image_webcam[220:400, 230:470]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(frame)
        image = pil_im.resize(size, Image.ANTIALIAS)
        classify.set_input(interpreter, image)
        interpreter.invoke()
        classes = classify.get_output(interpreter, args.top_k, args.threshold)
        str_c = ""
        for klass in classes:
            str_c = labels.get(klass.id, klass.id)  # + str(klass.score)
            buffer_prediction.append(str_c)
            tmp_pred = check_buf(buffer_prediction)
            if 1 == 1:
                count_pred = tmp_pred
                if (
                    count_pred is not prediction
                    and count_pred is not None
                    and count_pred is not "no_package"
                ):
                    prediction = count_pred
                    consecFrames = 0
                    cv2.imwrite(img_loc, image_orig)
                    msg = count_pred + "_" + str(int(time.time()))
                    print("uploaded with prediction", count_pred)
                    if not kcw.recording:
                        timestamp = datetime.datetime.now()
                        # p = "{}_{}.avi".format(count_pred,timestamp.strftime("%Y%m%d-%H%M%S"))
                        # print(p)
                        global p
                        p = count_pred + "_" + str(int(time.time())) + ".avi"

                        # zebraBlob = bucket.blob('package_delivery/video/'+p)
                        # zebraBlob.upload_from_filename(filename=dir_loc+p)
                        print(dir_loc + " " + p)
                        kcw.start(dir_loc + p, cv2.VideoWriter_fourcc(*"MJPG"), 15)
                # print("recording started")
                if updateConsecFrames:
                    consecFrames += 1

                # update the key frame clip buffer
                kcw.update(image_orig)

                if kcw.recording and consecFrames == frames_record:
                    for i in range(50):
                        kcw.update(image_orig)
                    kcw.finish()

                    video_fs_loc = dir_loc + p



            # print( count_pred)tmp_pred is not None
        cv2.putText(
            image_webcam,
            tmp_pred,
            (10, 40),
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            1,
            (0, 255, 0),
            2,
        )
        cv2.imshow("my webcam", image_webcam)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            exit
        # for klass in classes:
        # print("%s: %.5f" % (labels.get(klass.id, klass.id), klass.score))

    # print("----INFERENCE TIME----")
    # for _ in range(args.count):
    #     start = time.monotonic()
    #     interpreter.invoke()
    #     inference_time = time.monotonic() - start
    #     classes = classify.get_output(interpreter, args.top_k, args.threshold)
    #     print("%.1fms" % (inference_time * 1000))

    # print("-------RESULTS--------")
    # for klass in classes:
    #     print("%s: %.5f" % (labels.get(klass.id, klass.id), klass.score))


def show_webcam(mirror=False):
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, img = cam.read()
        img = img[220:400, 230:470]
        if mirror:
            img = cv2.flip(img, 1)
        cv2.imshow("my webcam", img)
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()


def capture_image():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    frame = frame[220:400, 230:470]
    cv2.imshow("window", frame)
    # image_hub.send_reply(b'OK')


if __name__ == "__main__":
    # show_webcam(mirror=True)
    # capture_image()
    main()