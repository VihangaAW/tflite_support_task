import ctypes
import time

# testlib = ctypes.CDLL('bazel-bin/tensorflow_lite_support/examples/task/text/desktop/libtestlib.so')
# testlib.myprint()

# testlib = ctypes.CDLL('bazel-bin/tensorflow_lite_support/examples/task/text/desktop/libtestlibone.so')
# testlib.myprint()

testlib = ctypes.CDLL('tflite_support_task/assets/libnl_classifier_demo.so')
testlib.randMethod()


# pData = (ctypes.c_char_p * 5)()
# pData[0] = "./nl_classifier_demo"
# pData[1] = "-model_path=/tmp/movie_review.tflite"
# pData[2] = "--text=What a waste of my time."
# pData[3] = "--input_tensor_name=input_text"
# pData[4] = "--output_score_tensor_name=probability"

testlib.mainOne.restype = None
testlib.mainOne.argtypes = ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)
timeStart = time.time()
args = (ctypes.c_char_p * 5)(b"./nl_classifier_demo", b"-model_path=/tmp/movie_review.tflite", b"--text=What a waste of my time.", b"--input_tensor_name=input_text", b"--output_score_tensor_name=probability")
testlib.mainOne(len(args),args)
timeEnd = time.time()
executedTime = timeEnd-timeStart
print(str(executedTime*1000)+" milliseconds")
# pData = ["./nl_classifier_demo", "-model_path=/tmp/movie_review.tflite", "--text=What a waste of my time.", "--input_tensor_name=input_text", "--output_score_tensor_name=probability"]

# testlib.mainOne(5, pData)


# 5
# ./nl_classifier_demo
# --model_path=/tmp/movie_review.tflite
# --text=What a waste of my time.
# --input_tensor_name=input_text
# --output_score_tensor_name=probability
# testlib = ctypes.CDLL('/home/vihanga/Vihanga/Tflite/tflite-support/bazel-bin/tensorflow_lite_support/examples/task/text/desktop/libbert_question_answerer_demo.so')


# testlib.mainOne.restype = None
# testlib.mainOne.argtypes = ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)
# timeStart = time.time()
# args = (ctypes.c_char_p * 4)(b"./bert_question_answerer_demo", b"--model_path=/tmp/mobilebert.tflite", b"--question='Where is Amazon rainforest?'", b"--context='The Amazon rainforest, alternatively, the Amazon Jungle, also known in English as Amazonia, is a moist broadleaf tropical rainforest in the Amazon biome that covers most of the Amazon basin of South America. This basin encompasses 7,000,000 km2 (2,700,000 sq mi), of which 5,500,000 km2 (2,100,000 sq mi) are covered by the rainforest. This region includes territory belonging to nine nations.'")
# testlib.mainOne(len(args),args)
# timeEnd = time.time()
# executedTime = timeEnd-timeStart
# print(str(executedTime*1000)+" milliseconds")