import ctypes
import time

class BertNLClassifier:
  def __init__(self, model_path):
    self.model_path = model_path
    self.sharedLib = ctypes.CDLL('tflite_support_task/assets/libinvoke_bert_nl_classifier.so')

    # Initialize a classifier
    self.sharedLib.InvokeInitializeModel.restype = None
    self.sharedLib.InvokeInitializeModel.argtypes = ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)
    args = (ctypes.c_char_p * 1)(bytes(model_path, encoding='utf-8'))
    self.sharedLib.InvokeInitializeModel(len(args),args)



  @classmethod
  def CreateFromFile(cls, model_path):
      return cls(model_path)

  def Classify(self,input):
      self.sharedLib.InvokeRunInference.restype = None
      self.sharedLib.InvokeRunInference.argtypes = ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)
      args = (ctypes.c_char_p * 1)(bytes(input, encoding='utf-8'))
      self.sharedLib.InvokeRunInference(len(args),args)
      return input  

timeStart = time.time()
a = BertNLClassifier.CreateFromFile("/tmp/movie_review.tflite")
timeEnd = time.time()
executedTime = timeEnd-timeStart
print(str(executedTime*1000)+" milliseconds")

timeStart = time.time()
b = a.Classify("It was a great movie")
timeEnd = time.time()
executedTime = timeEnd-timeStart
print(str(executedTime*1000)+" milliseconds")
print(str(b))

print(a.model_path)      
print(a.Classify("Hello World"))