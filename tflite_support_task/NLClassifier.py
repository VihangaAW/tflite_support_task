import ctypes
import time

class NLClassifier:
  def __init__(self, model_path, input_tensor_name, output_score_tensor_name):
    self.model_path = model_path
    self.input_tensor_name = input_tensor_name
    self.output_score_tensor_name = output_score_tensor_name
    self.sharedLib = ctypes.CDLL('tflite_support_task/assets/libinvoke_nl_classifier.so')

    # Initialize a classifier
    self.sharedLib.InvokeInitializeModel.restype = None
    self.sharedLib.InvokeInitializeModel.argtypes = ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)
    args = (ctypes.c_char_p * 4)(bytes(model_path, encoding='utf-8'), bytes(input_tensor_name, encoding='utf-8'), bytes(output_score_tensor_name, encoding='utf-8'))
    self.sharedLib.InvokeInitializeModel(len(args),args)



  @classmethod
  def CreateFromFileAndOptions(cls, model_path, input_tensor_name, output_score_tensor_name):
      return cls(model_path, input_tensor_name, output_score_tensor_name)

  def Classify(self,input):
      self.sharedLib.InvokeRunInference.restype = None
      self.sharedLib.InvokeRunInference.argtypes = ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)
      args = (ctypes.c_char_p * 1)(bytes(input, encoding='utf-8'))
      self.sharedLib.InvokeRunInference(len(args),args)
      return input  

timeStart = time.time()
a = NLClassifier.CreateFromFileAndOptions("/tmp/movie_review.tflite","input_text","probability")
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