import ctypes
import time

class BertQuestionAnswerer:
  def __init__(self, model_path):
    self.model_path = model_path
    self.sharedLib = ctypes.CDLL('./assets/libinvoke_bert_question_answerer.so')

    # Initialize a classifier
    self.sharedLib.InvokeInitializeModel.restype = None
    self.sharedLib.InvokeInitializeModel.argtypes = ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)
    args = (ctypes.c_char_p * 1)(bytes(model_path, encoding='utf-8'))
    self.sharedLib.InvokeInitializeModel(len(args),args)



  @classmethod
  def CreateFromFile(cls, model_path):
      return cls(model_path)

  def Answer(self,context_of_question, question_to_ask):
    self.sharedLib.InvokeRunInference.restype = None
    self.sharedLib.InvokeRunInference.argtypes = ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)
    args = (ctypes.c_char_p * 2)(bytes(context_of_question, encoding='utf-8'),bytes(question_to_ask, encoding='utf-8'))
    self.sharedLib.InvokeRunInference(len(args),args)
    return question_to_ask

timeStart = time.time()
a = BertQuestionAnswerer.CreateFromFile("/tmp/mobilebertqa.tflite")
timeEnd = time.time()
executedTime = timeEnd-timeStart
print(str(executedTime*1000)+" milliseconds")

timeStart = time.time()
b = a.Answer("The Amazon rainforest, alternatively, the Amazon Jungle, also known in  English as Amazonia, is a moist broadleaf tropical rainforest in the Amazon biome that covers most of the Amazon basin of South America. This basin encompasses 7,000,000 km2 (2,700,000 sq mi), of which 5,500,000 km2 (2,100,000 sq mi) are covered by the rainforest. This region includes territory belonging to nine nations","Where is Amazon rainforest?")
timeEnd = time.time()
executedTime = timeEnd-timeStart
print(str(executedTime*1000)+" milliseconds")
print(str(b))

print(a.model_path)      
