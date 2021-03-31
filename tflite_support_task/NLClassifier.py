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
      """

      CreateFromFileAndOptions is used to invoke the interpreter.

      Parameters:
      model_path               (string): tflite model path 
      input_tensor_name        (string): input tensor name
      output_score_tensor_name (string): output tensor name
      
      Returns: 
      NLClassifier: object 

      """
      return cls(model_path, input_tensor_name, output_score_tensor_name)

  def Classify(self,input):
      """

      Classify is used to execute the model with not preprocessed input and get the output.

      Parameters: 
      input                   (string): input text

      Returns
      list 

      """

      self.sharedLib.InvokeRunInference.restype = ctypes.c_char_p
      self.sharedLib.InvokeRunInference.argtypes = ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)

      string_buffers = [ctypes.create_string_buffer(8) for i in range(4)]
      pointers = (ctypes.c_char_p*4)(*map(ctypes.addressof, string_buffers))

      args = (ctypes.c_char_p * 1)(bytes(input, encoding='utf-8'))
      # Run the inference
      self.sharedLib.InvokeRunInference(len(args),args, pointers)
      results = [(s.value).decode('utf-8') for s in string_buffers]
      return results  


# Testing
timeStart = time.time()
nlClassifier = NLClassifier.CreateFromFileAndOptions("/tmp/movie_review.tflite","input_text","probability")
timeEnd = time.time()
executedTime = timeEnd-timeStart
print("Time spent for loading the model    : " + str(executedTime*1000)+" milliseconds")

timeStart = time.time()
output = nlClassifier.Classify("It was a great movie")
timeEnd = time.time()
executedTime = timeEnd-timeStart
print("Time spent for running the inference: " + str(executedTime*1000)+" milliseconds")
print("Output: " + str(output))