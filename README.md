# TensorFlow Lite Task Library for Python

TensorFlow Lite Task Library for Python is a Python wrapper for Google's [TensorFlow Lite Task Library](https://github.com/tensorflow/tflite-support/tree/master/tensorflow_lite_support/cc/task). The wrapper is based on the C++ version of the TensorFlow Lite Task Library.

## Supported Tasks
Currently TensorFlow Lite Task Library for Python is only supported  Natural Language (NL) APIs provided by the official task library.

## Text Task Libraries

### QuestionAnswerer

    answerer = BertQuestionAnswerer.CreateFromFile(model_path)
    answers = answerer.Answer(context_of_question, question_to_ask)

### NLClassifier

    classifier = NLClassifier.CreateFromFileAndOptions(model_path, kInputTensorName, kOutputScoreTensorName)
    categories = classifier.Classify(kInput)

### BertNLClassifier

    classifier = BertNLClassifier.CreateFromFile(model_path)
    categories = classifier.Classify(kInput)

## Background

This library uses shared libraries which are built using `bazel`.  C++ APIs developed to create shared libraries (.so files) can be found [here](https://github.com/VihangaAW/tflite-support).  To bridge C++ and Python APIs,  [ctypes](https://docs.python.org/3/library/ctypes.html) which is a foreign function library for Python is used.