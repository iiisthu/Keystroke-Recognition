Introduction to keystroke recoginition

======

### About the project
This project is using inertial sensors to capture human typing behaviours and try to optimize the result of sensors as closer to keyboard typing as possible


### Data Capture
We are using the accelarator and gyroscope as external data collector. Using USB serial ports the datas are passed to PC and are converted to features with labels as our training data. Here the repo only deal with how to train those data with best output using machine learning and nature language processing 

### Classification method
* Multiclass SVM with cross validation(correct rate: 85%)
* Logestic Regression:(In process)
* Deep learning with theano:(In process)

### nature language model : spell checking
* Naive edit distance counting: Using google 'did you mean' method by Peter norwig (word correctness: 22%->28%)
