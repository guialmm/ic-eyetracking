
# Real-Time Eye Movement Tracking for Hands-Free Mouse Control using Deep Learning

This project aims to develop a hands-free mouse control system using real-time eye movement tracking, leveraging deep learning techniques to improve the accuracy and responsiveness of gaze estimation. The system allows users to control the cursor based on eye movement, offering a practical solution for individuals with disabilities or limited mobility.

## Overview

With the increasing reliance on digital technologies, it has become essential to ensure that individuals with disabilities can interact with computers and mobile devices. Traditional input devices such as a mouse can be difficult to use for individuals with motor impairments. This project presents a solution that utilizes **eye tracking** to enable hands-free control of a computer's mouse pointer.

The system employs **real-time eye tracking** using a standard camera and deep learning models to enhance the accuracy of gaze estimation. It provides an alternative input method for individuals with disabilities while also improving the user experience by offering a more natural and intuitive interface.

## Key Features

- **Real-time eye tracking**: The system tracks eye movement using a common camera, eliminating the need for expensive specialized equipment.
- **Deep learning**: A simple neural network model is used to improve gaze estimation accuracy, allowing for smoother and more precise cursor control.
- **Hands-free control**: Users can control the mouse pointer using their eye movement, offering an alternative to traditional input methods for those with limited mobility.
- **Calibration process**: The system uses five fixed calibration points on the screen to improve tracking precision, with a **triple-loop calibration** approach to minimize errors.

## Libraries and Tools

The system is built with the following Python libraries:

- **OpenCV (`cv2`)**: Used for video capture and image processing.
- **PyAutoGUI**: Provides the functionality to control the mouse based on eye movements.
- **GazeTracking**: A library used for real-time eye position detection.
- **TensorFlow / Keras**: For implementing and training the neural network model.

## System Workflow

1. **Calibration**: The system performs three calibration trials using five fixed points on the screen: top-left, top-right, center, bottom-left, and bottom-right corners. The calibration process calculates the average eye position for each point to improve accuracy.

2. **Real-Time Eye Tracking**: Once calibrated, the system tracks the user's eye movements in real time, controlling the mouse pointer based on the detected eye position.

3. **Deep Learning Model**: A simple neural network model is applied to the eye tracking data to enhance gaze estimation accuracy and improve the responsiveness of the system. The neural network was trained using a dataset of 2300 images, and the model consists of three fully connected layers.

## Methodology

The system employs the **GazeTracking** library for detecting eye movements and uses **PyAutoGUI** to control the mouse cursor. The calibration process, which is repeated three times, helps refine the eye position data and ensures more accurate tracking. Additionally, a **neural network** model is incorporated to improve the precision of the eye tracking system.

The neural network model is trained with the following parameters:

- **Batch size**: 32
- **Epochs**: 150
- **Learning rate**: \(10^{-4}\) (fixed learning rate)
- **Validation split**: 20% for validation during training
- **Target image size**: 224x224 pixels

## Evaluation and Results

The system was evaluated based on:

- **Accuracy**: The angular degree of error between the estimated eye position and the actual eye position.
- **Responsiveness**: The system's ability to control the mouse cursor in real time, including sensitivity adjustments and smoothing of movements.

Results showed that the system was able to effectively control the mouse pointer, offering a viable alternative input method for individuals with disabilities or limited mobility.

## Conclusion

This project demonstrates an innovative solution for hands-free mouse control using real-time eye tracking and deep learning. By utilizing a common camera and eliminating the need for specialized hardware, the system provides an accessible and affordable solution to assistive technology. Future work will focus on improving the model's accuracy, expanding its capabilities, and exploring interaction through additional eye gestures.

## Future Work

- Explore more complex deep learning models to further enhance accuracy.
- Collect additional data to improve precision.
- Implement new interaction features, such as blink detection for mouse clicks.

## Installation

To run the system, you need the following dependencies:

```bash
pip install opencv-python pyautogui gaze_tracking tensorflow
```

Once the dependencies are installed, simply run the Python script to start the eye tracking system.

## Download the Article

You can access the full article here:

[Download the PDF of the Article](docs/Artigo_EyesOn_Guilherme-1.pdf)

## License

This project is **NOT** open source. All rights are reserved, and no part of this code may be used, modified, or redistributed without the author's explicit permission.

For inquiries regarding the use of this code, please contact the author at [guilherme.almeida@ges.inatel.br].

