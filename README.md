# Attendance System

## Overview

The Attendance System is a Python-based application designed to streamline the process of recording and managing attendance using facial recognition technology. This system captures images of individuals, processes them to recognize faces, and maintains attendance records efficiently.

## Features

- **Facial Recognition**: Utilizes OpenCV and Haar Cascade classifiers to detect and recognize faces.
- **Attendance Logging**: Records attendance with timestamps for each recognized individual.
- **User-Friendly Interface**: Provides a simple interface for capturing images and managing attendance records.
- **Data Storage**: Stores images and attendance logs systematically for easy retrieval and analysis.

## Step 1: Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/therehanhussain/AttendanceSystem.git
   cd AttendanceSystem
   
# Step 2: Set Up a Virtual Environment

1. Open a terminal or command prompt.
2. Create a virtual environment by running the following command:
   ```bash
   python3 -m venv venv

# Step 3: Install Dependencies (save as `step3.md`)
```markdown
Install Dependencies

1. Ensure that your virtual environment is activated (refer to Step 2).
2. Run the following command to install the required Python libraries:
   ```bash
   pip install -r requirements.txt


---

Usage (save as `usage.md`)


Follow these steps to use the Attendance System:

Step 1: Capture Training Images
1. Run the following script to capture images for training the model:
   ```bash
   python main_Run.py


Directory Structure

The following is the directory structure for the project:

- **`Attendance/`**: Contains attendance logs in `.csv` format.
- **`StudentDetails/`**: Stores details of individuals, including their unique identifiers.
- **`TrainingImage/`**: Holds the images used for training the facial recognition model.
- **`TrainingImageLabel/`**: Contains labels corresponding to the training images.
- **`haarcascades/`**: Includes Haar Cascade XML files for face detection.
- **`main_Run.py`**: Script for capturing training images.
- **`training.py`**: Script for training the facial recognition model.
- **`mini_app.py`**: Main application script for running the attendance system.
- **`requirements.txt`**: Lists the Python dependencies required for the project.



Dependencies

The Attendance System requires the following Python libraries:

- **Python 3.x**: Core programming language for the system.
- **OpenCV**: For face detection and recognition.
- **NumPy**: For numerical computations.
- **Pandas**: For managing and storing attendance data.
- **Streamlit**: For building a simple user interface.

## Installation
To install these dependencies, activate your virtual environment and run:
```bash
pip install -r requirements.txt

