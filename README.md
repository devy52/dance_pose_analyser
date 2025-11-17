# Dance Pose Analyser

A computer vision and machine learning toolkit for analyzing dance poses from video or image inputs. This repository leverages keypoint detection and pose estimation to help dancers and choreographers assess movement quality, alignment, and performance.

LIVE on : http://3.106.120.95:8000/

---

## Table of Contents

- [Features](#features)
- [Setup](#setup)
- [API Usage](#api-usage)
- [Deployment on AWS EC2](#deployment-on-aws-ec2)
- [Design & Thought Process](#design--thought-process)
- [Contributions](#contributions)
- [License](#license)

---

## Features

- Extracts pose keypoints from video or image sources
- Compares poses to reference data for feedback/scores
- Visualizes pose skeletons and analyses
- Extensible for new dance styles, feedback types, or custom scoring

---

## Setup

### Requirements

- Python 3.10+
- OpenCV (`opencv-python`)
- NumPy
- Matplotlib (optional, for visualization)
- mediapipe (for pose estimation)
- Any other dependencies listed in `requirements.txt`

### Local Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/devy52/dance_pose_analyser.git
   cd dance_pose_analyser
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify Setup**
   Try running the demo:
   ```bash
   python sample_dashboard.py

   or

   python main.py
   ```

---

## API Usage

(See full usage in prior markdown.)

---

## Deployment on AWS EC2

You can deploy and host this application on an [AWS EC2](https://aws.amazon.com/ec2/) instance for remote access and scalability.

### Steps to Deploy:

1. **Launch an EC2 Instance**
   - Recommended: Ubuntu Server (20.04 or later)
   - Choose instance type (e.g. t2.medium for moderate CPU use, GPU instance for faster inference)

2. **Connect to Your Instance**
   - Use SSH:
     ```bash
     ssh -i your-key.pem ubuntu@your-ec2-public-ip
     ```

3. **Install System Packages**
   ```bash
   sudo apt update
   sudo apt install python3-pip git
   # If using video inputs, also:
   sudo apt install ffmpeg
   ```

4. **Clone and Set Up the Repository**
   ```bash
   git clone https://github.com/devy52/dance_pose_analyser.git
   cd dance_pose_analyser
   pip3 install -r requirements.txt
   ```

5. **Run the Application**
   - For a command-line tool or script:
     ```bash
     python3 sample_dashboard.py

     or

     python3 main.py
     ```
   - For a web API (if using Flask/FastAPI):
     ```bash
     python3 server.py
     # Or with gunicorn/uvicorn for production
     ```

6. **Accessing Externally**
   - If running a web server, open the required port in EC2 security group (e.g., 80 or 5000)
   - Visit your server via `http://<EC2-public-IP>:<port>`

7. **(Optional) Set Up as a Service**
   - Use `systemd` or `screen`/`tmux` to keep the service running.

---

## Design & Thought Process

(See full section above.)

---

## Contributions

Contributions, issues, and feature requests are welcome.

---

## License

Distributed under the MIT License. See `LICENSE` for more information.

---

## References

- [AWS EC2 Setup Documentation](https://docs.aws.amazon.com/ec2/)
- [MediaPipe Pose Documentation](https://google.github.io/mediapipe/solutions/pose.html)
