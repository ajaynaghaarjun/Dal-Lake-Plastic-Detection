#Dal Lake Plastic Segmentation AI

An AI-powered dashboard to detect floating plastic debris in Dal Lake, Kashmir.
This project uses Deep Learning (U-Net with ResNet34) to segment plastic waste from water and vegetation.

#Features
- AI Detection: Distinguishes plastic from lotus/weeds with 0.67 IoU accuracy.
- Interactive Dashboard: built with Streamlit.
- Error Analysis: Visualizes False Positives and False Negatives.

#Tech Stack
- Model: PyTorch, Segmentation Models PyTorch
- App: Streamlit, OpenCV
- Data: Custom dataset of 2,000 annotated images.

#How to Run
1. Clone the repo:
   ```bash
   git clone [https://github.com/Adhi2485/Dal-Lake-Plastic-Detection.git](https://github.com/Adhi2485/Dal-Lake-Plastic-Detection.git)

2. Install requirements:
   pip install streamlit torch segmentation-models-pytorch opencv-python

3. Run the app:
   python -m streamlit run app.py
