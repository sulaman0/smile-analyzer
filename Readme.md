# Smile Analyzer  

## Overview  
Smile Analyzer is a Python-based tool that detects a smile from a facial image and computes key metrics such as lip width, upper and lower lip width, and lip height. Using MediaPipe FaceMesh, the tool extracts key facial landmarks to analyze the smile and generates a smile score

## Features  
- Detects facial landmarks related to the lips
- Measures overall smile width and lip height.  
- Computes a smile score on a scale of 1-10.  
- Displays facial landmarks for visualization.  

## Installation  

Clone the repository and install dependencies:  

```bash
git clone https://github.com/sulaman0/smile-analyzer.git  
cd smile-analyzer  
pip install -r requirements.txt 

python opencv_smile_analysis.py
```

## Installation 

_Image found, processing..._

- **Overall Lip Width:** 45.67
- **Upper Lip Width:** 30.12
- **Lower Lip Width:** 32.45
- **Upper Lip Height:** 10.23
- **Lower Lip Height:** 12.78

**Smile Score:** 7.8
