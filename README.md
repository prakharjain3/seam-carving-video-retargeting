# Overview
In this project, we have done the following things:

- Implemented the seam-carving algorithm in papers Seam Carving for Content-Aware Image Resizing and Improved Seam Carving for Video Retargeting.

# Setup and Dependencies

First, ensure you have the necessary libraries installed:


```bash
pip install opencv-python numpy networkx
```

# Example usage of seam_carving_image.py

```bash
python seam_carving_image.py -f <input_image_path> -dh <desired_height> -dw <desired_width>
```

OR
    
```bash
python seam_carving_image.py --file <input_image_path> --desired_height <desired_height> --desired_width <desired_width>
```