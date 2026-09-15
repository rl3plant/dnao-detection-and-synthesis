# DNAO Contour and Mask Dataset

## Properties

## Shapes
Rect + Tri + Z

## Images
- Images of AFM images are preprocessed accordingly and scaled to 8-bit grayscale images (0, 255).
- Mask images depict the background as 0 (black) and masked regions as 255 (white).

## Contours
- One .txt per holds all shape or mask contours for an equally named image file 
- One line per detected DNAO shape
- Each line contains the shape index and vertices, all seperated by semicolons ```;```.
- The vertex coordinates X, Y are separated by a blank space ``` ```.  

Example: ```0; 221 -2; 251 -2; 251 18; 221 18```  
The contour of shape ```0``` has vertices on coordinates (221, -2), (251, -2), (251, 18), (221,18).
