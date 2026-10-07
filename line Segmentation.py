from flask import Flask, request,jsonify
from flask_cors import CORS

import cv2 
import numpy as np
import os
from math import sqrt
import doxapy



    
def binarization(img):
    window = 75
    k = 0.2
    # gray_img = get_gray_image(img_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary_image = np.empty(gray_img.shape, gray_img.dtype)
    sauvola = doxapy.Binarization(doxapy.Binarization.Algorithms.SAUVOLA)
    sauvola.initialize(gray_img)
    sauvola.to_binary(binary_image, {"window": window, "k": k})
    return binary_image


class ConnectedComponent:
    def __init__(self, top, bottom, left, right, height, width, pixel_values):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
        self.height = height
        self.width = width
        self.value = pixel_values
        

class lineInfo:
    def __init__(self, top, bottom, height, width, pixel_values):
        self.top = top
        self.bottom = bottom
        self.height = height
        self.width = width
        self.value = pixel_values
        
        
def create_line(line,bin_img):
    # Find the maximum bottom and minimum top values
    max_bottom = max(mb.bottom for mb in line)
    min_top = min(mb.top for mb in line)
    height = max_bottom - min_top
    width = bin_img.shape[0]
    
    line_image = np.ones(bin_img.shape, dtype="uint8") * 255
    for i in line:
        top, left = i.top, i.left
        line_image[top:top+i.value.shape[0], left:left+i.value.shape[1]][line_image[top:top+i.value.shape[0], left:left+i.value.shape[1]]!=0] = i.value[line_image[top:top+i.value.shape[0], left:left+i.value.shape[1]]!=0]

#     orignal_line = line_image[min_top:max_bottom][0:width]
    
    return lineInfo(min_top,max_bottom,height,width,line_image)
    

def calculate_top_distance(line_img,diac,bin_img):
    top_distance = []
    for i in range(diac.left, diac.right):
        arr = line_img.value[0:diac.top, i:i+1]
        # reverse the array
        arr_reverse = np.flip(arr)

        indices = np.where(arr_reverse == 0)[0]

        if len(indices) == 0:
            idx = len(arr)
        else:
            idx = indices[0]
        # print the index
        
        top_distance.append(idx)
        
    minimum_top_distance = -1 if not top_distance else min(filter(bool, top_distance))
    return minimum_top_distance


def calculate_bottom_distance(line_img,diac,bin_img):
    top_distance = []
    for i in range(diac.left, diac.right):
        arr = line_img.value[diac.bottom:len(bin_img), i:i+1]
        indices = np.where(arr == 0)[0]

        if len(indices) == 0:
            idx = len(arr)
        else:
            idx = indices[0]
        
        
        top_distance.append(idx)
        
    minimum_top_distance = -1 if not top_distance else min(filter(bool, top_distance))
    return minimum_top_distance





app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
# CORS(app)
@app.route('/image', methods=['GET', 'POST'])
def image():
    if request.method == 'POST':
        # Get the image from the request
        image = request.json
        # Convert the image from JSON to a NumPy array
        img = np.array(image).astype('uint8')
        
        bin_img=binarization(img)

        test_img=np.copy(bin_img)

        test_img[test_img == 0] = 1   # black pixels set to 1
        test_img[test_img == 255] = 0 # white pixels set to 0

        components = []
        # Find connected components and their statistics
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(test_img)

        # Loop over all connected components except the background
        for i in range(1, num_labels):
            # Create a mask for the connected component
            mask = np.uint8(labels == i) * 255
            
            # Crop the mask to the bounding box of the connected component
            x, y, w, h, area = stats[i]
            mask = mask[y:y+h, x:x+w]
            # select the elements that are equal to 0 or 255
            mask_invert = np.logical_or(mask == 0, mask == 255)

            # swap their values
            mask[mask_invert] = np.where(mask[mask_invert] == 0, 255, 0) 
            
            # Create a new ConnectedComponent object
            component = ConnectedComponent(top=y, bottom=y+h-1, left=x, right=x+w-1, height=h, width=w, pixel_values=mask)
            
            # Append the new object to the list
            components.append(component)


        components.sort(key=lambda x: x.height * x.width)

        # Compute the 33rd and 67th percentiles of the component areas
        areas = [c.height * c.width for c in components]
        p40 = np.percentile(areas, 70)


        # Categorize the components based on the computed percentiles
        diacritics = [c for c in components if c.height * c.width < p40]
        main_bodies = [c for c in components if c.height * c.width >= p40]

        main_bodies=sorted(main_bodies, key = lambda x: x.top)

        lines=list()
        start=0
        end=int()
        a=list()
        for i in range(0,len(main_bodies)):
            if i==len(main_bodies)-1:
                end=i+1
                lines.append(main_bodies[start:end])    
            else:
                if main_bodies[i].bottom<main_bodies[i+1].top:
                    end=i+1
                    lines.append(main_bodies[start:end])
                    start=end
                    
                    
        lines_values = list()
        for line in lines:
            lines_values.append(create_line(line,test_img))  
            
            
        not_containned = list()

        while(len(diacritics) != 0):
            found = 0
            flag = True
            
            diac = diacritics.pop(0)
            for lin in lines_values:
                if diac.top > lin.top and diac.bottom < lin.bottom:
                    
                    for row in range(diac.height):
                        for col in range(diac.width):
                            if diac.value[row][col] != 255:
                                lin.value[row + diac.top][col + diac.left] = diac.value[row][col]
                    flag = False
                
            if flag:
                not_containned.append(diac)
                
        #             break
            
            
            
        still = list()

        while (len(not_containned) != 0):
            
            diac = not_containned.pop(0)
            flag = True
            if diac.bottom <= lines_values[0].bottom:  ## diacritics that are above first line
                flag = False
                for row in range(diac.height):
                        for col in range(diac.width):
                            if diac.value[row][col] != 255:
                                lines_values[0].value[row + diac.top][col+diac.left] = diac.value[row][col]
                            
            elif diac.top >= lines_values[-1].top:  ## diacritics below the last line
                flag = False
                for row in range(diac.height):
                        for col in range(diac.width):
                            if diac.value[row][col] != 255:
                                lines_values[-1].value[row + diac.top][col+diac.left] = diac.value[row][col]
                            
            else:  # in between first and last line
                above = 0
                down = 0
                for l in range(len(lines_values)):
                    if diac.top <= lines_values[l].bottom:
                        above = len(set(range(diac.top,diac.bottom+1)).intersection(set(range(lines_values[l].top,lines_values[l].bottom+1))))
                        if above == 0:
                            continue
                            
                        
                        else:
                            if l < len(lines_values)-1:
                                bottom = len(set(range(diac.top,diac.bottom+1)).intersection(set(range(lines_values[l+1].top,lines_values[l+1].bottom+1))))
                                if above > bottom:
                                    flag = False
                                    for row in range(diac.height):
                                        for col in range(diac.width):
                                            if diac.value[row][col] != 255:
                                                lines_values[l].value[row + diac.top][col+diac.left] = diac.value[row][col]
                                else:
                                    flag = False
                                    for row in range(diac.height):
                                        for col in range(diac.width):
                                            if diac.value[row][col] != 255:
                                                lines_values[l+1].value[row + diac.top][col+diac.left] = diac.value[row][col]
                                                
            if flag:
                still.append(diac)
                
                
                
        more = list()
        while(len(still) != 0):
            flag = True
            diac = still.pop(0)
            for lin in range(len(lines_values)):
                
                if lin < len(lines_values)-1:
        #             print("ok!")
                    above = 0
                    down = 0
                    if diac.top >= lines_values[lin].bottom and diac.bottom <= lines_values[lin+1].top:
                        # print("ok!")
                        above = calculate_top_distance(lines_values[lin],diac,bin_img)
                        down = calculate_bottom_distance(lines_values[lin+1],diac,bin_img)
                        if above < down:
                            # print("ok!")
                            flag = False
                            for row in range(diac.height):
                                for col in range(diac.width):
                                    if diac.value[row][col] != 255:
                                        lines_values[lin].value[row + diac.top][col+diac.left] = diac.value[row][col]
                        else:
                            flag = False
                            # print("ok!!")
                            for row in range(diac.height):
                                for col in range(diac.width):
                                    if diac.value[row][col] != 255:
                                        lines_values[lin+1].value[row + diac.top][col+diac.left] = diac.value[row][col]
                                        
            if flag == True:
                more.append(diac)
                            
        # output_dir = r"D:\FYP\LINE_SEGMENTATION_IMPROVEMENT\out_put"
        ret_dic = dict()
        count = 1
        for i in lines_values:
            img = i.value.copy()
            
            # Find the rows that contain non-white pixels
            non_white_rows = np.any(img != 255, axis=1)

            # Index the original array with the non-white rows
            cropped_img = img[non_white_rows]
            
            # Define the number of pixels to add at the top and bottom
            padding_top = 10
            padding_bottom = 10

            # Define the padding values (white pixels)
            padding_value = 255

            # Add the padding to the image
            padded_img = np.pad(cropped_img, ((padding_top, padding_bottom), (0, 0)), mode='constant', constant_values=padding_value)
            ret_dic[str(count)] = padded_img.tolist()
            
            # cv2.imwrite(os.path.join(output_dir,f"final_lines_c{page}_{count}.jpg"),padded_img )
            count+=1     
        return jsonify(ret_dic)
    else:
        return 'Please send an image using POST request'

if __name__ == '__main__':
    app.run(debug=True,port=5002)