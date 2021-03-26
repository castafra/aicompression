# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 17:07:47 2021

@author: samze
"""
from PIL import Image, ImageDraw
import pandas as pd

#output= array([xmin1,ymin1,xmax1,ymax1],[xmin2,ymin2,ymin2,ymax2])
'''detection_classes': array([0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 2, 2, 0, 2, 0, 1, 0, 2, 0, 2, 2, 2,
       0, 0, 2, 1, 1, 0, 2, 0, 2, 0, 0, 1, 2, 2, 0, 2, 0, 2, 2, 0, 2, 0,
       0, 2, 2, 0, 0, 2, 2, 0, 2, 0, 0, 1, 2, 2, 0, 0, 2, 0, 0, 2, 0, 0,
       2, 0, 1, 2, 2, 2, 0, 2, 2, 0, 2, 0, 0, 0, 2, 0, 1, 0, 0, 2, 2, 0,
       0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 2, 0]'''

data = {'box': [[223,287,707,655],[274,1117,530,1749], [130, 1291, 210, 1390]], 'class': ['image', 'text', 'logo']}
df_test= pd.DataFrame(data)
img=Image.open("C:\\Users\\samze\\Pictures\\test.png")
#output = [[ymin1,xmin1,ymax1,xmax1],...]
#output=[[223,287,707,655], [274,1117,530,1749], [130, 1291, 210, 1390]]

def output_from_df(df):
    output=[]
    classes=[]
    #verify the threshold
    for i in range(len(df)):
        output.append(df['box'].loc[i])
        classes.append(df['class'].loc[i])
    return output, classes


def remove_elements(out, slide, classes):
    draw_slide=ImageDraw.Draw(slide)
    for i in range(len(out)):
        draw_slide.rectangle([(out[i][1],out[i][0]),(out[i][3],out[i][2])] , '#000000' )
    slide.show()
    return ('slide')
    
  
#version 0, hypothesis same color
def fill_black_0(out, slide, classes):
    draw_slide=ImageDraw.Draw(slide)
    for i in range(len(out)):
        r, g, b = slide.getpixel(((out[i][1]+out[i][3])//2 , out[i][0]-1 ))
        draw_slide.rectangle([(out[i][1],out[i][0]),(out[i][3],out[i][2])] , (r,g,b) )
    slide.show()
    return slide  

def retrieve_background(df,slide):
    draw_slide=ImageDraw.Draw(slide)
    for i in range(len(df)):
        r, g, b = slide.getpixel(((df['box'].loc[i][1]+df['box'].loc[i][3])//2 , df['box'].loc[i][0]-1 ))
        draw_slide.rectangle([(df['box'].loc[i][1],df['box'].loc[i][0]),(df['box'].loc[i][3], df['box'].loc[i][2])] , (r,g,b) )
    slide.show()
    return slide
output, classes=output_from_df(df_test)
remove_elements(output,img,classes)

#---------------------------- BROUILLON ---------------------------------------

def remove_elements_0(out, slide, classes):
    width, height = slide.size
    BLACK_THRESHOLD =   1
    for i in range(len(out)):
    #remove all items except text
        if classes[i] != 0: 
            for x in range(out[i][0],out[i][1]):
                for y in range (out[i][2],out[i][3]):
                    #pixel = img.getpixel((x, y))
                    slide.putpixel((x, y), BLACK_THRESHOLD)
    slide.show()
    #identify forms 
    return slide

    
def fill_black_1(out, slide, classes):
    for i in range(len(out)):
        pixel_up = slide.getpixel((out[i][0]+out[i][2])//2, out[i][1]-1)
        pixel_down = slide.getpixel((out[i][0]+out[i][2])//2, out[i][3]+1)
        pixel_left = slide.getpixel(out[i][0]-1, (out[i][1]+out[i][3])//2)
        pixel_right = slide.getpixel(out[i][2]+1, (out[i][1]+out[i][3])//2)
        for x in range(out[i][0],out[i][1]):
                for y in range (out[i][2],out[i][3]):
                    slide.putpixel((x, y), 1)
    return 'ok'
        
        
                       
            
            






    
    
    