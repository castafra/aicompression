# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 17:07:47 2021

@author: samze
"""
from PIL import Image, ImageDraw, ImageChops
import pandas as pd


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
        draw_slide.rectangle([(df['box'].loc[i][1]-10,df['box'].loc[i][0]-10),(df['box'].loc[i][3]+10, df['box'].loc[i][2]+10)] , (r,g,b) )
    slide.show()
    return ()

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


def fill_black_1(detected_objects, image):
    draw_slide=ImageDraw.Draw(image)
    for i in range(len(detected_objects)):
        x=detected_objects['box'].loc[i][1]
        y=detected_objects['box'].loc[i][0]
        r0, b0, g0 = image.getpixel(x,y)
        vert_band_not_found = True
        hor_band_not_found = True
        #hypothesis : max 1 color band per box
        while x< detected_objects['box'].loc[i][3] and vert_band_not_found:
            r, g, b = image.getpixel((x, y))
            vert_band_not_found=vert_band_not_found and (r, g, b == image.getpixel((x+1, y)))
            x+=1
        if vert_band_not_found:
            """draw_slide.rectangle([(detected_objects['box'].loc[i][1],detected_objects['box'].loc[i][0]),
                                  (detected_objects['box'].loc[i][3], detected_objects['box'].loc[i][2])] , (r,g,b) )
        #add vertical band research
        else:"""
            while y< detected_objects['box'].loc[i][2] and hor_band_not_found:
                r, g, b = image.getpixel((x, y))
                hor_band_not_found = hor_band_not_found and (r, g, b == image.getpixel((x, y+1)))
                y+=1
            if not hor_band_not_found:
                draw_slide.rectangle([(detected_objects['box'].loc[i][1],
                                           detected_objects['box'].loc[i][0]),
                                              (detected_objects['box'].loc[i][3],
                                                   y)],
                                                       (r0, b0, g0) )
                draw_slide.rectangle([(detected_objects['box'].loc[i][1],y+1),
                                          (detected_objects['box'].loc[i][3],
                                               detected_objects['box'].loc[i][2])],
                                                    (r,g,b) )
            else: #a check
                draw_slide.rectangle([(detected_objects['box'].loc[i][1],
                                 detected_objects['box'].loc[i][0]),
                                     (x, detected_objects['box'].loc[i][2])],
                                         (r0,g0,b0) )
                
        else:
            draw_slide.rectangle([(detected_objects['box'].loc[i][1],
                                 detected_objects['box'].loc[i][0]),
                                     (x, detected_objects['box'].loc[i][2])],
                                         (r0,g0,b0) )

            r,g,b= image.getpixel((x+1, y))
            draw_slide.rectangle([(x+1,detected_objects['box'].loc[i][0]),
                                  (detected_objects['box'].loc[i][3], 
                                   detected_objects['box'].loc[i][2])] , (r,g,b) )
        

        image.show()
    return None        
        
point_table = ([0] + ([255] * 255))

def black_or_b(a, b):
    diff = ImageChops.difference(a, b)
    diff = diff.convert('L')
    diff = diff.point(point_table)
    new = diff.convert('RGB')
    new.paste(b, mask=diff)
    return new

a = Image.open('a.png')
b = Image.open('b.png')
c = black_or_b(a, b)
c.save('c.png')






    
    
    