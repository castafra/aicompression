from PIL import Image, ImageFont, ImageDraw
from transformers import pipeline, set_seed
import textwrap
import pandas as pd
import random
import colorsys
import math
import os
from xml.etree.ElementTree import Element, SubElement, Comment, tostring
from xml.etree import ElementTree
from xml.dom import minidom
import codecs
from tqdm import tqdm


generator = pipeline('text-generation', model='sshleifer/tiny-gpt2', device=0)


words = pd.read_csv("common_words.csv")

slide_types = ['illustration','description','explanation']
fonts = os.listdir('fonts')

words_list = words.list

def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def hsv2rgb(h, s, v):
    h = float(h)
    s = float(s)
    v = float(v)
    h60 = h / 60.0
    h60f = math.floor(h60)
    hi = int(h60f) % 6
    f = h60 - h60f
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r, g, b = 0, 0, 0
    if hi == 0: r, g, b = v, t, p
    elif hi == 1: r, g, b = q, v, p
    elif hi == 2: r, g, b = p, v, t
    elif hi == 3: r, g, b = p, q, v
    elif hi == 4: r, g, b = t, p, v
    elif hi == 5: r, g, b = v, p, q
    r, g, b = int(r * 255), int(g * 255), int(b * 255)
    return r, g, b

def generate_text(length = 50, maxi = 100):
    N = len(words_list)
    pos = random.randint(0,N-1)
    text = generator(words_list[pos], min_length=length, max_length = maxi, num_return_sequences=1)
    #text = [{"generated_text" : "Hello i am the betst in the world how are you my firend i am happy to meet you!"}]
    return text[0]['generated_text'].replace('\n','')

def initiazalize_xml(name,path_file,size,ratio):
    annotation = Element('annotation')
    folder = SubElement(annotation,'folder')
    folder.text = 'Slide Generator'
    filename = SubElement(annotation,'filename')
    filename.text = name + '.jpg'
    path = SubElement(annotation,'path')
    path.text = path_file + name + ".jpg"
    source = SubElement(annotation,'source')
    database = SubElement(source,"database")
    database.text = "Unknown"
    size_el = SubElement(annotation,"size")
    width_el = SubElement(size_el,"width")
    width_el.text = str(size)
    height_el = SubElement(size_el,"height")
    height_el.text = str(int(ratio*size))
    depth_el = SubElement(size_el,"width")
    depth_el.text = '3'
    segmented = SubElement(annotation,'segmented')
    segmented.text = '0'
    return annotation

def add_xml_object(type,elem, xmin, ymin,xmax,ymax):
    object = SubElement(elem,'object')
    name_obj = SubElement(object,'name')
    name_obj.text = type
    pose = SubElement(object,'pose')
    pose.text = 'Unspecified'
    truncated = SubElement(object,'truncated')
    truncated.text = '0'
    difficult = SubElement(object,'difficult')
    difficult.text = '0'
    bndbox = SubElement(object,'bndbox')
    xm = SubElement(bndbox,'xmin')
    xm.text = str(int(xmin))
    ym = SubElement(bndbox,'ymin')
    ym.text = str(int(ymin))
    xM = SubElement(bndbox,'xmax')
    xM.text = str(int(xmax))
    yM = SubElement(bndbox,'ymax')
    yM.text = str(int(ymax))

def create_slide(ratio = 2/3, size = 1920, name = "slide1", type = "illustration"):
    #Initialize xml file
    annotation = initiazalize_xml(name = name, path_file = "C:\\Users\\François\\Documents\\GitHub\\aicompression\\Slide Generator\\images_generated\\",size = size, ratio = ratio)


    font_chosen = fonts[random.randint(0,len(fonts)-1)]
    ## Objects import 
    logos = os.listdir('sample_logo')
    logo_path = 'sample_logo/'+ logos[random.randint(0,len(logos)-1)]
    print(logo_path)
    logo = Image.open(logo_path,'r')
    if random.random() > 0.5 : 
        logo = logo.rotate(180)

    images = os.listdir('sample_images')
    image1_path = 'sample_images/'+ images[random.randint(0,len(images)-1)]
    image1 = Image.open(image1_path,'r')
    if type == 'illustration':
        image2_path = 'sample_images/'+ images[random.randint(0,len(images)-1)]
        image2 = Image.open(image2_path,'r')

    text1 = generate_text()
    if type == "explanation":
        text2 = generate_text(maxi = 150)


    ## Choose Theme
    m = random.random()
    mode_clair = True
    S = random.randint(0,25)/100
    H = random.randint(0,360)
    if m < 0.8 :#Mode clair
        V = 1
        V_opp = random.randint(80,100)/100
        S_opp = 1
        if H > 208 : 
            title_color = (255,255,255)
        else : 
            title_color = (0,0,0)
        text_color = hsv2rgb(random.randint(0,360),1,random.randint(50,70)/100)
    else : # Mode sombre
        mode_clair = False
        V = random.randint(20,40)/100
        V_opp = 1
        S_opp = random.randint(0,20)/100
        title_color = (0,0,0)
        text_color = (255,255,255)
    
    background_color  = hsv2rgb(H,S,V)
    bar_color = hsv2rgb(H,S_opp,V_opp)

    background = Image.new('RGB', (size,int(ratio*size)), color = background_color)
    W, H = background.size
    

    image_editable = ImageDraw.Draw(background)

    ## Create bar 
    tirage = random.random()
    is_bar = False 
    if tirage < 0.76 : 
        is_bar = True 
    

    ## Add title 
    title = generate_text(maxi = 6, length=3)
    font = ImageFont.truetype('fonts/' + font_chosen, int(W/20))
    w_title, h_title = font.getsize(title)

    if is_bar : # Add bar
        image_editable.rectangle([(0,0),(W,3*h_title/2)], fill = bar_color)

    if type == "illustration":
        title_x_pos = (W-w_title)/10
    elif type == "description":
        title_x_pos = (W - w_title)*9/10 
    else : 
        title_x_pos = (W-w_title)/2
    title_y_pos = h_title/4

    image_editable.text((title_x_pos,title_y_pos), title, title_color, font=font)
    #image_editable.rectangle([(title_x_pos,title_y_pos),(title_x_pos + w_title,title_y_pos +h_title)], outline= (0, 0, 0))
    add_xml_object('text',annotation,title_x_pos,title_y_pos,title_x_pos + w_title,title_y_pos +h_title)

    
    ## Add logo 
    x_range,y_range = h_title, h_title
    logo.thumbnail((x_range,y_range), Image.ANTIALIAS)
    w_logo,h_logo = logo.size
    if type == 'illustration':
        y_logo = int((h_logo/2)*random.random())
        x_logo = int(W - w_logo*(1.01+random.random()))
    elif type == 'description':
        y_logo = int((h_logo/2)*random.random())
        x_logo = int(w_logo*(random.random()))
    else : 
        y_logo = int(3*h_title/2 + (h_logo/2)*random.random())
        x_logo = int((W-w_logo)*random.random())
    
    background.paste(logo,(x_logo,y_logo), logo)
    #image_editable.rectangle([x_logo,y_logo,x_logo+w_logo,y_logo+h_logo], outline= (0, 0, 0))
    add_xml_object('logo',annotation,x_logo,y_logo,x_logo+w_logo,y_logo+h_logo)

    ## Add slide number 

    number = str(random.randint(1,99))
    font = ImageFont.truetype('fonts/' + font_chosen, int(W/70))
    w_num, h_num = font.getsize(number)
    pos = random.uniform(w_num/W,1-w_num/W)
    if type == "explanation":
        if random.random() > 0.5 : 
            x_pos_num = w_num*(1 + random.random())
        else : 
            x_pos_num = W - w_num*(1 + random.random())
        image_editable.text((int(x_pos_num),h_num/2), number, title_color, font=font)
        #image_editable.rectangle([(int(x_pos_num),h_num/2),(int(x_pos_num) + w_num,3*h_num/2)], outline= (0, 0, 0))
        add_xml_object('text',annotation,int(x_pos_num),h_num/2,int(x_pos_num) + w_num,3*h_num/2)
    else : 
        if is_bar : # Add bar
            image_editable.rectangle([(0,H-2*h_num),(W,H)], fill = bar_color)
        image_editable.text((int(pos*W),H-3*h_num/2), number, title_color, font=font)
        #image_editable.rectangle([(int(pos*W),H-3*h_num/2),(int(pos*W) + w_num,H-h_num/2)], outline= (0, 0, 0))
        add_xml_object('text',annotation,int(pos*W),H-3*h_num/2,int(pos*W) + w_num,H-h_num/2)

    h_bar = 3*h_title/2 # height of the color bar
    h_logo = y_logo + h_logo - h_bar # height the logo takes


    ## add body text(s)
    if type == "explanation":
        ##Text middle right
        width_text = 50
        lines = textwrap.wrap(text1,width=width_text)
        y_text = 0
        ws = []
        size_text = int(W/(1.5*width_text))
        font = ImageFont.truetype('fonts/' + font_chosen, size_text)

        x_pos = int(W/2 + W/(15 +random.random()))
        y_pos = int(h_bar + h_logo + random.random()*(H-h_bar - h_logo)/8)
        for line in lines:
            w, h = font.getsize(line)
            if y_text+y_pos + 3*h/2 < (H+h_bar+h_logo)/2 :
                ws.append(w)
                image_editable.text((x_pos,y_pos+y_text), line, text_color, font=font)
                y_text+= 3*h/2
        #image_editable.rectangle([(x_pos,y_pos),(x_pos + max(ws),y_pos+y_text-h/2)], outline= (0, 0, 0))
        add_xml_object('text',annotation,x_pos,y_pos,x_pos + max(ws),y_pos+y_text-h/2)

        ## Text bottom
        width_text = 100
        lines = textwrap.wrap(text2,width=width_text)
        y_text = 0
        ws = []
        size_text = int(W/(1.5*50))
        font = ImageFont.truetype('fonts/' + font_chosen, size_text)

        x_pos = int(W/8)
        y_pos = int((H+ h_bar + h_logo)/2 + random.random()*(H-h_bar - h_logo)/8)
        for line in lines:
            w, h = font.getsize(line)
            if y_text+y_pos + 3*h/2< H :
                
                ws.append(w)
                image_editable.text((x_pos,y_pos+y_text), line, text_color, font=font)
                y_text+= 3*h/2
        #image_editable.rectangle([(x_pos,y_pos),(x_pos + max(ws),y_pos+y_text-h/2)], outline= (0, 0, 0))
        add_xml_object('text',annotation,x_pos,y_pos,x_pos + max(ws),y_pos+y_text-h/2)
    
    elif type == "illustration":
        
        ## Text bottom
        width_text = 100
        lines = textwrap.wrap(text1,width=width_text)
        y_text = 0
        ws = []
        size_text = int(W/(1.5*50))
        font = ImageFont.truetype('fonts/' + font_chosen, size_text)

        x_pos = int(W/8)
        y_pos = int((H+ h_bar -2*h_num)/2 + random.random()*(H-h_bar - 2*h_num)/8)
        for line in lines:
            w, h = font.getsize(line)
            if y_text+y_pos+3*h/2 < (H - (2*h_num)) :
                ws.append(w)
                image_editable.text((x_pos,y_pos+y_text), line, text_color, font=font)
                y_text+= 3*h/2
        #image_editable.rectangle([(x_pos,y_pos),(x_pos + max(ws),y_pos+y_text-h/2)], outline= (0, 0, 0))
        add_xml_object('text',annotation,x_pos,y_pos,x_pos + max(ws),y_pos+y_text-h/2)

    else : 
        ##Text middle right
        width_text = 50
        lines = textwrap.wrap(text1,width=width_text)
        y_text = 0
        ws = []
        size_text = int(W/(1.5*50))
        font = ImageFont.truetype('fonts/' + font_chosen, size_text)

        x_pos = int(W/(15 +random.random()))
        y_pos = int(h_bar + random.random()*(H-h_bar)/8)
        for line in lines:
            w, h = font.getsize(line)
            if y_text+y_pos+3*h/2 < (H - 2*h_num) :
                ws.append(w)
                image_editable.text((x_pos,y_pos+y_text), line, text_color, font=font)
                y_text+= 3*h/2
        #image_editable.rectangle([(x_pos,y_pos),(x_pos + max(ws),y_pos+y_text-h/2)], outline= (0, 0, 0))
        add_xml_object('text',annotation,x_pos,y_pos,x_pos + max(ws),y_pos+y_text-h/2)
    

    ## add random image(s) 
    
    if type == "illustration":
        img1_x_range,img1_y_range = int(3*W/8), int(3*(H-h_bar-2*h_num)/8)
        image1.thumbnail((img1_x_range,img1_y_range), Image.ANTIALIAS)
        w_img1,h_img1 = image1.size
        x_img1 = int(random.random()*(W/2 -img1_x_range))
        y_img1 = int(h_bar + random.random()*((H-h_bar-2*h_num)/2 - img1_y_range))
        background.paste(image1,(x_img1,y_img1))
        #image_editable.rectangle([x_img1,y_img1,x_img1+w_img1,y_img1+h_img1], outline= (0, 0, 0))
        add_xml_object('image',annotation,x_img1,y_img1,x_img1+w_img1,y_img1+h_img1)

        img2_x_range,img2_y_range = int(3*W/8), int(3*(H-h_bar-2*h_num)/8)
        image2.thumbnail((img2_x_range,img2_y_range), Image.ANTIALIAS)
        w_img2,h_img2 = image2.size
        x_img2 =  int(W/2 + random.random()*(W/2 -img2_x_range))
        y_img2 = int(h_bar + random.random()*((H-h_bar-2*h_num)/2 - img2_y_range))
        background.paste(image2,(x_img2,y_img2))
        #image_editable.rectangle([x_img2,y_img2,x_img2+w_img2,y_img2+h_img2], outline= (0, 0, 0))
        add_xml_object('image',annotation,x_img2,y_img2,x_img2+w_img2,y_img2+h_img2)
    elif type == "description":
        img1_x_range,img1_y_range = int(3*W/8), int(3*(H-h_bar-2*h_num)/4)
        image1.thumbnail((img1_x_range,img1_y_range), Image.ANTIALIAS)
        w_img1,h_img1 = image1.size
        x_img1 = int(W/2 + random.random()*(W/2-img1_x_range)) 
        y_img1 = int(h_bar+ (H - h_bar - 2*h_num - img1_y_range)*random.random())
        background.paste(image1,(x_img1,y_img1))
        #image_editable.rectangle([x_img1,y_img1,x_img1+w_img1,y_img1+h_img1], outline= (0, 0, 0))
        add_xml_object('image',annotation,x_img1,y_img1,x_img1+w_img1,y_img1+h_img1)
    else : 
        img1_x_range,img1_y_range = int(W/(4-random.random())), int((H-h_bar-h_logo)/(2+(1/2)*random.random()))
        image1.thumbnail((img1_x_range,img1_y_range), Image.ANTIALIAS)
        w_img1,h_img1 = image1.size
        x_img1 = int(random.random()*(W/2 -img1_x_range))
        y_img1 = int(h_bar+ h_logo +((H - h_bar - h_logo)/2 - img1_y_range)*random.random())
        background.paste(image1,(x_img1,y_img1))
        #image_editable.rectangle([x_img1,y_img1,x_img1+w_img1,y_img1+h_img1], outline= (0, 0, 0))
        add_xml_object('image',annotation,x_img1,y_img1,x_img1+w_img1,y_img1+h_img1)
    

    background.save("images_generated/"+name+".jpg") ## Save Image
    xml_file = codecs.open("images_generated/"+name +'.xml',"w",'utf-8')
    xml_file.write(prettify(annotation))

def text_wrap(text,font,writing,max_width,max_height):
    lines = [[]]
    words = text.split()
    for word in words:
        # try putting this word in last line then measure
        lines[-1].append(word)
        (w,h) = writing.multiline_textsize('\n'.join([' '.join(line) for line in lines]), font=font)
        if w > max_width: # too wide
            # take it back out, put it on the next line, then measure again
            lines.append([lines[-1].pop()])
            (w,h) = writing.multiline_textsize('\n'.join([' '.join(line) for line in lines]), font=font)
            if h > max_height: # too high now, cannot fit this word in, so take out - add ellipses
                lines.pop()
                # try adding ellipses to last word fitting (i.e. without a space)
                #lines[-1][-1] += '...'
                # keep checking that this doesn't make the textbox too wide, 
                # if so, cycle through previous words until the ellipses can fit
                while writing.multiline_textsize('\n'.join([' '.join(line) for line in lines]),font=font)[0] > max_width:
                    if len(lines[-1])>0:
                        lines[-1].pop()
                    else :
                        lines.pop()
                    #lines[-1][-1] += '...'
                break
    return '\n'.join([' '.join(line) for line in lines])

def create_text_images(name, features):
    font_id = random.randint(0,len(fonts)-1)
    font_chosen = fonts[font_id]
    ## Choose Theme
    m = random.random()
    mode_clair = True
    S = random.randint(0,25)/100
    H = random.randint(0,360)
    if m < 0.8 :#Mode clair
        V = 1
        text_color = hsv2rgb(random.randint(0,360),1,random.randint(50,70)/100)
    else : # Mode sombre
        mode_clair = False
        V = random.randint(20,40)/100
        text_color = (255,255,255)
    
    background_color  = hsv2rgb(H,S,V)

    width_min = 200
    width_max = 1500
    height_min = 75
    height_max = 800


    background = Image.new('RGB', (random.randint(width_min,width_max),random.randint(height_min,height_max)), color = background_color)
    image_editable = ImageDraw.Draw(background)
    W, H = background.size
    text = generate_text(length = 500, maxi=700)
    #size_text = int(W/random.randint(1.5*10,1.5*70))
    size_text = random.randint(40,60)
    font = ImageFont.truetype('fonts/' + font_chosen, size_text)
    
    text_wrapped = text_wrap(text,font,image_editable,W,H)
    """
    y_text = 0
    w_num, h_num = font.getsize(text)
    x_pos = 0
    y_pos = 0
    print(W)
    lines = textwrap.wrap(text,width=int(W/size_text))
    print(lines)
    ws = []
    for line in lines:
        w, h = font.getsize(line)
        if y_text+y_pos+3*h/2 < H :
            ws.append(w)
            image_editable.text((x_pos,y_pos+y_text), line, text_color, font=font)
            y_text+= 3*h/2
    """
    image_editable.text((0,0), text_wrapped, text_color, font=font)
    
    features = features.append(pd.DataFrame({"name": [name], "font": [font_chosen], 'size': [size_text], "color": [text_color]}), ignore_index = True)
    

    background.save("text_images_generated/"+name+".jpg")
    return features


def generate_slides(n=5000):
    for k in range(5000):
        type_int = random.randint(0,2)
        type_ill = slide_types[type_int]
        create_slide(type= type_ill, name = 'slide_'+str(k+6126))

def generate_text_images(n = 100, start = 0):
    features = pd.read_csv("text_images_generated/features.csv")[["name","font","size","color"]]
    for k in range(n):
        features = create_text_images("img_"+str(k+start),features)
    features.to_csv("text_images_generated/features.csv")

if __name__ == "__main__":
    #Generate 5000 images
    #generate_slides(5000)
    #font_dt = pd.DataFrame({"fonts" : fonts})
    #font_dt.to_csv("text_images_generated/fonts.csv")
    generate_text_images(n = 500, start=1102)
    
    #Generate images of text for font recognition 