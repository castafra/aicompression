from tarfile import ENCODING
from xml.etree.ElementTree import Element, SubElement, Comment, tostring
from xml.etree import ElementTree
from xml.dom import minidom
import codecs

def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ElementTree.tostring(elem, encoding='unicode')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

size = 1920
name = 'test1'

annotation = Element('annotation')
folder = SubElement(annotation,'folder')
folder.text = 'Slide Generator'
filename = SubElement(annotation,'filename')
filename.text = name + '.jpg'
path = SubElement(annotation,'path')
path.text = "C:\\Users\\François\\Documents\\GitHub\\aicompression\\Slide Generator\\images_generated\\" + name + ".jpg"
source = SubElement(annotation,'source')
database = SubElement(source,"database")
database.text = "Unknown"
size_el = SubElement(annotation,"size")
width_el = SubElement(size_el,"width")
width_el.text = str(size)
height_el = SubElement(size_el,"height")
height_el.text = str(2)
depth_el = SubElement(size_el,"width")
depth_el.text = str(size)

xml_file = codecs.open(name +'.xml',"w",'utf-8')
xml_file.write(prettify(annotation))
print(prettify(annotation))