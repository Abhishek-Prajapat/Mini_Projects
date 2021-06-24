import pdf2image as pi
import pytesseract as pt
import numpy as np
import cv2
import argparse

arg = argparse.ArgumentParser()
arg.add_argument('-pp', '--pdf_path',  required=True)
arg.add_argument('-conf', '--confidence',  required=True)
args = vars(arg.parse_args())

poppler_path='C:\\Program Files\\poppler-21.03.0\\Library\\bin'
pt.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract'

def get_text(image):
    image = np.array(image)
    output = pt.image_to_data(image, output_type='dict')
    text = []
    for i in range(0, len(output["text"])):

        x = output["left"][i]
        y = output["top"][i]
        w = output["width"][i]
        h = output["height"][i]

        if float(output['conf'][i]) > int(args['confidence']):
            text.append(output['text'][i])
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
    text = ' '.join(text)
        
    return text, image


def get_pdf(pdf_path):
    images = pi.convert_from_path(pdf_path, poppler_path=poppler_path)
    texts = []

    for idx, image in enumerate(images):
        text, image = get_text(image)
        texts.append(text)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f'./{idx}.jpg', image)

    return texts

if __name__ == '__main__':
    
    texts = get_pdf(str(args['pdf_path'])) 

    with open('./text.txt', 'w') as file:
        for page in texts:
            file.write(page + '\n\n')