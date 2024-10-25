from PIL import Image
image = Image.open('f1CarSide.jpg')
mask=image.convert("L")
th=150 # the value has to be adjusted for an image of interest 
mask = mask.point(lambda i: i < th and 255)
mask.save('3000 (1).jpg')