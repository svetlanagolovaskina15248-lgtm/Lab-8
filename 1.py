from PIL import Image

image = Image.open("variant-1.jpg")

grayscale_image = image.convert("L")

grayscale_image.save("polyton.png")
