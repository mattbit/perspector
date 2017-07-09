from perspector import Perspector, preview

####################################
# Demo                             #
####################################

IMAGE_PATH = "demo.jpg"

p = Perspector(IMAGE_PATH)
preview(p.outline())
preview(p.transform())
# p.write('demo_result.jpg')


####################################
# Process multiple images          #
####################################
'''
for image in glob.glob("input/*.jpg"):
    output = image.replace("input", "output")
    p = Perspector(image)

    try:
        p.write(output)
    except Exception:
        logging.error("cannot process {}".format(image))
        p.write_original(output)
'''
