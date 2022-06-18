import PIL.Image
import PIL.ImageDraw
import face_recognition


given_image = face_recognition.load_image_file('kls.jpg')

face_locations = face_recognition.face_locations(given_image)

number_of_faces = len(face_locations)
print("We found {} face(s) in this image.".format(number_of_faces))

pil_image = PIL.Image.fromarray(given_image)


for face_location in face_locations:
    top, left, bottom, right = face_location
    print("A face is detected at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

    draw = PIL.ImageDraw.Draw(pil_image)
draw.rectangle([left, top, right, bottom], outline="blue", width=10)

pil_image.show()