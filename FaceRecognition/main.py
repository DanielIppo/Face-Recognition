import cv2 as cv
import face_recognition


personName = "Person"
imgPath = "FaceRecognition\Faces\Person.jpg"
personImage = face_recognition.load_image_file(imgPath)
personFaceEncoding = face_recognition.face_encodings(personImage)[0]

imageIrl = cv.VideoCapture(0)

if not imageIrl.isOpened():
    print("impossibile aprire la webcam")
    exit()

def draw_rectangle(image, rect):
    (x, y, w, h) = rect
    cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

def draw_text(image, text, x, y):
    cv.putText(image, text, (x, y), cv.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

def detect(image):
    global recognition
    faceRects = face_recognition.face_locations(image)
    if len(faceRects) > 0:
        faceEncodings = face_recognition.face_encodings(image, faceRects)
        for encoding in faceEncodings:
            matches = face_recognition.compare_faces([personFaceEncoding], encoding)
            if matches[0]:
                recognition = True
                draw_text(image, personName, faceRects[0][3], faceRects[0][0])
            else:
                recognition = False
                draw_text(image, "Unknown", faceRects[0][3], faceRects[0][0])
    return image

while True:
    ret, frame = imageIrl.read()
    if ret:
        frame = detect(frame)
        cv.imshow("Face Recognition", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("impossibile leggere il frame")
        break