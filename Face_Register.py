import dlib
import cv2
import tkinter as tk
from PIL import Image, ImageTk
import pymongo
from bson.binary import Binary
import io
import os

detector = dlib.get_frontal_face_detector()
client = pymongo.MongoClient("mongodb://localhost:27017/")  # Connect to MongoDB
db = client["FaceImages"]  # Use or create a database called 'FaceImages'
collection = db["Images"]  # Use or create a collection called 'Images'

class FaceRegister:
    def __init__(self):
        self.input_name = ""
        self.path_photos = "face_data/"  # Directory to save face images
        
        if not os.path.exists(self.path_photos):
            os.makedirs(self.path_photos)
        
        # Initialize GUI, camera, and other variables
        self.win = tk.Tk()
        self.win.title("Face Register")
        self.win.geometry("800x600")

        self.label = tk.Label(self.win)
        self.label.pack()

        self.frame_right_info = tk.Frame(self.win)
        self.input_name_label = tk.Label(self.frame_right_info, text="Enter Name: ")
        self.input_name_label.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.input_name_entry = tk.Entry(self.frame_right_info)
        self.input_name_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        self.input_name_button = tk.Button(self.frame_right_info, text="Capture", command=self.capture)
        self.input_name_button.grid(row=0, column=2, padx=5, pady=5)
        self.frame_right_info.pack()

        self.cap = cv2.VideoCapture(0)  # Get video stream from camera
        self.process()
        self.win.mainloop()

    def process(self):
        ret, frame = self.cap.read()

        if ret:
            frame = cv2.flip(frame, 1)
            faces = detector(frame, 0)
            for face in faces:
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.label.imgtk = imgtk
            self.label.configure(image=imgtk)
            self.label.after(10, self.process)

    def capture(self):
        self.input_name = self.input_name_entry.get()
        if self.input_name:
            ret, frame = self.cap.read()
            if ret:
                # Save image to directory
                filename = f"face_data/{self.input_name}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Image captured and saved as {filename}")

                # Save image to MongoDB
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                byte_io = io.BytesIO()
                image.save(byte_io, format='JPEG')
                image_data = byte_io.getvalue()
                image_binary = Binary(image_data)

                data = {
                    'name': self.input_name,
                    'image': image_binary
                }
                collection.insert_one(data)
                print(f"Image saved to MongoDB for {self.input_name}")

                # Display the captured image in the GUI
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                imgtk = ImageTk.PhotoImage(image=img)
                self.label.imgtk = imgtk
                self.label.configure(image=imgtk)

def main():
    FaceRegister()

if __name__ == '__main__':
    main()
