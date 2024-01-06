import cv2
import face_recognition
import pandas as pd
from datetime import datetime

# Function to recognize faces from the webcam and mark attendance
def recognize_faces_webcam():
    known_faces = {
        "Mohit": face_recognition.face_encodings(face_recognition.load_image_file(r"C:\Users\Mohit\Pictures\Camera Roll\WIN_20240104_17_49_28_Pro.jpg"))[0]
    }

    video_capture = cv2.VideoCapture(0)

    # Create a DataFrame to store attendance
    attendance_df = pd.DataFrame(columns=["Name"])

    while True:
        # Capture each frame from the webcam
        ret, frame = video_capture.read()

        # Check if the frame was read successfully
        if not ret:
            print("Error reading frame")
            break

        # Convert the frame to RGB format (required by face_recognition)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find all face locations and face encodings in the frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(list(known_faces.values()), face_encoding)

            name = "Unknown"
            date = datetime.now().strftime("%Y-%m-%d")
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if True in matches:
                first_match_index = matches.index(True)
                name = list(known_faces.keys())[first_match_index]

                # Check if the person has already been marked present for the day
                if name in attendance_df["Name"].values:
                    # Check if the date column already exists, if not, create it
                    if date not in attendance_df.columns:
                        attendance_df[date] = "No"  # Initialize with "No"
                    
                    # Update the attendance for the current date to "Yes"
                    row_index = attendance_df.index[attendance_df["Name"] == name][0]
                    attendance_df.loc[row_index, date] = "Yes"
                    
                    print(f"{name} marked present at {timestamp}")
                else:
                    # Add to the DataFrame
                    attendance_df = pd.concat([attendance_df, pd.DataFrame({"Name": [name], date: ["Yes"]})], ignore_index=True)
                    print(f"{name} marked present at {timestamp}")

            # Draw a rectangle around the face and display the name
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        cv2.imshow('Video', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

    # Save attendance to a CSV file
    attendance_df.to_csv("attendance.csv", index=False)
    print("Attendance saved to 'attendance.csv'")

# Call the function to start facial recognition and mark attendance from the webcam
recognize_faces_webcam()
