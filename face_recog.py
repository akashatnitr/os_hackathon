import cv2
import face_recognition
import numpy as np
from livekit import agents, rtc
import os
# Your LiveKit connection details
LIVEKIT_URL = os.environ['LIVEKIT_URL']#'ws://your-livekit-server-url'
API_KEY = os.environ['LIVEKIT_API_KEY']
SECRET_KEY = os.environ['LIVEKIT_API_SECRET']

# Initialize LiveKit agent for RTC connection
agent = agents.Agent(LIVEKIT_URL, API_KEY, SECRET_KEY)

# Function to handle incoming video tracks and apply face recognition
def process_video_track(track: rtc.VideoTrack):
    # Load known face image and encode it
    known_face_image = face_recognition.load_image_file("known_person.jpg")
    known_face_encoding = face_recognition.face_encodings(known_face_image)[0]

    known_face_encodings = [known_face_encoding]
    known_face_names = ["Known Person"]

    # Callback function to process each frame
    def on_frame(frame):
        # Convert RTC video frame to OpenCV format
        img = frame.to_ndarray()  # Assuming LiveKit provides ndarray format
        
        # Resize frame for faster face recognition processing
        small_frame = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR (OpenCV) to RGB (face_recognition)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Detect all faces in the current frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        # Loop through each detected face
        face_names = []
        for face_encoding in face_encodings:
            # Check if the detected face matches any known face encoding
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

        # Draw bounding boxes and labels on the frame
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(img, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting frame
        cv2.imshow('Video with Face Recognition', img)

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False

        return True

    # Set the callback to process frames from the video track
    track.on_frame(on_frame)

# Join a LiveKit room and start processing video streams
def start_livekit_session():
    # Create and join a LiveKit room
    room = agent.join_room('your-room-id')

    # Get participants' video tracks and start processing them
    for participant in room.participants.values():
        for track in participant.video_tracks:
            process_video_track(track)

# Run the LiveKit session and face recognition
if __name__ == "__main__":
    start_livekit_session()

    # Cleanup when done
    cv2.destroyAllWindows()
