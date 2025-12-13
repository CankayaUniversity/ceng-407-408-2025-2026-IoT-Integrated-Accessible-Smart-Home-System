import cv2
import mediapipe as mp

# --- Setup ---
# Initialize MediaPipe modules
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

url = "http://192.x.x.x:4747/video"

# Open the camera stream (IP camera / DroidCam, etc.)
cap = cv2.VideoCapture(url)

print("Starting eye/face tracking... Press 'q' to quit.")

# FaceMesh configuration:
# refine_landmarks=True enables more accurate iris (pupil) tracking.
with mp_face_mesh.FaceMesh(
    max_num_faces=2,            # how many faces to track at once
    refine_landmarks=True,      # better iris detail
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Could not read a frame from the camera stream.")
            continue

        # Small speed-up: mark image as read-only before processing
        image.flags.writeable = False

        # OpenCV uses BGR, MediaPipe expects RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Run FaceMesh inference
        results = face_mesh.process(image)

        # Convert back to BGR for drawing + displaying
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # If we detected at least one face, draw overlays
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:

                # --- Draw face contours (cleaner than drawing the full mesh) ---
                # FACEMESH_CONTOURS draws eye outlines, lips, and the face oval.
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_contours_style()
                )

                # --- Draw the irises (pupils) ---
                # Works because refine_landmarks=True is enabled above.
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_iris_connections_style()
                )

        # Optional: mirror the view to feel like a front camera preview
        # image = cv2.flip(image, 1)

        cv2.imshow("MediaPipe Eye Tracking (PoC)", image)

        # Quit on 'q'
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
