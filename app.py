from flask import Flask, render_template, Response, jsonify, request
import cv2
from utils.face_verification import verify_face
from utils.object_detection import detect_objects
import time

app = Flask(__name__)
camera = None  # Will open when exam starts
exam_running = False

def generate_frames():
    global camera, exam_running
    # Stream frames only while an exam session is running
    while exam_running:
        if camera is None:
            camera = cv2.VideoCapture(0)  # Open webcam when exam starts
            # Small delay to let camera warm up
            time.sleep(0.1)

        success, frame = camera.read()
        if not success:
            break

        # Mirror the frame horizontally so the displayed video is like a mirror
        try:
            frame = cv2.flip(frame, 1)
        except Exception:
            # If flipping fails for any reason, continue with the original frame
            pass

        # Face Verification
        verified, status = verify_face(frame)

        # Object Detection
        frame = detect_objects(frame)

        # Status Text
        cv2.putText(frame, status, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_exam', methods=['POST'])
def start_exam():
    global exam_running
    exam_running = True
    return jsonify({'status': 'Exam Started'})

@app.route('/stop_exam', methods=['POST'])
def stop_exam():
    global exam_running, camera
    exam_running = False
    # Release camera if it was opened
    if camera is not None:
        try:
            camera.release()
        except Exception:
            pass
        camera = None
    return jsonify({'status': 'Stopped'})

if __name__ == "__main__":
    app.run(debug=True)
