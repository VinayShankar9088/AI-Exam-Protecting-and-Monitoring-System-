import cv2

REFERENCE_IMG = "known_faces/student1.jpg"

def verify_face(frame):
    """Verify a face in `frame` against `REFERENCE_IMG`.

    This function imports DeepFace lazily so failures to import the library
    (e.g. incompatible TensorFlow/Keras) won't crash the whole app at module
    import time. If DeepFace cannot be imported or raises an error, the
    function returns `(False, "DeepFace Error")` and prints a helpful message.
    """
    cv2.imwrite("temp_frame.jpg", frame)

    try:
        # Import DeepFace only when the function is called. This avoids raising
        # an ImportError during application startup if DeepFace or TF is broken.
        from deepface import DeepFace
    except Exception as e:
        print("DeepFace import failed:", e)
        return False, "DeepFace Error"

    try:
        result = DeepFace.verify(
            img1_path="temp_frame.jpg",
            img2_path=REFERENCE_IMG,
            detector_backend="opencv",
            enforce_detection=False
        )
        if result.get("verified"):
            return True, "Verified ✅"
        else:
            return False, "Unverified ❌"
    except Exception as e:
        print("DeepFace error during verification:", e)
        return False, "DeepFace Error"
