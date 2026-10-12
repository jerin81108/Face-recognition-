import cv2
import json
from core.pipeline import RecognitionPipeline

def main():
    pipeline = RecognitionPipeline()
    print("Enrolled roster:")
    for p in pipeline.database.get_persons():
        print(f" - {p['name']}: {p['samples_count']} sample(s)")

    print("\n--- Testing on meeting_test.jpg ---")
    img = cv2.imread("data/test_images/meeting_test.jpg")
    if img is None:
        print("Error: Could not load test image!")
        return

    result = pipeline.process_image(img)
    print(f"Detected {result['count']} face(s) in {result['latency_ms']} ms:")
    for i, face in enumerate(result["faces"]):
        print(f" Face #{i+1}:")
        print(f"   Name:       {face['name']}")
        print(f"   Is Known:   {face['is_known']}")
        print(f"   Similarity: {face['similarity']}")
        print(f"   Confidence: {face['confidence']}%")
        print(f"   BBox:       {face['bbox']}")

    cv2.imwrite("data/test_images/meeting_annotated.jpg", result["annotated_image"])
    print("\nSaved annotated result to data/test_images/meeting_annotated.jpg")

    print("\n--- Testing on alex_portrait.jpg ---")
    alex_img = cv2.imread("data/test_images/alex_portrait.jpg")
    res_alex = pipeline.process_image(alex_img)
    for i, face in enumerate(res_alex["faces"]):
        print(f" Face #{i+1}: Name={face['name']}, Sim={face['similarity']}, Conf={face['confidence']}%")

if __name__ == "__main__":
    main()
