import cv2
import json

def keypoints_to_dict(keypoints):
    """
    cv2.KeyPoint 객체 리스트를 JSON 직렬화가 가능한 딕셔너리 리스트로 변환합니다.
    각 딕셔너리는 keypoint의 속성을 저장합니다.
    """
    data = []
    for kp in keypoints:
        kp_data = {
            "pt": kp.pt,
            "size": kp.size,
            "angle": kp.angle,
            "response": kp.response,
            "octave": kp.octave,
            "class_id": kp.class_id
        }
        data.append(kp_data)
    return data

def dict_to_keypoints(data):
    """
    딕셔너리 리스트를 cv2.KeyPoint 객체 리스트로 복원합니다.
    각 딕셔너리는 {"pt": (x, y), "size": size, "angle": angle, "response": response, "octave": octave, "class_id": class_id} 형태여야 합니다.
    """
    keypoints = []
    for kp_data in data:
        kp = cv2.KeyPoint(
            kp_data["pt"][0],
            kp_data["pt"][1],
            kp_data["size"],
            kp_data["angle"],
            kp_data["response"],
            kp_data["octave"],
            kp_data["class_id"]
        )
        keypoints.append(kp)
    return keypoints

def dmatches_to_dict(dmatches):
    """
    cv2.DMatch 객체 리스트를 JSON 직렬화가 가능한 딕셔너리 리스트로 변환합니다.
    각 딕셔너리는 DMatch의 queryIdx, trainIdx, imgIdx, distance 속성을 포함합니다.
    """
    dmatch_list = []
    for d in dmatches:
        dmatch_list.append({
            "queryIdx": d.queryIdx,
            "trainIdx": d.trainIdx,
            "imgIdx": d.imgIdx,
            "distance": d.distance
        })
    return dmatch_list

def dict_to_dmatches(data):
    """
    딕셔너리 리스트를 cv2.DMatch 객체 리스트로 복원합니다.
    """
    dmatches = []
    for d in data:
        dmatch = cv2.DMatch(_queryIdx=d["queryIdx"],
                            _trainIdx=d["trainIdx"],
                            _imgIdx=d["imgIdx"],
                            _distance=d["distance"])
        dmatches.append(dmatch)
    return dmatches