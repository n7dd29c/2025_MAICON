# -*- coding: utf-8 -*-
"""
ArUco 마커 ID 감지 및 파일명 매칭 스크립트
이미지 폴더 내의 모든 마커 사진에서 ID를 감지하여 파일명과 매칭
"""
import cv2
import os
import glob
import sys
from pathlib import Path


def detect_aruco_id(image_path, dictionary_id=8, use_preprocessing=True):
    """
    이미지에서 ArUco 마커 ID 감지 (개선된 전처리 및 파라미터)
    
    Args:
        image_path: 이미지 파일 경로
        dictionary_id: ArUco 딕셔너리 ID (기본: 8 = DICT_6X6_50)
        use_preprocessing: 전처리 사용 여부
    
    Returns:
        detected_ids: 감지된 마커 ID 리스트 (없으면 None)
    """
    # 이미지 읽기
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # 그레이스케일 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # ArUco 딕셔너리 초기화
    try:
        # OpenCV 4.7+ 버전
        aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_id)
        aruco_params = cv2.aruco.DetectorParameters()
    except AttributeError:
        # OpenCV 4.6 이하 버전
        aruco_dict = cv2.aruco.Dictionary_get(dictionary_id)
        aruco_params = cv2.aruco.DetectorParameters_create()
    
    # 매우 관대한 파라미터 설정 (test.py에서 성공한 설정 + 회전된 마커 감지 개선)
    aruco_params.minMarkerPerimeterRate = 0.01
    aruco_params.maxMarkerPerimeterRate = 20.0
    aruco_params.adaptiveThreshConstant = 3
    aruco_params.errorCorrectionRate = 1.0
    # 회전된 마커 감지를 위한 추가 파라미터
    aruco_params.polygonalApproxAccuracyRate = 0.05  # 다각형 근사 정확도 (더 관대하게)
    aruco_params.minCornerDistanceRate = 0.01  # 코너 간 최소 거리
    aruco_params.minDistanceToBorder = 1  # 경계로부터 최소 거리
    
    # 전처리 방법들 (test.py에서 성공한 순서대로)
    if use_preprocessing:
        preprocess_methods = [
            ("대비 향상", cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)),
            ("원본", gray),
            ("이진화", cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)),
            ("역이진화", cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)),
        ]
    else:
        preprocess_methods = [("원본", gray)]
    
    # 모든 감지 결과 저장 (면적과 ID)
    all_detections = []  # [(area, marker_id), ...]
    
    # 여러 전처리 방법으로 시도
    for method_name, processed_img in preprocess_methods:
        corners, ids, rejected = cv2.aruco.detectMarkers(
            processed_img, aruco_dict, parameters=aruco_params
        )
        
        if ids is not None and len(ids) > 0:
            # 각 마커의 면적 계산
            for i, marker_id in enumerate(ids.flatten()):
                marker_corners = corners[i][0]
                area = cv2.contourArea(marker_corners)
                all_detections.append((area, int(marker_id)))
    
    # 회전된 마커 감지: 이미지를 회전시켜서도 시도
    h, w = gray.shape
    center = (w // 2, h // 2)
    rotation_angles = [90, 180, 270]  # 90도, 180도, 270도 회전
    
    for angle in rotation_angles:
        # 회전 변환 행렬 생성
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(gray, M, (w, h))
        
        # 회전된 이미지로 감지 시도
        for method_name, processed_img in preprocess_methods:
            # 회전된 이미지에 전처리 적용
            if method_name == "대비 향상":
                processed = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(rotated)
            elif method_name == "이진화":
                processed = cv2.adaptiveThreshold(rotated, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            elif method_name == "역이진화":
                processed = cv2.adaptiveThreshold(rotated, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            else:
                processed = rotated
            
            corners, ids, rejected = cv2.aruco.detectMarkers(
                processed, aruco_dict, parameters=aruco_params
            )
            
            if ids is not None and len(ids) > 0:
                # 각 마커의 면적 계산
                for i, marker_id in enumerate(ids.flatten()):
                    marker_corners = corners[i][0]
                    area = cv2.contourArea(marker_corners)
                    all_detections.append((area, int(marker_id)))
    
    # 감지 결과가 있으면 처리
    if all_detections:
        # 면적 기준으로 정렬 (큰 것부터)
        all_detections.sort(reverse=True, key=lambda x: x[0])
        
        # 가장 큰 마커의 ID만 반환 (중복 제거)
        # 같은 ID가 여러 번 감지되면 가장 큰 면적의 것만 사용
        seen_ids = set()
        result_ids = []
        
        for area, marker_id in all_detections:
            if marker_id not in seen_ids:
                seen_ids.add(marker_id)
                result_ids.append(marker_id)
        
        # 가장 큰 마커 하나만 반환 (오탐 방지)
        if result_ids:
            return [result_ids[0]]
    
    return None


def try_multiple_dictionaries(image_path):
    """
    여러 딕셔너리로 시도하여 마커 ID 감지 (전처리 포함)
    
    Args:
        image_path: 이미지 파일 경로
    
    Returns:
        (dictionary_name, detected_ids): 딕셔너리 이름과 감지된 ID
    """
    dictionaries = [
        (8, "DICT_6X6_50"),      # test.py에서 성공한 딕셔너리 우선
        (10, "DICT_4X4_50"),
        (11, "DICT_4X4_100"),
        (12, "DICT_4X4_250"),
        (13, "DICT_4X4_1000"),
        (7, "DICT_5X5_50"),
    ]
    
    for dict_id, dict_name in dictionaries:
        ids = detect_aruco_id(image_path, dict_id, use_preprocessing=True)
        if ids:
            return dict_name, ids
    
    return None, None


def scan_markers_in_folder(folder_path, dictionary_id=8, try_other_dicts=False):
    """
    폴더 내 모든 이미지에서 마커 ID 감지
    
    Args:
        folder_path: 이미지 폴더 경로
        dictionary_id: ArUco 딕셔너리 ID
        try_other_dicts: 다른 딕셔너리도 시도할지 여부
    
    Returns:
        results: [(파일명, [ID들]), ...] 리스트
    """
    # 지원하는 이미지 확장자
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG']
    
    # 모든 이미지 파일 찾기 (중복 제거)
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
    
    # 중복 제거 (대소문자 구분 없이)
    seen_files = set()
    unique_files = []
    for img_file in image_files:
        img_file_lower = img_file.lower()
        if img_file_lower not in seen_files:
            seen_files.add(img_file_lower)
            unique_files.append(img_file)
    
    image_files = sorted(unique_files)  # 정렬
    
    if not image_files:
        print(f"폴더에 이미지 파일이 없습니다: {folder_path}")
        return []
    
    print(f"총 {len(image_files)}개의 이미지 파일을 찾았습니다.\n")
    print("=" * 60)
    print(f"{'파일명':<30} {'감지된 ID':<20}")
    print("=" * 60)
    
    results = []
    
    for image_path in sorted(image_files):
        filename = os.path.basename(image_path)
        detected_ids = detect_aruco_id(image_path, dictionary_id, use_preprocessing=True)
        
        # 감지되지 않았고 다른 딕셔너리 시도 옵션이 켜져있으면 시도
        if detected_ids is None and try_other_dicts:
            dict_name, detected_ids = try_multiple_dictionaries(image_path)
            if detected_ids:
                print(f"{filename:<30} ID={', '.join(map(str, detected_ids))} ({dict_name})")
            else:
                print(f"{filename:<30} 마커 미감지")
        elif detected_ids:
            ids_str = ', '.join(map(str, detected_ids))
            print(f"{filename:<30} ID={ids_str}")
        else:
            print(f"{filename:<30} 마커 미감지")
        
        results.append((filename, detected_ids))
    
    print("=" * 60)
    
    # 결과를 파일로 저장 (중복 제거)
    output_file = os.path.join(folder_path, "marker_ids.txt")
    seen_filenames = set()
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("파일명\t감지된 ID\n")
        for filename, ids in results:
            # 파일명 중복 제거
            if filename not in seen_filenames:
                seen_filenames.add(filename)
                if ids:
                    # ID 리스트에서 중복 제거
                    unique_ids = list(dict.fromkeys(ids))  # 순서 유지하면서 중복 제거
                    f.write(f"{filename}\t{', '.join(map(str, unique_ids))}\n")
                else:
                    f.write(f"{filename}\t미감지\n")
    
    print(f"\n결과가 저장되었습니다: {output_file}")
    
    return results


def main():
    """메인 함수"""
    # 사용법 출력
    if len(sys.argv) < 2:
        print("=" * 60)
        print("ArUco 마커 ID 감지 스크립트")
        print("=" * 60)
        print("\n사용법:")
        print("  python detect_marker_ids.py <이미지_폴더_경로> [옵션]")
        print("\n옵션:")
        print("  --dict <ID>     딕셔너리 ID 지정 (기본: 8)")
        print("  --try-all       모든 딕셔너리 시도")
        print("\n예시:")
        print("  python detect_marker_ids.py ./markers")
        print("  python detect_marker_ids.py ./markers --dict 8")
        print("  python detect_marker_ids.py ./markers --try-all")
        print("\n딕셔너리 ID:")
        print("  8  = DICT_6X6_50 (기본값)")
        print("  10 = DICT_4X4_50")
        print("  11 = DICT_4X4_100")
        print("  12 = DICT_4X4_250")
        print("  13 = DICT_4X4_1000")
        print("  7  = DICT_5X5_50")
        print("=" * 60)
        sys.exit(1)
    
    folder_path = sys.argv[1]
    dictionary_id = 8  # 기본값 (DICT_6X6_50)
    try_other_dicts = False
    
    # 옵션 파싱
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '--dict' and i + 1 < len(sys.argv):
            dictionary_id = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--try-all':
            try_other_dicts = True
            i += 1
        else:
            print(f"알 수 없는 옵션: {sys.argv[i]}")
            sys.exit(1)
    
    if not os.path.exists(folder_path):
        print(f"오류: 폴더를 찾을 수 없습니다: {folder_path}")
        sys.exit(1)
    
    if not os.path.isdir(folder_path):
        print(f"오류: 경로가 폴더가 아닙니다: {folder_path}")
        sys.exit(1)
    
    # 마커 ID 감지
    results = scan_markers_in_folder(folder_path, dictionary_id, try_other_dicts)
    
    # 통계 출력
    detected_count = sum(1 for _, ids in results if ids is not None)
    undetected_count = len(results) - detected_count
    
    print(f"\n통계:")
    print(f"  총 파일 수: {len(results)}")
    print(f"  감지 성공: {detected_count}")
    print(f"  감지 실패: {undetected_count}")
    
    if undetected_count > 0 and not try_other_dicts:
        print(f"\n팁: 감지되지 않은 파일이 있습니다. --try-all 옵션으로 다른 딕셔너리를 시도해보세요.")


if __name__ == "__main__":
    main()

