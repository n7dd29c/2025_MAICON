# -*- coding: utf-8 -*-
"""
대시보드 통신 모듈
드론/로봇에서 대시보드로 데이터 전송
"""
import json
import requests
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


class DashboardComm:
    """대시보드 통신 클래스"""
    
    def __init__(self, 
                 server_url: str = "http://58.229.150.23:5000",
                 mission_code: str = "2W9G"):
        """
        Args:
            server_url: 대시보드 서버 URL
            mission_code: 임무 코드
        """
        self.server_url = server_url
        self.mission_code = mission_code
        self.json_filename = f"{mission_code}.json"
        
        # 현재 데이터 상태
        self.current_data = {
            "mission_code": mission_code,
            "fire_buildings": [],
            "points": [],
            "detection": {
                "Alpha": [],
                "Bravo": [],
                "Charlie": []
            }
        }
        
        # ai_outbox 폴더 생성
        self.outbox_dir = Path("ai_outbox")
        self.outbox_dir.mkdir(exist_ok=True)
    
    def send_dashboard_json(self, data: Optional[Dict] = None) -> bool:
        """
        대시보드로 JSON 데이터 전송
        
        Args:
            data: 전송할 데이터 (None이면 현재 데이터 사용)
            
        Returns:
            성공 여부
        """
        if data is None:
            data = self.current_data
        
        try:
            json_content = json.dumps(data, indent=2, ensure_ascii=False)
            
            files = {
                'file': (self.json_filename, json_content, 'application/json')
            }
            
            print(f"대시보드로 전송 중: {self.json_filename} ...")
            
            response = requests.post(
                f'{self.server_url}/dashboard_json',
                files=files,
                timeout=10
            )
            
            if response.status_code == 200:
                print(f"대시보드 전송 성공: {response.text}")
                # 로컬에도 저장
                json_path = self.outbox_dir / self.json_filename
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                return True
            else:
                print(f"대시보드 전송 실패: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"대시보드 전송 오류: {e}")
            return False
    
    def send_dashboard_image(self, image_path: str) -> bool:
        """
        대시보드로 이미지 전송
        
        Args:
            image_path: 이미지 파일 경로
            
        Returns:
            성공 여부
        """
        image_file = Path(image_path)
        
        if not image_file.exists():
            print(f"이미지 파일 없음: {image_file}")
            return False
        
        try:
            print(f"이미지 전송 중: {image_file.name}")
            
            with open(image_file, "rb") as f:
                files = {
                    'file': (image_file.name, f, 'image/jpeg')
                }
                
                response = requests.post(
                    f'{self.server_url}/img/dashboard/fire_building',
                    files=files,
                    timeout=10
                )
            
            if response.status_code == 200:
                print(f"이미지 전송 성공: {response.text}")
                return True
            else:
                print(f"이미지 전송 실패: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"이미지 전송 오류: {e}")
            return False
    
    def update_fire_buildings(self, buildings: List[str]) -> bool:
        """
        화재 건물 목록 업데이트 및 전송
        
        Args:
            buildings: 화재 건물 리스트 (예: ["sector3", "sector6"])
            
        Returns:
            성공 여부
        """
        self.current_data["fire_buildings"] = buildings
        return self.send_dashboard_json()
    
    def update_detection(self, 
                        sector: str, 
                        detections: List[Dict]) -> bool:
        """
        특정 섹터의 객체 감지 정보 업데이트 및 전송
        
        Args:
            sector: 섹터 이름 ("Alpha", "Bravo", "Charlie")
            detections: 감지 정보 리스트 [{"type": "hazmat", "count": 1}, ...]
            
        Returns:
            성공 여부
        """
        if sector not in self.current_data["detection"]:
            print(f"알 수 없는 섹터: {sector}")
            return False
        
        self.current_data["detection"][sector] = detections
        return self.send_dashboard_json()
    
    def add_detection(self, 
                     sector: str, 
                     obj_type: str, 
                     count: int = 1) -> bool:
        """
        특정 섹터에 객체 감지 추가 (기존 카운트에 추가)
        
        Args:
            sector: 섹터 이름 ("Alpha", "Bravo", "Charlie")
            obj_type: 객체 타입 ("hazmat", "enemy", "tank", "mortar", "box", "car", "missile")
            count: 개수
            
        Returns:
            성공 여부
        """
        if sector not in self.current_data["detection"]:
            print(f"알 수 없는 섹터: {sector}")
            return False
        
        # 기존 항목 찾기
        detections = self.current_data["detection"][sector]
        found = False
        for item in detections:
            if item.get("type") == obj_type:
                item["count"] = item.get("count", 0) + count
                found = True
                break
        
        # 없으면 새로 추가
        if not found:
            detections.append({"type": obj_type, "count": count})
        
        return self.send_dashboard_json()
    
    def add_point(self, point: str) -> bool:
        """
        포인트 추가 및 전송
        
        Args:
            point: 포인트 이름 (예: "alpha", "bravo", "charlie")
            
        Returns:
            성공 여부
        """
        if point not in self.current_data["points"]:
            self.current_data["points"].append(point)
            return self.send_dashboard_json()
        return True
    
    def get_current_data(self) -> Dict:
        """현재 데이터 반환"""
        return self.current_data.copy()

