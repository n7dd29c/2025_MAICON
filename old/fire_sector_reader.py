# -*- coding: utf-8 -*-
"""
화재 섹터 정보 읽기 모듈
드론이 생성한 report/2W9G.json 파일을 읽어서 화재 섹터 목록을 관리
"""
import json
import time
from pathlib import Path
from typing import List, Optional


class FireSectorReader:
    """화재 섹터 정보 읽기 클래스"""
    
    def __init__(self, 
                 report_dir: str = "report",
                 json_filename: str = "2W9G.json",
                 poll_interval: float = 1.0):
        """
        Args:
            report_dir: JSON 파일이 있는 디렉터리
            json_filename: JSON 파일명
            poll_interval: 파일 존재 여부를 체크할 주기(초)
        """
        self.report_dir = Path(report_dir)
        self.json_filename = json_filename
        self.target_path = self.report_dir / json_filename
        self.poll_interval = poll_interval
        
        # 화재 섹터 목록
        self.fire_buildings: List[str] = []
        
        # report 폴더 생성
        self.report_dir.mkdir(exist_ok=True)
    
    def wait_for_file(self) -> bool:
        """
        JSON 파일이 생성될 때까지 대기
        
        Returns:
            파일 발견 여부
        """
        print(f"[INFO] {self.target_path} 파일을 기다리는 중입니다...")
        
        max_wait_time = 300  # 최대 5분 대기
        start_time = time.time()
        
        while not self.target_path.exists():
            if time.time() - start_time > max_wait_time:
                print(f"[WARNING] 파일 대기 시간 초과: {self.target_path}")
                return False
            time.sleep(self.poll_interval)
        
        # 파일이 막 생성되는 중일 수 있으니, 살짝 대기 후 진행
        time.sleep(0.5)
        print(f"[INFO] 파일 감지: {self.target_path}")
        return True
    
    def read_fire_sectors(self) -> bool:
        """
        JSON 파일을 읽어서 화재 섹터 목록을 저장
        
        Returns:
            성공 여부
        """
        if not self.target_path.exists():
            return False
        
        try:
            with self.target_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            
            fire_buildings = data.get("fire_buildings", [])
            self.fire_buildings = fire_buildings
            print(f"[INFO] 화재 감지 섹터: {fire_buildings}")
            return True
            
        except Exception as e:
            print(f"[ERROR] JSON 파일 읽기 오류: {e}")
            return False
    
    def update_fire_sectors(self) -> bool:
        """
        JSON 파일을 주기적으로 확인하여 화재 섹터 목록 업데이트
        
        Returns:
            업데이트 여부 (새로운 정보가 있으면 True)
        """
        if not self.target_path.exists():
            return False
        
        try:
            with self.target_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            
            new_fire_buildings = data.get("fire_buildings", [])
            
            # 변경사항이 있으면 업데이트
            if new_fire_buildings != self.fire_buildings:
                old_buildings = self.fire_buildings.copy()
                self.fire_buildings = new_fire_buildings
                print(f"[INFO] 화재 섹터 업데이트: {old_buildings} → {new_fire_buildings}")
                return True
            
            return False
            
        except Exception as e:
            print(f"[ERROR] JSON 파일 읽기 오류: {e}")
            return False
    
    def is_fire_building_sector(self, sector: str) -> bool:
        """
        특정 섹터가 화재 건물 섹터인지 확인
        
        Args:
            sector: 섹터 이름 (예: "sector3", "sector6")
            
        Returns:
            화재 건물 섹터 여부
        """
        return sector in self.fire_buildings
    
    def get_fire_buildings(self) -> List[str]:
        """현재 화재 건물 섹터 목록 반환"""
        return self.fire_buildings.copy()

