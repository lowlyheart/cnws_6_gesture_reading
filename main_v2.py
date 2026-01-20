"""
CBrain Speed Game Agent - Phase 2 완성판
시계판 기반 제스처 인식 스피드 게임

Features:
- 실시간 손 검출 및 시계판 매핑
- 상태 머신 기반 순차 입력
- 오디오 신호 (준비/실행 구분)
- 한글 렌더링
- 단어 추천 엔진
- 패스 기능

실행: python main_v2.py [--simulation] [--no-audio]
"""

import cv2
import numpy as np
import math
import time
from datetime import datetime
from typing import Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum

# 로컬 모듈
from gesture_agent import (
    ClockRegion, HandDetector, GestureStateMachine, WordMatcher,
    GameState, CATEGORIES, CONSONANTS, VOWELS, combine_hangul,
    InputSession
)


# ============== 설정 ==============

@dataclass
class GameConfig:
    """게임 설정"""
    # 화면 설정
    frame_width: int = 640
    frame_height: int = 480
    
    # 시간 설정 (초)
    prep_time: float = 5.0
    exec_time: float = 1.0
    
    # 안정성 설정
    stability_threshold: float = 0.7
    
    # 오디오 설정
    use_audio: bool = True
    
    # 디스플레이 설정
    show_landmarks: bool = True
    mirror_mode: bool = True  # 거울 모드
    
    # 게임 규칙
    max_attempts: int = 3  # 단어당 최대 시도 횟수
    time_limit: float = 300.0  # 5분


# ============== UI 컴포넌트 ==============

class GameUI:
    """게임 UI 렌더링"""
    
    def __init__(self, config: GameConfig):
        self.config = config
        self.korean_renderer = None
        self._init_renderer()
        
        # 색상 정의 (BGR)
        self.colors = {
            'prep': (0, 200, 255),      # 주황 (준비)
            'exec': (0, 255, 0),        # 초록 (실행)
            'stable': (0, 255, 0),      # 초록 (안정)
            'unstable': (0, 0, 255),    # 빨강 (불안정)
            'warning': (0, 255, 255),   # 노랑
            'text': (255, 255, 255),    # 흰색
            'dim': (150, 150, 150),     # 회색
            'highlight': (255, 255, 0), # 시안
            'panel_bg': (40, 40, 40),   # 패널 배경
        }
    
    def _init_renderer(self):
        """한글 렌더러 초기화"""
        try:
            from korean_renderer import KoreanTextRenderer
            self.korean_renderer = KoreanTextRenderer(default_size=24)
        except ImportError:
            print("[UI] 한글 렌더러 없음, 기본 텍스트 사용")
            self.korean_renderer = None
    
    def put_text(self, frame: np.ndarray, text: str, 
                 position: Tuple[int, int], 
                 font_size: int = 24,
                 color: Tuple[int, int, int] = None) -> np.ndarray:
        """텍스트 그리기 (한글 지원)"""
        if color is None:
            color = self.colors['text']
        
        if self.korean_renderer:
            return self.korean_renderer.put_text(frame, text, position, font_size, color)
        else:
            scale = font_size / 30
            cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                       scale, color, max(1, int(scale * 2)))
            return frame
    
    def draw_clock(self, frame: np.ndarray, clock: ClockRegion,
                   state: GameState, current_region: Optional[int],
                   is_prep: bool) -> np.ndarray:
        """시계판 그리기"""
        cx, cy = clock.center
        radius = clock.radius
        
        # 반투명 오버레이
        overlay = frame.copy()
        
        # 외곽 원
        cv2.circle(overlay, (cx, cy), radius, self.colors['dim'], 2)
        
        # 중심 무효 영역
        cv2.circle(overlay, (cx, cy), int(radius * 0.2), self.colors['dim'], 1)
        
        # 12개 영역 및 라벨
        for i in range(12):
            angle_rad = math.radians(i * 30 - 90)
            
            # 구분선
            x1 = int(cx + radius * 0.2 * math.cos(angle_rad))
            y1 = int(cy + radius * 0.2 * math.sin(angle_rad))
            x2 = int(cx + radius * math.cos(angle_rad))
            y2 = int(cy + radius * math.sin(angle_rad))
            cv2.line(overlay, (x1, y1), (x2, y2), self.colors['dim'], 1)
            
            # 라벨 위치
            lx = int(cx + radius * 0.7 * math.cos(angle_rad))
            ly = int(cy + radius * 0.7 * math.sin(angle_rad))
            
            # 상태별 라벨 및 색상
            label, color, is_valid = self._get_region_label(i, state)
            
            if label:
                # 유효 영역 강조
                if is_valid:
                    # 배경 원
                    cv2.circle(overlay, (lx, ly), 22, (60, 60, 60), -1)
                    cv2.circle(overlay, (lx, ly), 22, color, 2)
                
                # 라벨 텍스트
                frame = self.put_text(frame, label, (lx - 12, ly + 8), 24, color)
        
        # 현재 선택 영역 하이라이트
        if current_region is not None:
            angle_rad = math.radians(current_region * 30 - 90)
            hx = int(cx + radius * 0.5 * math.cos(angle_rad))
            hy = int(cy + radius * 0.5 * math.sin(angle_rad))
            
            # 선택 표시
            highlight_color = self.colors['prep'] if is_prep else self.colors['exec']
            cv2.circle(overlay, (hx, hy), 35, highlight_color, -1)
            
            # 선택된 라벨 다시 그리기
            label, _, _ = self._get_region_label(current_region, state)
            if label:
                frame = self.put_text(frame, label, (hx - 12, hy + 8), 28, (0, 0, 0))
        
        # 블렌딩
        result = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
        return result
    
    def _get_region_label(self, region: int, state: GameState) -> Tuple[str, tuple, bool]:
        """영역별 라벨, 색상, 유효 여부 반환"""
        if state in [GameState.CATEGORY, GameState.SYLLABLE_COUNT]:
            if 1 <= region <= 5:
                return str(region), self.colors['highlight'], True
            return "", self.colors['dim'], False
        
        elif state == GameState.CONSONANT:
            label = CONSONANTS.get(region, "")
            return label, self.colors['highlight'], bool(label)
        
        elif state == GameState.VOWEL:
            label = VOWELS.get(region, "")
            return label, (255, 200, 100), bool(label)
        
        else:
            return str(region), self.colors['dim'], False
    
    def draw_info_panel(self, frame: np.ndarray, status: dict,
                        candidates: List[str], 
                        game_time: float) -> np.ndarray:
        """우측 정보 패널"""
        h, w = frame.shape[:2]
        panel_width = 220
        
        # 패널 배경
        cv2.rectangle(frame, (w - panel_width, 0), (w, h), 
                     self.colors['panel_bg'], -1)
        
        x = w - panel_width + 15
        y = 30
        line_height = 28
        
        # 게임 시간
        minutes = int(game_time) // 60
        seconds = int(game_time) % 60
        frame = self.put_text(frame, f"Time: {minutes}:{seconds:02d}", (x, y), 22)
        y += line_height
        
        # 구분선
        cv2.line(frame, (x, y), (w - 15, y), self.colors['dim'], 1)
        y += 15
        
        # 현재 상태
        state_name = status['state']
        state_korean = {
            'IDLE': '대기',
            'CATEGORY': '카테고리',
            'SYLLABLE_COUNT': '글자수',
            'CONSONANT': '자음',
            'VOWEL': '모음',
            'COMPLETE': '완료'
        }.get(state_name, state_name)
        
        frame = self.put_text(frame, f"상태: {state_korean}", (x, y), 22)
        y += line_height
        
        # 위상
        phase = status['phase']
        phase_color = self.colors['prep'] if phase == '준비' else self.colors['exec']
        frame = self.put_text(frame, f"위상: {phase}", (x, y), 22, phase_color)
        y += line_height
        
        # 남은 시간
        remaining = status['remaining']
        frame = self.put_text(frame, f"남은시간: {remaining:.1f}s", (x, y), 20)
        y += line_height
        
        # 안정성 (실행 중일 때만)
        if status['phase'] == '실행':
            stability = status.get('stability', 0)
            stab_color = self.colors['stable'] if stability >= 0.7 else self.colors['unstable']
            frame = self.put_text(frame, f"안정성: {stability:.0%}", (x, y), 20, stab_color)
        y += line_height
        
        # 구분선
        cv2.line(frame, (x, y), (w - 15, y), self.colors['dim'], 1)
        y += 15
        
        # 세션 정보
        session = status['session']
        frame = self.put_text(frame, f"카테고리: {session['category']}", (x, y), 20)
        y += line_height
        
        frame = self.put_text(frame, f"글자수: {session['syllable_count'] or '-'}", (x, y), 20)
        y += line_height
        
        frame = self.put_text(frame, f"진행: {session['progress']}", (x, y), 20)
        y += line_height
        
        # 현재 입력
        current_word = session['current_word']
        frame = self.put_text(frame, f"입력: {current_word or '-'}", (x, y), 24, 
                             self.colors['highlight'])
        y += line_height + 10
        
        # 구분선
        cv2.line(frame, (x, y), (w - 15, y), self.colors['dim'], 1)
        y += 15
        
        # 후보 단어
        frame = self.put_text(frame, "추천 단어:", (x, y), 20)
        y += line_height
        
        for i, word in enumerate(candidates[:5]):
            frame = self.put_text(frame, f" {i+1}. {word}", (x, y), 20, 
                                 (100, 255, 100))
            y += 24
        
        return frame
    
    def draw_progress_bar(self, frame: np.ndarray, status: dict,
                          state_machine) -> np.ndarray:
        """하단 진행 바"""
        h, w = frame.shape[:2]
        
        phase = status['phase']
        remaining = status['remaining']
        
        if phase == '준비':
            color = self.colors['prep']
            max_time = state_machine.prep_time
        else:
            color = self.colors['exec']
            max_time = state_machine.exec_time
        
        # 바 크기 및 위치
        bar_width = 350
        bar_height = 25
        bar_x = 20
        bar_y = h - 45
        
        # 진행률
        progress = 1 - (remaining / max_time) if max_time > 0 else 1
        fill_width = int(bar_width * progress)
        
        # 배경
        cv2.rectangle(frame, (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height),
                     (60, 60, 60), -1)
        
        # 채우기
        cv2.rectangle(frame, (bar_x, bar_y),
                     (bar_x + fill_width, bar_y + bar_height),
                     color, -1)
        
        # 테두리
        cv2.rectangle(frame, (bar_x, bar_y),
                     (bar_x + bar_width, bar_y + bar_height),
                     (200, 200, 200), 2)
        
        # 텍스트
        phase_text = f"{phase} ({remaining:.1f}s)"
        frame = self.put_text(frame, phase_text, (bar_x, bar_y - 8), 22, color)
        
        # 안정성 바 (실행 시)
        if phase == '실행':
            stability = status.get('stability', 0)
            stab_y = bar_y - 35
            stab_fill = int(bar_width * stability)
            
            stab_color = self.colors['stable'] if stability >= 0.7 else \
                        self.colors['warning'] if stability >= 0.4 else \
                        self.colors['unstable']
            
            cv2.rectangle(frame, (bar_x, stab_y),
                         (bar_x + bar_width, stab_y + 15),
                         (40, 40, 40), -1)
            cv2.rectangle(frame, (bar_x, stab_y),
                         (bar_x + stab_fill, stab_y + 15),
                         stab_color, -1)
            
            frame = self.put_text(frame, f"안정성: {stability:.0%}",
                                 (bar_x, stab_y - 5), 18, stab_color)
        
        return frame
    
    def draw_state_guide(self, frame: np.ndarray, state: GameState) -> np.ndarray:
        """상단 상태 안내"""
        guides = {
            GameState.IDLE: ("SPACE를 눌러 시작", self.colors['warning']),
            GameState.CATEGORY: ("카테고리 선택 (1~5시)", self.colors['highlight']),
            GameState.SYLLABLE_COUNT: ("글자수 선택 (1~5)", self.colors['highlight']),
            GameState.CONSONANT: ("자음 선택 (12방향)", self.colors['highlight']),
            GameState.VOWEL: ("모음 선택 (12방향)", (255, 200, 100)),
            GameState.COMPLETE: ("입력 완료! SPACE로 재시작", self.colors['stable']),
        }
        
        text, color = guides.get(state, ("", self.colors['text']))
        if text:
            frame = self.put_text(frame, text, (20, 35), 26, color)
        
        return frame
    
    def draw_hand_marker(self, frame: np.ndarray, 
                         hand_pos: Optional[Tuple[int, int]]) -> np.ndarray:
        """손 위치 마커"""
        if hand_pos:
            # 외곽 원
            cv2.circle(frame, hand_pos, 20, self.colors['stable'], 2)
            # 중심 점
            cv2.circle(frame, hand_pos, 5, self.colors['stable'], -1)
            # 십자선
            cv2.line(frame, (hand_pos[0] - 25, hand_pos[1]),
                    (hand_pos[0] + 25, hand_pos[1]), self.colors['stable'], 1)
            cv2.line(frame, (hand_pos[0], hand_pos[1] - 25),
                    (hand_pos[0], hand_pos[1] + 25), self.colors['stable'], 1)
        return frame
    
    def draw_complete_screen(self, frame: np.ndarray, 
                             word: str, candidates: List[str]) -> np.ndarray:
        """완료 화면"""
        h, w = frame.shape[:2]
        
        # 반투명 오버레이
        overlay = frame.copy()
        cv2.rectangle(overlay, (50, h//2 - 80), (w - 270, h//2 + 80),
                     (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # 입력된 단어
        frame = self.put_text(frame, "입력 완료!", (70, h//2 - 50), 28, 
                             self.colors['stable'])
        frame = self.put_text(frame, word, (70, h//2), 48, 
                             self.colors['highlight'])
        
        # 매칭 결과
        if candidates and word in candidates:
            frame = self.put_text(frame, "✓ 단어 매칭 성공!", (70, h//2 + 50), 24,
                                 self.colors['stable'])
        elif candidates:
            frame = self.put_text(frame, f"유사: {candidates[0]}", (70, h//2 + 50), 24,
                                 self.colors['warning'])
        
        return frame


# ============== 메인 애플리케이션 ==============

class SpeedGameAppV2:
    """스피드 게임 메인 애플리케이션 (Phase 2)"""
    
    def __init__(self, config: GameConfig = None):
        self.config = config or GameConfig()
        
        # 컴포넌트 초기화
        self.hand_detector = HandDetector(use_mediapipe=True)
        self.state_machine = GestureStateMachine(
            prep_time=self.config.prep_time,
            exec_time=self.config.exec_time,
            stability_threshold=self.config.stability_threshold
        )
        self.word_matcher = WordMatcher()
        self.ui = GameUI(self.config)
        
        # 오디오 초기화
        self.audio = None
        if self.config.use_audio:
            self._init_audio()
        
        # 시계판 설정
        self.clock = ClockRegion(
            center=(self.config.frame_width // 2, self.config.frame_height // 2),
            radius=min(self.config.frame_width, self.config.frame_height) // 3
        )
        
        # 카메라
        self.cap = None
        
        # 게임 상태
        self.game_start_time = None
        self.last_phase = None
        self.attempts = 0
    
    def _init_audio(self):
        """오디오 시스템 초기화"""
        try:
            from audio_signal import get_audio_system
            self.audio = get_audio_system(use_pyaudio=True)
            print("[Audio] 오디오 시스템 초기화 완료")
        except ImportError:
            print("[Audio] 오디오 모듈 없음")
            self.audio = None
    
    def connect_camera(self, device_index: int = 0) -> bool:
        """카메라 연결"""
        self.cap = cv2.VideoCapture(device_index)
        if not self.cap.isOpened():
            print(f"[Error] 카메라를 열 수 없습니다 (인덱스 {device_index})")
            return False
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
        return True
    
    def disconnect_camera(self):
        """카메라 연결 해제"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
    
    def _handle_audio(self, status: dict):
        """오디오 신호 처리"""
        if not self.audio:
            return
        
        current_phase = status['phase']
        
        # 위상 전환 감지
        if self.last_phase != current_phase:
            if current_phase == '준비':
                self.audio.play_prep_signal()
            elif current_phase == '실행':
                self.audio.play_exec_signal()
            self.last_phase = current_phase
    
    def _handle_state_change(self, old_state: GameState, new_state: GameState):
        """상태 변경 처리"""
        if not self.audio:
            return
        
        if new_state == GameState.COMPLETE:
            self.audio.play_complete_signal()
    
    def run(self, device_index: int = 0):
        """메인 실행 루프"""
        if not self.connect_camera(device_index):
            return
        
        print("=" * 50)
        print("CBrain Speed Game Agent - Phase 2")
        print("=" * 50)
        print("조작법:")
        print("  SPACE : 게임 시작/재시작")
        print("  P     : 패스 (현재 입력 건너뛰기)")
        print("  R     : 리셋")
        print("  Q     : 종료")
        print("=" * 50)
        
        # 콜백 설정
        old_state = self.state_machine.state
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("[Error] 프레임 읽기 실패")
                break
            
            # 거울 모드
            if self.config.mirror_mode:
                frame = cv2.flip(frame, 1)
            
            # 손 검출
            hand_pos = self.hand_detector.detect(frame)
            
            # 상태 머신 업데이트
            status = self.state_machine.update(hand_pos, self.clock)
            
            # 상태 변경 감지
            new_state = self.state_machine.state
            if old_state != new_state:
                self._handle_state_change(old_state, new_state)
                old_state = new_state
            
            # 오디오 처리
            self._handle_audio(status)
            
            # 게임 시간 계산
            game_time = 0.0
            if self.game_start_time:
                game_time = time.time() - self.game_start_time
            
            # 단어 후보 검색
            session = self.state_machine.session
            candidates = self.word_matcher.match(
                session.get_current_word(),
                CATEGORIES.get(session.category),
                session.syllable_count
            )
            
            # UI 렌더링
            frame = self.ui.draw_clock(frame, self.clock, 
                                       self.state_machine.state,
                                       status['current_region'],
                                       status['phase'] == '준비')
            
            frame = self.ui.draw_hand_marker(frame, hand_pos)
            frame = self.ui.draw_state_guide(frame, self.state_machine.state)
            frame = self.ui.draw_info_panel(frame, status, candidates, game_time)
            
            # 진행 바 (게임 중일 때만)
            if self.state_machine.state not in [GameState.IDLE, GameState.COMPLETE]:
                frame = self.ui.draw_progress_bar(frame, status, self.state_machine)
            
            # 완료 화면
            if self.state_machine.state == GameState.COMPLETE:
                word = session.get_current_word()
                frame = self.ui.draw_complete_screen(frame, word, candidates)
            
            # 화면 표시
            cv2.imshow("CBrain Speed Game - Phase 2", frame)
            
            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):
                self.state_machine.start_game()
                self.game_start_time = time.time()
                self.last_phase = None
                self.attempts = 0
                print("[Game] 게임 시작!")
            elif key == ord('r'):
                self.state_machine.reset()
                self.game_start_time = None
                self.last_phase = None
                print("[Game] 리셋")
            elif key == ord('p'):
                # 패스 기능
                if self.state_machine.state not in [GameState.IDLE, GameState.COMPLETE]:
                    print("[Game] 패스!")
                    if self.audio:
                        self.audio.play_error_signal()
                    self.state_machine.start_game()  # 처음부터 다시
        
        self.disconnect_camera()


# ============== 시뮬레이션 모드 ==============

class SimulationAppV2:
    """마우스로 테스트하는 시뮬레이션 모드 (Phase 2)"""
    
    def __init__(self, config: GameConfig = None):
        self.config = config or GameConfig()
        
        self.state_machine = GestureStateMachine(
            prep_time=self.config.prep_time,
            exec_time=self.config.exec_time,
            stability_threshold=self.config.stability_threshold
        )
        self.word_matcher = WordMatcher()
        self.ui = GameUI(self.config)
        
        self.clock = ClockRegion(
            center=(self.config.frame_width // 2, self.config.frame_height // 2),
            radius=180
        )
        
        self.mouse_pos = (0, 0)
        self.game_start_time = None
    
    def mouse_callback(self, event, x, y, flags, param):
        """마우스 콜백"""
        self.mouse_pos = (x, y)
    
    def run(self):
        """시뮬레이션 실행"""
        cv2.namedWindow("Simulation - Phase 2")
        cv2.setMouseCallback("Simulation - Phase 2", self.mouse_callback)
        
        print("=" * 50)
        print("시뮬레이션 모드 (Phase 2)")
        print("=" * 50)
        print("마우스로 손 위치를 시뮬레이션합니다.")
        print("SPACE: 시작, P: 패스, R: 리셋, Q: 종료")
        print("=" * 50)
        
        while True:
            frame = np.zeros((self.config.frame_height, 
                             self.config.frame_width, 3), dtype=np.uint8)
            frame[:] = (30, 30, 30)
            
            # 상태 업데이트
            status = self.state_machine.update(self.mouse_pos, self.clock)
            
            # 게임 시간
            game_time = 0.0
            if self.game_start_time:
                game_time = time.time() - self.game_start_time
            
            # 단어 후보
            session = self.state_machine.session
            candidates = self.word_matcher.match(
                session.get_current_word(),
                CATEGORIES.get(session.category),
                session.syllable_count
            )
            
            # UI 렌더링
            frame = self.ui.draw_clock(frame, self.clock,
                                       self.state_machine.state,
                                       status['current_region'],
                                       status['phase'] == '준비')
            
            frame = self.ui.draw_hand_marker(frame, self.mouse_pos)
            frame = self.ui.draw_state_guide(frame, self.state_machine.state)
            frame = self.ui.draw_info_panel(frame, status, candidates, game_time)
            
            if self.state_machine.state not in [GameState.IDLE, GameState.COMPLETE]:
                frame = self.ui.draw_progress_bar(frame, status, self.state_machine)
            
            if self.state_machine.state == GameState.COMPLETE:
                word = session.get_current_word()
                frame = self.ui.draw_complete_screen(frame, word, candidates)
            
            cv2.imshow("Simulation - Phase 2", frame)
            
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                self.state_machine.start_game()
                self.game_start_time = time.time()
            elif key == ord('r'):
                self.state_machine.reset()
                self.game_start_time = None
            elif key == ord('p'):
                if self.state_machine.state not in [GameState.IDLE, GameState.COMPLETE]:
                    self.state_machine.start_game()
        
        cv2.destroyAllWindows()


# ============== 엔트리 포인트 ==============

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CBrain Speed Game Agent - Phase 2")
    parser.add_argument("--device", type=int, default=0, help="카메라 인덱스")
    parser.add_argument("--simulation", action="store_true", 
                       help="시뮬레이션 모드 (웹캠 없이)")
    parser.add_argument("--no-audio", action="store_true",
                       help="오디오 비활성화")
    parser.add_argument("--prep-time", type=float, default=1.5,
                       help="준비 시간 (초)")
    parser.add_argument("--exec-time", type=float, default=1.0,
                       help="실행 시간 (초)")
    
    args = parser.parse_args()
    
    # 설정
    config = GameConfig(
        use_audio=not args.no_audio,
        prep_time=args.prep_time,
        exec_time=args.exec_time
    )
    
    # 실행
    if args.simulation:
        app = SimulationAppV2(config)
    else:
        app = SpeedGameAppV2(config)
        
    if args.simulation:
        app.run()
    else:
        app.run(device_index=args.device)
