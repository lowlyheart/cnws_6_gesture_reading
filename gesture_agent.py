"""
시계판 기반 제스처 인식 에이전트
팀 전략: 손 위치를 시계판 영역으로 매핑하여 카테고리/글자수/자음/모음 인코딩

핵심 아이디어:
- 화면을 시계처럼 12등분
- 손의 위치가 어느 영역에 있는지 감지
- 상태 머신으로 순차적으로 정보 입력
- 준비 시간(이동) / 실행 시간(확정) 구분
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from enum import Enum, auto
from collections import deque
import time
import math


# ============== 상수 정의 ==============

class GameState(Enum):
    """게임 상태"""
    IDLE = auto()           # 대기
    CATEGORY = auto()       # 카테고리 선택
    SYLLABLE_COUNT = auto() # 글자수 선택
    CONSONANT = auto()      # 자음 입력
    VOWEL = auto()          # 모음 입력
    COMPLETE = auto()       # 완료


# ============== 동적 시계판 매핑 ==============
# 각 세션 단계별로 다른 등분 사용:
# - 카테고리: 5등분
# - 글자수: 8등분 (1~8글자)
# - 자음(초성): 14등분
# - 모음(중성): 10등분

# 카테고리 정의 (5등분)
CATEGORIES = {
    0: "감정 및 상태",
    1: "행동 및 동사",
    2: "동물",
    3: "스포츠",
    4: "날씨 및 자연"
}
NUM_CATEGORIES = 5

# 글자수 (8등분, 1~8글자)
NUM_SYLLABLE_OPTIONS = 8

# 자음 배치 (14등분) - ㄱ부터 ㅎ까지 순서대로
CONSONANTS = {
    0: 'ㄱ',
    1: 'ㄴ',
    2: 'ㄷ',
    3: 'ㄹ',
    4: 'ㅁ',
    5: 'ㅂ',
    6: 'ㅅ',
    7: 'ㅇ',
    8: 'ㅈ',
    9: 'ㅊ',
    10: 'ㅋ',
    11: 'ㅌ',
    12: 'ㅍ',
    13: 'ㅎ',
}
NUM_CONSONANTS = 14

# 모음 배치 (10등분) - 기본 모음
VOWELS = {
    0: 'ㅏ',
    1: 'ㅑ',
    2: 'ㅓ',
    3: 'ㅕ',
    4: 'ㅗ',
    5: 'ㅛ',
    6: 'ㅜ',
    7: 'ㅠ',
    8: 'ㅡ',
    9: 'ㅣ',
}
NUM_VOWELS = 10

# 각 상태별 등분 수
DIVISIONS_BY_STATE = {
    GameState.CATEGORY: NUM_CATEGORIES,        # 5등분
    GameState.SYLLABLE_COUNT: NUM_SYLLABLE_OPTIONS,  # 8등분
    GameState.CONSONANT: NUM_CONSONANTS,       # 14등분
    GameState.VOWEL: NUM_VOWELS,               # 10등분
}


# ============== 한글 조합 유틸리티 ==============

def combine_hangul(cho: str, jung: str, jong: str = '') -> str:
    """초성, 중성, (종성)을 조합하여 한글 글자 생성"""
    CHO_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 
                'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
    JUNG_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ',
                 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
    JONG_LIST = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ',
                 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ',
                 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
    
    try:
        cho_idx = CHO_LIST.index(cho)
        jung_idx = JUNG_LIST.index(jung)
        jong_idx = JONG_LIST.index(jong) if jong else 0
        
        # 한글 유니코드 공식: 0xAC00 + (초성×21×28) + (중성×28) + 종성
        code = 0xAC00 + (cho_idx * 21 * 28) + (jung_idx * 28) + jong_idx
        return chr(code)
    except ValueError:
        return cho + jung + jong  # 조합 실패시 그대로 반환


# ============== 시계판 영역 매핑 ==============

@dataclass
class ClockRegion:
    """동적 시계판 영역 - 등분 수에 따라 영역 크기 조절"""
    center: Tuple[int, int]  # 화면 중심
    radius: int              # 시계판 반지름
    
    def get_region_index(self, x: int, y: int, num_divisions: int = 12) -> Optional[int]:
        """
        좌표가 어느 영역에 있는지 반환 (동적 등분)
        
        Args:
            x, y: 좌표
            num_divisions: 등분 수 (5, 8, 10, 14 등)
        
        Returns:
            영역 인덱스 (0 ~ num_divisions-1) 또는 None (중심 영역)
        """
        cx, cy = self.center
        dx = x - cx
        dy = y - cy
        
        distance = math.sqrt(dx*dx + dy*dy)
        
        # 중심 근처는 무효 (반지름의 20% 이내)
        if distance < self.radius * 0.2:
            return None
        
        # 각도 계산 (12시 방향이 0도)
        angle = math.atan2(dx, -dy)  # y축 반전 (화면 좌표계)
        angle_deg = math.degrees(angle)
        if angle_deg < 0:
            angle_deg += 360
        
        # 동적 등분 (각 영역 = 360 / num_divisions 도)
        degrees_per_region = 360.0 / num_divisions
        half_region = degrees_per_region / 2
        
        region = int((angle_deg + half_region) % 360 / degrees_per_region)
        return region
    
    def get_region_for_state(self, x: int, y: int, state: 'GameState') -> Optional[int]:
        """
        현재 게임 상태에 맞는 등분으로 영역 반환
        """
        num_divisions = DIVISIONS_BY_STATE.get(state, 12)
        return self.get_region_index(x, y, num_divisions)
    
    def get_region_angle(self, region_index: int, num_divisions: int) -> float:
        """영역의 중심 각도 반환 (라디안)"""
        degrees_per_region = 360.0 / num_divisions
        angle_deg = region_index * degrees_per_region - 90  # 12시 방향 기준
        return math.radians(angle_deg)
    
    def get_region_center(self, region_index: int, num_divisions: int, 
                          radius_ratio: float = 0.65) -> Tuple[int, int]:
        """영역 중심 좌표 반환"""
        angle = self.get_region_angle(region_index, num_divisions)
        cx, cy = self.center
        x = int(cx + self.radius * radius_ratio * math.cos(angle))
        y = int(cy + self.radius * radius_ratio * math.sin(angle))
        return (x, y)


# ============== 손 검출기 ==============

class HandDetector:
    """
    간단한 손 검출기 (검정 장갑 기반 색상 검출)
    실제 구현에서는 MediaPipe 등 사용 권장
    """
    
    def __init__(self, use_mediapipe: bool = True):
        self.use_mediapipe = use_mediapipe
        self.mp_hands = None
        self.hands = None
        
        if use_mediapipe:
            try:
                import mediapipe as mp
                self.mp_hands = mp.solutions.hands
                self.hands = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.5
                )
                self.mp_draw = mp.solutions.drawing_utils
            except ImportError:
                print("MediaPipe 없음, 색상 기반 검출 사용")
                self.use_mediapipe = False
    
    def detect(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        손 중심 좌표 검출
        Returns: (x, y) 또는 None
        """
        if self.use_mediapipe and self.hands is not None:
            return self._detect_mediapipe(frame)
        else:
            return self._detect_color(frame)
    
    def _detect_mediapipe(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """MediaPipe를 사용한 손 검출"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            h, w = frame.shape[:2]
            
            # 손바닥 중심 (landmark 9: 중지 MCP)
            cx = int(hand.landmark[9].x * w)
            cy = int(hand.landmark[9].y * h)
            
            return (cx, cy)
        return None
    
    def _detect_color(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """색상 기반 검출 (검정 장갑)"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 검정색 범위 (낮은 채도, 낮은 명도)
        lower = np.array([0, 0, 0])
        upper = np.array([180, 255, 50])
        
        mask = cv2.inRange(hsv, lower, upper)
        
        # 노이즈 제거
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        # 컨투어 찾기
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 가장 큰 컨투어
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 1000:  # 최소 면적
                M = cv2.moments(largest)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    return (cx, cy)
        
        return None
    
    def draw_landmarks(self, frame: np.ndarray, results=None):
        """MediaPipe 랜드마크 그리기"""
        if self.use_mediapipe and self.hands is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)


# ============== 상태 머신 ==============

@dataclass
class InputSession:
    """현재 입력 세션 정보"""
    category: Optional[int] = None
    syllable_count: Optional[int] = None
    consonants: List[str] = field(default_factory=list)
    vowels: List[str] = field(default_factory=list)
    current_syllable: int = 0
    
    def get_current_word(self) -> str:
        """현재까지 입력된 글자 조합"""
        result = []
        for i in range(len(self.consonants)):
            cho = self.consonants[i] if i < len(self.consonants) else ''
            jung = self.vowels[i] if i < len(self.vowels) else ''
            if cho and jung:
                result.append(combine_hangul(cho, jung))
            elif cho:
                result.append(cho)
        return ''.join(result)
    
    def reset(self):
        """세션 초기화"""
        self.category = None
        self.syllable_count = None
        self.consonants = []
        self.vowels = []
        self.current_syllable = 0


class GestureStateMachine:
    """제스처 인식 상태 머신"""
    
    def __init__(self, prep_time: float = 1.5, exec_time: float = 1.0,
                 stability_threshold: float = 0.7):
        """
        Args:
            prep_time: 준비 시간 (초) - 손 이동 허용
            exec_time: 실행 시간 (초) - 손 고정, 값 확정
            stability_threshold: 안정성 임계값 (0~1, 해당 비율 이상 같은 영역이면 확정)
        """
        self.prep_time = prep_time
        self.exec_time = exec_time
        self.stability_threshold = stability_threshold
        
        self.state = GameState.IDLE
        self.session = InputSession()
        
        self.state_start_time = None
        self.is_prep_phase = True
        self.current_region = None
        self.confirmed_region = None
        
        # 안정성 검증용: 실행 시간 동안의 영역 기록
        self.exec_region_history: deque = deque(maxlen=30)  # 약 1초 (30fps 가정)
        
        # 콜백
        self.on_state_change = None
        self.on_input_confirmed = None
    
    def start_game(self):
        """게임 시작"""
        self.state = GameState.CATEGORY
        self.session.reset()
        self._start_new_phase()
    
    def _start_new_phase(self):
        """새 위상 시작"""
        self.state_start_time = time.time()
        self.is_prep_phase = True
        self.confirmed_region = None
        self.exec_region_history.clear()
    
    def _get_stable_region(self) -> Optional[int]:
        """
        실행 시간 동안 기록된 영역 중 가장 안정적인(빈도 높은) 영역 반환
        안정성 임계값을 넘지 못하면 None
        """
        if not self.exec_region_history:
            return None
        
        # None 제외하고 카운트
        valid_regions = [r for r in self.exec_region_history if r is not None]
        if not valid_regions:
            return None
        
        # 가장 빈번한 영역
        from collections import Counter
        counter = Counter(valid_regions)
        most_common_region, count = counter.most_common(1)[0]
        
        # 안정성 검증
        stability = count / len(self.exec_region_history)
        if stability >= self.stability_threshold:
            return most_common_region
        
        return None  # 불안정
    
    def update(self, hand_position: Optional[Tuple[int, int]], 
               clock: ClockRegion) -> Dict:
        """
        매 프레임 업데이트
        
        Returns:
            상태 정보 딕셔너리
        """
        if self.state == GameState.IDLE or self.state == GameState.COMPLETE:
            return self._get_status()
        
        elapsed = time.time() - self.state_start_time
        
        # 준비 → 실행 전환
        if self.is_prep_phase and elapsed >= self.prep_time:
            self.is_prep_phase = False
            self.state_start_time = time.time()
            self.exec_region_history.clear()
            elapsed = 0
        
        # 손 위치 → 영역 매핑 (상태에 따른 동적 등분)
        if hand_position:
            self.current_region = clock.get_region_for_state(*hand_position, self.state)
        else:
            self.current_region = None
        
        # 실행 시간 동안 영역 기록
        if not self.is_prep_phase:
            self.exec_region_history.append(self.current_region)
        
        # 실행 시간 종료 → 값 확정
        if not self.is_prep_phase and elapsed >= self.exec_time:
            self._confirm_input()
        
        return self._get_status()
    
    def _confirm_input(self):
        """현재 입력 확정 및 다음 상태로 전환"""
        # 안정적인 영역 확인
        region = self._get_stable_region()
        
        if region is None:
            # 불안정한 입력 → 다시 시도
            print(f"[Warning] 불안정한 입력 감지, 다시 시도하세요.")
            self._start_new_phase()
            return
        
        self.confirmed_region = region
        
        if self.state == GameState.CATEGORY:
            self.session.category = region  # 0~4
            self.state = GameState.SYLLABLE_COUNT
            
        elif self.state == GameState.SYLLABLE_COUNT:
            self.session.syllable_count = region + 1  # 0→1글자, 1→2글자, ...
            self.state = GameState.CONSONANT
            
        elif self.state == GameState.CONSONANT:
            consonant = CONSONANTS.get(region, '?')
            self.session.consonants.append(consonant)
            self.state = GameState.VOWEL
            
        elif self.state == GameState.VOWEL:
            vowel = VOWELS.get(region, '?')
            self.session.vowels.append(vowel)
            self.session.current_syllable += 1
            
            # 모든 글자 입력 완료?
            if self.session.current_syllable >= self.session.syllable_count:
                self.state = GameState.COMPLETE
            else:
                self.state = GameState.CONSONANT
        
        if self.state != GameState.COMPLETE:
            self._start_new_phase()
        
        # 콜백 호출
        if self.on_input_confirmed:
            self.on_input_confirmed(self.state, self.session)
    
    def _get_status(self) -> Dict:
        """현재 상태 정보"""
        elapsed = time.time() - self.state_start_time if self.state_start_time else 0
        
        if self.is_prep_phase:
            phase = "준비"
            remaining = max(0, self.prep_time - elapsed)
        else:
            phase = "실행"
            remaining = max(0, self.exec_time - elapsed)
        
        # 안정성 계산
        stability = 0.0
        if self.exec_region_history:
            valid = [r for r in self.exec_region_history if r is not None]
            if valid:
                from collections import Counter
                counter = Counter(valid)
                _, count = counter.most_common(1)[0]
                stability = count / len(self.exec_region_history)
        
        return {
            "state": self.state.name,
            "phase": phase,
            "remaining": remaining,
            "current_region": self.current_region,
            "stability": stability,
            "session": {
                "category": CATEGORIES.get(self.session.category, ""),
                "syllable_count": self.session.syllable_count,
                "current_word": self.session.get_current_word(),
                "progress": f"{self.session.current_syllable}/{self.session.syllable_count or '?'}"
            }
        }
    
    def reset(self):
        """상태 초기화"""
        self.state = GameState.IDLE
        self.session.reset()
        self.state_start_time = None
        self.exec_region_history.clear()


# ============== 단어 매칭 엔진 ==============

class WordMatcher:
    """단어 추천 엔진 (유사도 매칭 포함)"""
    
    # 기본 단어 리스트 (50개)
    WORD_LIST = {
        "감정 및 상태": ["기쁨", "슬픔", "화남", "졸음", "배고픔", 
                     "덥다", "춥다", "사랑", "놀람", "창피"],
        "행동 및 동사": ["수영", "달리기", "요리", "운전", "자전거",
                     "낚시", "사진찍기", "전화하기", "청소", "공부"],
        "동물": ["토끼", "코끼리", "나비", "고양이", "고릴라",
               "펭귄", "기린", "악어", "뱀", "새"],
        "스포츠": ["농구", "야구", "탁구", "양궁", "역도",
                 "태권도", "볼링", "골프", "스쿼시", "스키"],
        "날씨 및 자연": ["해", "비", "눈", "번개", "바람",
                     "구름", "달", "별", "꽃", "나무"]
    }
    
    # 유사 자음 매핑 (혼동하기 쉬운 자음들)
    SIMILAR_CONSONANTS = {
        'ㄱ': ['ㄲ', 'ㅋ'],
        'ㄲ': ['ㄱ', 'ㅋ'],
        'ㄷ': ['ㄸ', 'ㅌ'],
        'ㄸ': ['ㄷ', 'ㅌ'],
        'ㅂ': ['ㅃ', 'ㅍ'],
        'ㅃ': ['ㅂ', 'ㅍ'],
        'ㅅ': ['ㅆ'],
        'ㅆ': ['ㅅ'],
        'ㅈ': ['ㅉ', 'ㅊ'],
        'ㅉ': ['ㅈ', 'ㅊ'],
    }
    
    def __init__(self):
        # 플랫 리스트 생성
        self.all_words = []
        for words in self.WORD_LIST.values():
            self.all_words.extend(words)
    
    def match(self, partial: str, category: Optional[str] = None,
              syllable_count: Optional[int] = None) -> List[str]:
        """
        부분 입력으로 단어 추천 (정확 매칭 + 유사 매칭)
        
        Args:
            partial: 현재까지 입력된 글자들
            category: 카테고리 (있으면 해당 카테고리만 검색)
            syllable_count: 글자수 (있으면 해당 글자수만 검색)
        """
        # 카테고리 필터
        if category and category in self.WORD_LIST:
            search_words = self.WORD_LIST[category]
        else:
            search_words = self.all_words
        
        exact_matches = []
        fuzzy_matches = []
        
        for word in search_words:
            # 글자수 필터
            if syllable_count and len(word) != syllable_count:
                continue
            
            if not partial:
                exact_matches.append(word)
            elif word.startswith(partial):
                exact_matches.append(word)
            elif self._is_similar(partial, word):
                fuzzy_matches.append(word)
        
        # 정확 매칭 우선, 그 다음 유사 매칭
        result = exact_matches + fuzzy_matches
        return result[:5]
    
    def _is_similar(self, partial: str, word: str) -> bool:
        """두 문자열이 유사한지 확인 (자음 혼동 허용)"""
        if len(partial) > len(word):
            return False
        
        for i, char in enumerate(partial):
            if i >= len(word):
                return False
            
            if char == word[i]:
                continue
            
            # 초성 비교
            partial_cho = self._get_choseong(char)
            word_cho = self._get_choseong(word[i])
            
            if partial_cho and word_cho:
                # 유사 자음인지 확인
                similar = self.SIMILAR_CONSONANTS.get(partial_cho, [])
                if word_cho in similar or partial_cho == word_cho:
                    continue
            
            return False
        
        return True
    
    def _get_choseong(self, char: str) -> Optional[str]:
        """글자에서 초성 추출"""
        CHO_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 
                    'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
        
        # 이미 자음이면 그대로 반환
        if char in CHO_LIST:
            return char
        
        # 한글 글자면 초성 추출
        code = ord(char)
        if 0xAC00 <= code <= 0xD7A3:
            cho_idx = (code - 0xAC00) // (21 * 28)
            return CHO_LIST[cho_idx]
        
        return None
    
    def get_exact_match(self, word: str, category: Optional[str] = None) -> bool:
        """정확히 일치하는 단어가 있는지 확인"""
        if category and category in self.WORD_LIST:
            return word in self.WORD_LIST[category]
        return word in self.all_words


# ============== 메인 테스트 ==============

if __name__ == "__main__":
    # 한글 조합 테스트
    print("한글 조합 테스트:")
    print(f"  ㅌ + ㅗ = {combine_hangul('ㅌ', 'ㅗ')}")
    print(f"  ㄱ + ㅣ = {combine_hangul('ㄱ', 'ㅣ')}")
    print(f"  토 + 끼 = 토끼")
    
    # 시계판 영역 테스트
    print("\n시계판 영역 테스트:")
    clock = ClockRegion(center=(320, 240), radius=200)
    
    test_points = [
        (320, 40),   # 12시
        (420, 100),  # 1-2시
        (520, 240),  # 3시
        (320, 440),  # 6시
    ]
    
    for x, y in test_points:
        region = clock.get_region_index(x, y)
        print(f"  ({x}, {y}) → 영역 {region}")
    
    # 단어 매칭 테스트
    print("\n단어 매칭 테스트:")
    matcher = WordMatcher()
    print(f"  '토' 입력, 동물 카테고리: {matcher.match('토', '동물')}")
    print(f"  빈 입력, 2글자: {matcher.match('', syllable_count=2)}")
