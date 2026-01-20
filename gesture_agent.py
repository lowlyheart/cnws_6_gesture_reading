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
    
    # 기본 단어 리스트 (핵심 단어 우선)
    CORE_WORDS = {
        "동물": ['토끼', '코끼리', '나비', '고양이', '고릴라', '펭귄', '기린', '악어', '뱀', '새'],
        "스포츠": ['농구', '야구', '탁구', '양궁', '역도', '태권도', '볼링', '골프', '스쿼시', '스키']
    }

    # 확장 데이터 (dic3,4.py에서 가져옴)
    SPORTS_RAW = [
        '농구', '야구', '탁구', '양궁', '역도', '태권도', '볼링', '골프', '스쿼시', '스키',
        '3x3 농구', '5인제 축구', '7인제 럭비', '7인제 축구', '가라테', '걷기', '걷기대회', '검도',
        '검술', '게이트볼', '게일릭풋볼', '경보', '고', '고누', '골볼', '공수도', '국무도', '국술',
        '국학기공', '궁도', '그라운드골프', '그래블', '근대5종', '글라이딩', '기계체조', '나인핀 볼링',
        '낚시', '널뛰기', '넷볼', '노르딕복합', '노르딕스키', '높이뛰기', '다이빙', '다트', '다트체리',
        '단거리달리기', '당구', '댄스스포츠', '도그슬레딩', '듀애슬론', '드래곤보트', '드래그레이싱',
        '드론 레이싱', '등산', '디스크골프', '라운더스', '라켓볼', '라크로스', '래프팅', '랠리',
        '럭비', '럭비리그', '럭비유니온', '레슬링', '로데오', '로드사이클', '론볼', '롤러더비',
        '롤러스포츠', '롤러프리스타일', '롤러하키', '루지', '리드클라이밍', '리듬체조', '마라톤',
        '마작', '멀리뛰기', '모터레이싱', '모터보트', '모터사이클', '모터스포츠', '모토크로스',
        '무도', '무에타이', '미니골프', '미식축구', '바둑', '바디보드', '바이애슬론', '배구',
        '배드민턴', '밴디', '벨리댄스', '보디빌딩', '보치아', '복싱', '볼더링', '볼링', '봅슬레이', '부울',
        '불', '브라질리언주짓수', '브레이킹', '브리지', '비둘기레이싱', '비엠엑스', '비엠엑스 프리스타일',
        '비치럭비', '비치발리볼', '비치사커', '비치테니스', '비치핸드볼', '빅에어', '사격', '사바트',
        '사이클', '사이클로크로스', '사이클링', '사이클폴로', '산악', '산악오리엔티어링', '산악자전거',
        '삼보', '생활무용', '샹치', '서핑', '세단뛰기', '세일링', '세팍타크로', '소방', '소프트볼',
        '쇼다운', '쇼댄스', '쇼트트랙', '수구', '수상구조', '수상스키', '수영', '수중 핀수영',
        '수중럭비', '수중하키', '슈퍼모토', '슈퍼볼', '슐런', '스노모빌', '스노보드', '스노우슈잉',
        '스노클링', '스누커', '스카이다이빙', '스케이트보드', '스케이트크로스', '스케이팅', '스켈레톤',
        '스쿠버다이빙', '스쿠터링', '스키오리엔티어링', '스키점프', '스키크로스', '스탠드업패들',
        '스트롱맨', '스트리트볼', '스포츠스태킹', '스포츠클라이밍', '스포츠피싱', '스피닝',
        '스피드민턴', '스피드스케이팅', '스피드웨이', '스피드클라이밍', '스피어피싱', '슬레지하키',
        '슬레지활강', '슬로프스타일', '승마', '씨름', '아이스댄스', '아이스슬레지', '아이스클라이밍',
        '아이스하키', '아이키도', '아쿠아슬론', '아티스틱스위밍', '아티스틱스케이팅', '알파인스키',
        '에어로빅', '에어스포츠', '엔듀로', '연날리기', '예술', '오리엔티어링', '오목',
        '오스트레일리안 풋볼', '오지풋볼', '오토크로스', '오픈워터수영', '오픈워터스위밍', '요트',
        '용무도', '우드볼', '우슈', '울트라마라톤', '원반던지기', '웨이크보드', '웨이크서핑',
        '웨이크스포츠', '윈드서핑', '윙수트', '유도', '육상', '이스포츠', '이식인체육', '이종격투기',
        '인라인다운힐', '인라인스피드스케이팅', '인라인알파인', '인라인하키', '인명구조',
        '자동차경주', '장거리달리기', '장기', '장대높이뛰기', '전통선술', '정구', '제기차기',
        '조정', '족구', '종합격투기', '종합무술', '좌식배구', '좌식아이스하키', '주드폼', '주짓수',
        '줄넘기', '줄다리기', '중거리달리기', '창던지기', '창술', '철인3종', '체스', '체조',
        '축구', '치어리딩', '카누', '카누슬라럼', '카누폴로', '카라테', '카바디', '카약', '카약낚시',
        '카약폴로', '카이트보딩', '카이트서핑', '카트', '캐나다풋볼', '캐롬', '캐스팅', '컬링',
        '케틀벨 스포츠', '코넷', '코프볼', '쿵후', '크로스컨트리', '크로스컨트리스키',
        '크로스트라이애슬론', '크로스핏', '크로케', '크리켓', '클라이밍', '킥복싱', '킥볼',
        '택견', '테니스', '테크볼', '투호', '트라이애슬론', '트라이얼', '트랙사이클', '트램펄린',
        '트레일러닝', '티볼', '파델', '파라사이클', '파라수영', '파라스노보드',
        '파워리프팅', '파워보트', '파워사커', '파쿠르', '패들', '패들보드', '패들테니스',
        '패러글라이딩', '패러모터', '패러조정', '패러카누', '페새팔로', '페탕크', '펜싱', '펠로타',
        '포격', '포뮬러원', '포커', '포켓볼', '포환던지기', '폴로', '풋볼', '풋살', '프리다이빙',
        '프리스타일', '프리스타일스키', '프리즈비', '프리테니스', '플라잉디스크', '플로어볼',
        '피겨스케이팅', '피구', '피클볼', '피트니스', '핀수영', '필드하키', '하이다이빙', '하키',
        '하프파이프', '한궁', '합기도', '항공', '항공스포츠', '해머던지기', '핸드볼', '행글라이딩',
        '허들', '헐링', '헤이볼', '휠체어농구', '휠체어럭비', '휠체어테니스', '힙합'
    ]

    ANIMAL_RAW = [
        '토끼', '코끼리', '나비', '고양이', '고릴라', '펭귄', '기린', '악어', '뱀', '새',
        '사자', '호랑이', '하마', '치타', '표범', '늑대', '여우', '판다', '낙타', '사슴',
        '돼지', '염소', '고래', '상어', '문어', '개미', '매미', '거미', '나방', '쥐', '게',
        '말', '소', '양', '닭', '오리', '거위', '까치', '참새', '제비', '매', '수달',
        '강아지', '다람쥐', '햄스터', '두더지', '캥거루', '코알라', '얼룩말', '코뿔소',
        '너구리', '스컹크', '거북이', '오징어', '해파리', '독수리', '부엉이', '까마귀',
        '개구리', '도마뱀', '앵무새', '비둘기', '올빼미', '잠자리', '메뚜기', '귀뚜라미',
        '무당벌레', '불가사리', '바다표범', '하이에나', '나무늘보', '도롱뇽', '오리너구리',
        '카멜레온', '아르마딜로', '미어캣', '플라밍고', '고슴도치', '붉감펭', '쏨뱅이',
        '붉은쏨뱅이', '홍살치', '큰미역치', '풀미역치', '성대', '밑달갱이', '쌍뿔달재', '꼬마달재',
        '히메성대', '가시달강어', '고지달재', '뿔성대', '달강어', '밑성대', '황성대', '별성대',
        '부채꼬리실고기', '띠거물가시치', '별실고기', '신도해마', '왕관해마', '가시해마', '복해마',
        '산호해마', '점해마', '유령실고기', '잘피실고기', '실고기', '거물가시치', '풀해마', '홍대치',
        '청대치', '대주둥치', '드렁허리', '걸장어', '쭉지성대', '별쭉지성대', '벌감펭', '에보시감펭',
        '퉁쏠치', '홍감펭', '꽃감펭', '미역치', '쑤기미', '일지말락쏠치', '말락쏠치', '제주쏠치',
        '도자감펭', '감펭', '사자고기', '점줄쏠치', '솔배감펭', '청쏠치', '점쏠배감펭', '나비쏠배감펭',
        '기점쏠배감펭', '흰점쏠배감펭', '뿔쏠배감펭', '독가시치', '기름갈치꼬치', '갈치꼬치', '꼬치삼치',
        '동갈삼치', '삼치', '평삼치', '재방어', '샛돔', '연어병치', '섬샛돔', '보라병치',
        '가시돔', '얼룩가시돔', '동강가시돔', '연줄가시돔', '돛새치', '청새치', '녹새치', '백새치',
        '황새치', '민달고기', '달고기', '민태', '민태붙이', '가시민태', '말쥐치', '객주리',
        '날개쥐치', '나뭇잎쥐치', '그물쥐치', '쥐치', '별쥐치', '무늬쥐치', '거울쥐치', '불룩쥐치',
        '실쥐치', '민쥐치', '회색쥐치', '흑밀복', '흰점복', '청복', '복섬', '졸복',
        '매끄러운복섬', '매끄러운복', '까칠복', '까치복', '검복', '황복', '자주복', '별복',
        '흰밀복', '은밀복', '밀복', '가시복', '불가시복', '강복', '국수사촌', '국수융단상어',
        '전자리상어', '수염상어', '고래상어', '환도상어', '청상아리', '백상아리', '흉상어', '무태흉상어', '갈색꼬리흉상어', '흑기흉상어', '뱀상어', '귀상어', '홍어', '상가오리',
        '노랑가오리', '흰가오리', '색가오리', '목탁가오리', '매가오리', '가오리', '만타가오리', '톱가오리',
        '은상어', '갈은상어', '뿔은상어', '칠성상어', '두툽상어', '개상어', '곱상어', '돔발상어',
        '모조리상어', '기름상어', '뱀상어붙이', '청상어', '악상어붙이', '환도상어붙이', '고양이상어', '점박이상어',
        '까치상어', '별상어', '민달팽이', '군소', '개조개', '대합', '바지락', '가리비',
        '굴', '전복', '소라', '멍게', '해삼', '말미잘', '산호', '해파리', '물벼룩',
        '새우', '가재', '게', '투구게', '지네', '거미', '전갈', '지렁이', '거머리',
        '달팽이', '굴미역치', '미역치사촌', '쏠치사촌', '민쏠치', '미역쏠치', '흰점쏠치', '범쏠치',
        '말락쏠치사촌', '점쏠치', '가시쏠치', '뿔쏠치', '돌쏠치', '장미감펭', '홍합', '피조개',
        '꼬막', '전복사촌', '대합사촌', '가리비사촌', '멍게사촌', '해삼사촌', '말미잘사촌', '산호사촌',
        '해파리사촌', '물벼룩사촌', '새우사촌', '가재사촌', '게사촌', '투구게사촌', '지네사촌', '거미사촌',
        '전갈사촌', '지렁이사촌', '거머리사촌', '달팽이사촌', '곤들매기', '열목어', '산천어', '송어',
        '무지개송어', '은어', '빙어', '멸치', '청어', '정어리', '눈치', '누치',
        '잉어', '붕어', '떡붕어', '금어', '잉어붙이', '참붕어', '모래무지', '피라미',
        '갈겨니', '참갈겨니', '끄리', '치리', '살치', '강준치', '가시납줄개', '납줄개',
        '흰줄납줄개', '각시붕어', '떡납줄경', '줄납자루', '칼납자루', '묵납자루', '임실납자루', '납자루',
        '큰납지리', '납지리', '가시복', '졸복', '복섬', '검복', '황복', '자주복',
        '흰밀복', '밀복', '매끄러운복', '까치복', '물메기', '꼼치', '미역치', '성대',
        '달강어', '양태', '능성어', '다금바리', '붉바리', '자바리', '민어', '조기',
        '부세', '수조기', '보구치', '흑조기', '참조기', '돔', '감성돔', '참돔',
        '돌돔', '강담돔', '벵에돔', '긴꼬리벵에돔', '범돔', '황돔', '옥돔', '자돔',
        '벤자리', '어름돔', '하스돔', '군평선이', '딱돔', '동갈돔', '줄도화돔', '킨메다이',
        '금눈돔', '달고기', '강담돔', '줄가자미', '노랑가오리', '홍어', '가오리', '상어',
        '고등어', '삼치', '참치', '다랑어', '가다랑어', '황다랑어', '눈다랑어', '참다랑어',
        '날개다랑어', '방어', '부시리', '잿방어', '전갱이', '병어', '덕대', '갈치',
        '꽁치', '학공치', '숭어', '가숭어', '문절망둑', '짱뚱어', '말뚝망둑', '망둑어',
        '베도라치', '뱀장어', '갯장어', '붕장어', '아나고', '곰치', '해마', '실고기',
        '날치', '빨간빨대물고기', '아귀', '씬벵이', '황아귀', '개구리사촌', '도롱뇽사촌', '맹꽁이',
        '두꺼비', '개구리', '청개구리', '참개구리', '옴개구리', '계곡산개구리', '북방산개구리', '한국산개구리',
        '황소개구리', '무당개구리', '남생이', '자라', '거북', '바다거북', '장수거북', '붉은바다거북',
        '도마뱀', '장지뱀', '줄장지뱀', '아리랑장지뱀', '표범장지뱀', '유혈목이', '누룩뱀', '무자치',
        '실뱀', '능구렁이', '대륙유혈목이', '비바리뱀', '살모사', '쇠살모사', '까치살모사', '논병아리',
        '뿔논병아리', '검은목논병아리', '귀뿔논병아리', '알바트로스', '풀마갈매기', '슴새', '바다제비',
        '사다새', '가마우지', '쇠가마우지', '민물가마우지', '해오라기', '검은댕기해오라기', '황로', '쇠백로',
        '중백로', '대백로', '중대백로', '노랑부리백로', '왜가리', '붉은왜가리', '황새', '먹황새',
        '따오기', '저어새', '노랑부리저어새', '혹고니', '고니', '큰고니', '흑기러기', '회색기러기',
        '쇠기러기', '큰소리기러기', '큰기러기', '개리', '가창오리', '혹부리오리', '원앙', '쇠오리',
        '청머리오리', '홍머리오리', '청둥오리', '흰뺨검둥오리', '넓적부리오리', '고방오리', '흰죽지',
        '댕기흰죽지', '검은머리흰죽지', '바다비오리', '비오리', '검독수리', '참수리', '흰꼬리수리', '벌매',
        '솔개', '물수리', '말똥가리', '털발말똥가리', '잿빛개구리매', '개구리매', '알락개구리매', '매',
        '새매', '조롱이', '참매', '황조롱이', '비둘기조롱이', '들꿩', '멧닭', '메추라기',
        '꿩', '두루미', '재두루미', '흑두루미', '검은목두루미', '쇠재두루미', '뜸부기', '쇠뜸부기',
        '물닭', '쇠물닭', '호사도요', '검은머리물떼새', '댕기물떼새', '개꿩', '검은가슴물떼새', '흰목물떼새',
        '꼬마물떼새', '흰눈썹물떼새', '물떼새', '맷도요', '멧도요', '도요', '마도요', '알락꼬리마도요',
        '중부리도요', '쇠청다리도요', '청다리도요', '빽빽도요', '알락도요', '부리긴도요', '큰부리도요', '검은머리갈매기',
        '쇠제비갈매기', '에위니아제비갈매기', '붉은부리큰제비갈매기', '검은등제비갈매기', '넓적부리도요', '노랑발도요', '송곳부리도요', '목도리도요',
        '누른도요', '매사촌', '큰매사촌', '항라머리검독수리', '말똥가리', '쇠재두루미', '시베리아흰두루미', '쇠뜸부기사촌',
        '한국뜸부기', '쇠뜸부기', '북방쇠종다리', '검은할미새사촌', '홍방울새', '쇠홍방울새', '방울새', '검은머리방울새',
        '재때까치', '붉은등때까치', '때까치', '물까마귀', '물까치', '까치', '까마귀', '갈까마귀',
        '떼까마귀', '큰부리까마귀', '잣까마귀', '휘파람새', '개개비', '산솔새', '솔새', '멧새',
        '노랑턱멧새', '검은머리쑥새', '검은머리쑥새', '쑥새', '꼬마쑥새', '노랑멧새', '되새', '방울새',
        '양진이', '밀화부리', '콩새', '참새', '섬참새', '찌르레기', '쇠찌르레기', '꾀꼬리',
        '검은머리꾀꼬리', '어치', '산까치', '바다직박구리', '직박구리', '딱새', '검은딱새', '노랑딱새',
        '흰눈썹황금새', '유리딱새', '박새', '진박새', '쇠박새', '곤줄박이', '스윈호오목눈이', '오목눈이',
        '종다리', '뿔종다리', '직박구리', '멧비둘기', '양비둘기', '흑비둘기', '녹색비둘기', '꿩비둘기',
        '뻐꾸기', '벙어리뻐꾸기', '두견이', '매부리뻐꾸기', '검은부리뻐꾸기', '올빼미', '수리부엉이', '부엉이',
        '솔부엉이', '쇠부엉이', '칡부엉이', '소쩍새', '큰소쩍새', '싹새', '밤새', '쏙독새',
        '칼새', '쇠칼새', '호반새', '물반새', '청호반새', '물청새', '팔색조', '후투티',
        '개미잡이', '오색딱다구리', '큰오색딱다구리', '청딱다구리', '까막딱다구리', '크낙새', '아리랑딱다구리', '제비',
        '귀제비', '털발제비', '백로', '두루미', '고니', '기러기', '오리', '매', '독수리',
        '부엉이', '올빼미', '비둘기', '까치', '까마귀', '참새', '제비', '꾀꼬리', '뻐꾸기',
        '딱다구리', '갈매기', '펭귄', '타조', '독수리', '콘도르', '벌새', '극락조', '공작',
        '앵무새', '원숭이', '침팬지', '고릴라', '오랑우탄', '여우원숭이', '나무늘보', '개미핥기', '아르마딜로',
        '토끼', '쥐', '다람쥐', '청설모', '하늘다람쥐', '두더지', '고슴도치', '박쥐', '고래',
        '돌고래', '범고래', '북극곰', '불곰', '반달가슴곰', '팬더', '너구리', '오소리', '수달',
        '족제비', '스컹크', '하이에나', '사자', '호랑이', '표범', '치타', '살쾡이', '고양이',
        '늑대', '여우', '코요테', '개', '바다표범', '바다사자', '물개', '바다코끼리', '매너티',
        '듀공', '코끼리', '매머드', '뿔말', '말', '당나귀', '얼룩말', '코뿔소', '맥',
        '멧돼지', '하마', '낙타', '라마', '기린', '사슴', '순록', '무스', '영양',
        '가젤', '소', '물소', '양', '염소', '산양', '캥거루', '코알라', '오리너구리',
        '두더지', '두꺼비', '개구리', '도롱뇽', '악어', '거북', '뱀', '도마뱀', '카멜레온',
        '이구아나', '상어', '가오리', '참치', '고등어', '연어', '장어', '메기', '잉어',
        '금붕어', '해마', '복어', '아귀', '문어', '오징어', '낙지', '꼴뚜기', '해파리',
        '불가사리', '성게', '해삼', '새우', '가재', '게', '소라', '조개', '굴',
        '가리비', '전복', '민달팽이', '군소', '갯민숭달팽이', '잠자리', '메뚜기',
        '귀뚜라미', '사마귀', '매미', '노린재', '딱정벌레', '풍뎅이', '무당벌레', '반딧불이', '하늘소',
        '벌', '개미', '파리', '모기', '벼룩', '이', '거미', '전갈', '진드기',
        '지네', '그리마', '노래기', '지렁이', '거머리', '플라나리아', '말미잘', '산호', '해면',
        '말똥가리', '참수리', '벌매', '검독수리', '흰꼬리수리', '잿빛개구리매', '개구리매', '새매',
        '조롱이', '참매', '황조롱이', '비둘기조롱이', '매', '수염상어', '고래상어', '환도상어',
        '악상어', '청상아리', '백상아리', '흉상어', '뱀상어', '귀상어', '전자리상어', '가오리',
        '노랑가오리', '매가오리', '만타가오리', '톱가오리', '은상어', '칠성상어', '두툽상어', '개상어',
        '곱상어', '돔발상어', '기름상어', '까치상어', '별상어', '매부리바다거북', '좁은띠큰바다뱀', '푸른바다거북',
        '짧은입두동가리돔', '참잔가시고기', '큰씬벵이', '검은눈띠망둑', '물방울바다뱀장어', '붉은동갈새우붙이망둑', '빗살아씨놀래기', '두점긴주둥이놀래기',
        '민무늬물수배기', '덕유모치', '진홍얼게비늘', '카이야꽃동멸', '왕연어', '검은꼬리파랑눈매퉁이', '브라운송어', '짧은털비늘베도라치',
        '가막청황문절', '동해물뱀', '흑전갱이', '민무늬물수배기', '점줄퉁돔', '황줄육각복', '흑점나비고기', '다섯줄색동놀래기',
        '민무늬둑중개', '벚꽃무늬둥글넙치', '흑상어', '별꽃곰치', '뿔베도라치', '노랑무늬양쥐돔', '추사어름돔', '일곱줄베도라치',
        '금강자가사리', '황줄전갱이', '긴머리달재'
    ]
    
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
        # 1. 마스터 딕셔너리 구축 (검색 속도 최적화를 위해 길이별 분리)
        self.master_lexicon = {
            "동물": self._build_sub_dict(self.ANIMAL_RAW),
            "스포츠": self._build_sub_dict(self.SPORTS_RAW),
            "감정 및 상태": {2: ['기쁨', '슬픔', '화남'], 3: ['배고픔'], 2: ['덥다', '춥다', '사랑', '놀람', '창피']},
            "행동 및 동사": {2: ['수영', '요리', '낚시', '운전', '청소', '공부'], 3: ['달리기', '자전거'], 4: ['사진찍기', '전화하기']},
            "날씨 및 자연": {1: ['해', '비', '눈', '달', '별', '꽃'], 2: ['번개', '바람', '구름', '나무']}
        }
    
    def _build_sub_dict(self, word_list):
        """단어 리스트를 글자 수별로 인덱싱"""
        sub_dict = {}
        for word in word_list:
            length = len(word.replace(" ", ""))
            # 8글자 이상은 8로 통합 관리
            key = length if length < 8 else 8
            sub_dict.setdefault(key, []).append(word)
        return sub_dict
    
    def match(self, partial: str, category: Optional[str] = None,
              syllable_count: Optional[int] = None) -> List[str]:
        """
        부분 입력(초성 포함)으로 단어 추천 (핵심 단어 우선)
        
        Args:
            partial: 현재까지 입력된 글자들 (완성형+초성)
            category: 카테고리
            syllable_count: 목표 글자수
        """
        if not category:
            return []
            
        # 해당 카테고리의 딕셔너리 가져오기
        full_category = category
        if category not in self.master_lexicon:
            for k in self.master_lexicon.keys():
                if category in k:
                    full_category = k
                    break
        
        target_dict = self.master_lexicon.get(full_category, {})
        
        # 후보군 수집
        candidates = []
        if syllable_count:
            # 1. 정확한 글자 수 매칭
            key = syllable_count if syllable_count < 8 else 8
            candidates.extend(target_dict.get(key, []))
            
            # 2. 유사 글자 수 (오타 가능성 고려, ±1글자) - 선택적
            # candidates.extend(target_dict.get(key-1, []))
            # candidates.extend(target_dict.get(key+1, []))
        else:
            for words in target_dict.values():
                candidates.extend(words)
        
        if not candidates:
            return []

        # 필터링 로직
        exact_matches = []
        fuzzy_matches = []
        
        # 입력값에서 초성 추출 (예: "토ㄲ" -> "ㅌㄲ")
        partial_cs = self._get_chosung_string(partial)
        
        for word in candidates:
            # 1. 정확히 시작하는 경우 (startswith)
            if word.startswith(partial):
                exact_matches.append(word)
                continue
                
            # 2. 초성만으로 시작하는 경우
            word_cs = self._get_chosung_string(word)
            if word_cs.startswith(partial_cs):
                fuzzy_matches.append(word)
                continue
                
            # 3. 유사 자음 매칭 ("토기" -> "토끼")
            # 글자 수가 같고, 유사도가 높으면 추가
            if len(partial) == len(word):
                if self._is_similar(partial, word):
                    fuzzy_matches.append(word)

        # 중복 제거 및 정렬
        # 우선순위: 1. 정확 매칭, 2. 핵심 단어(Core), 3. 나머지
        core_list = self.CORE_WORDS.get(category, [])
        if not core_list and full_category in self.CORE_WORDS:
            core_list = self.CORE_WORDS[full_category]
            
        # 결과 리스트 조합
        results = []
        
        # 1. Core Words 중 Exact Match
        results.extend(sorted([w for w in exact_matches if w in core_list]))
        # 2. 나머지 Exact Match
        results.extend(sorted([w for w in exact_matches if w not in core_list]))
        # 3. Core Words 중 Fuzzy Match
        results.extend(sorted([w for w in fuzzy_matches if w in core_list]))
        # 4. 나머지 Fuzzy Match
        results.extend(sorted([w for w in fuzzy_matches if w not in core_list]))
        
        # 중복 제거 (순서 유지)
        seen = set()
        final_results = []
        for r in results:
            if r not in seen:
                final_results.append(r)
                seen.add(r)
                
        return final_results[:10]  # 상위 10개 반환
    
    def _get_chosung_string(self, text: str) -> str:
        """문자열 전체의 초성을 추출"""
        result = ""
        for char in text:
            # 한글 완성형인 경우
            if '가' <= char <= '힣':
                cho_idx = (ord(char) - 0xAC00) // 588
                result += self._idx_to_chosung(cho_idx)
            # 자음만 있는 경우 (ㄱ~ㅎ)
            elif 'ㄱ' <= char <= 'ㅎ':
                result += char
            # 그 외 (공백 등)
            else:
                result += char
        return result

    def _idx_to_chosung(self, idx: int) -> str:
        CHO_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 
                    'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
        return CHO_LIST[idx] if 0 <= idx < len(CHO_LIST) else ''
    
    def _is_similar(self, partial: str, word: str) -> bool:
        """두 문자열이 유사한지 확인 (자음 혼동 허용)"""
        # 기존 유사도 로직 유지 (필요 시 사용)
        if len(partial) > len(word):
            return False
        
        for i, char in enumerate(partial):
            if i >= len(word):
                return False
            if char == word[i]:
                continue
            
            partial_cho = self._get_choseong(char)
            word_cho = self._get_choseong(word[i])
            
            if partial_cho and word_cho:
                similar = self.SIMILAR_CONSONANTS.get(partial_cho, [])
                if word_cho in similar or partial_cho == word_cho:
                    continue
            return False
        return True
    
    def _get_choseong(self, char: str) -> Optional[str]:
        """글자에서 초성 추출 (단일 문자용)"""
        CHO_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 
                    'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
        if char in CHO_LIST:
            return char
        code = ord(char)
        if 0xAC00 <= code <= 0xD7A3:
            cho_idx = (code - 0xAC00) // (21 * 28)
            return CHO_LIST[cho_idx]
        return None
    
    def get_exact_match(self, word: str, category: Optional[str] = None) -> bool:
        """정확히 일치하는 단어가 있는지 확인"""
        if category:
            # 카테고리 매핑
            full_cat = category
            if category not in self.master_lexicon:
                for k in self.master_lexicon.keys():
                    if category in k:
                        full_cat = k
                        break
            
            target_dict = self.master_lexicon.get(full_cat, {})
            # 전체 검색
            for w_list in target_dict.values():
                if word in w_list:
                    return True
        return False


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
    print("\n단어 매칭 테스트 (dic3,4 데이터 통합):")
    matcher = WordMatcher()
    
    # 1. 동물, 2글자, 'ㅌ' -> 토끼 (핵심 단어)
    print(f"  [동물/2글자/'ㅌ'] -> {matcher.match('ㅌ', '동물', 2)}")
    
    # 2. 스포츠, 2글자, 'ㄴ' -> 농구 (핵심 단어)
    print(f"  [스포츠/2글자/'ㄴ'] -> {matcher.match('ㄴ', '스포츠', 2)}")
    
    # 3. 동물, 3글자, 'ㅎ' -> 하마, 호랑이 등
    print(f"  [동물/3글자/'ㅎ'] -> {matcher.match('ㅎ', '동물', 3)}")