"""
한글 렌더링 모듈
OpenCV에서 한글을 표시하기 위한 PIL 기반 렌더러
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import os


class KoreanTextRenderer:
    """PIL을 사용한 한글 텍스트 렌더러"""
    
    def __init__(self, font_path: Optional[str] = None, default_size: int = 30):
        """
        Args:
            font_path: 한글 폰트 파일 경로 (None이면 시스템 폰트 탐색)
            default_size: 기본 폰트 크기
        """
        self.default_size = default_size
        self._pil_available = False
        self._font = None
        self._font_path = font_path
        
        self._init_pil()
    
    def _init_pil(self):
        """PIL 초기화 및 폰트 로드"""
        try:
            from PIL import Image, ImageDraw, ImageFont
            self._Image = Image
            self._ImageDraw = ImageDraw
            self._ImageFont = ImageFont
            self._pil_available = True
            
            # 폰트 찾기
            self._font_path = self._find_font()
            if self._font_path:
                self._font = ImageFont.truetype(self._font_path, self.default_size)
                print(f"[Font] 한글 폰트 로드: {self._font_path}")
            else:
                self._font = ImageFont.load_default()
                print("[Font] 기본 폰트 사용 (한글 제한적)")
                
        except ImportError:
            print("[Font] PIL 없음, OpenCV 기본 텍스트 사용")
            print("       설치: pip install Pillow")
            self._pil_available = False
    
    def _find_font(self) -> Optional[str]:
        """시스템에서 한글 폰트 찾기"""
        # 우선순위 폰트 목록
        font_candidates = [
            # Windows
            "C:/Windows/Fonts/malgun.ttf",      # 맑은 고딕
            "C:/Windows/Fonts/gulim.ttc",       # 굴림
            "C:/Windows/Fonts/batang.ttc",      # 바탕
            
            # macOS
            "/System/Library/Fonts/AppleSDGothicNeo.ttc",
            "/Library/Fonts/AppleGothic.ttf",
            "/System/Library/Fonts/Supplemental/AppleMyungjo.ttf",
            
            # Linux (Ubuntu/Debian)
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            
            # 현재 디렉토리
            "./fonts/NanumGothic.ttf",
            "./NanumGothic.ttf",
        ]
        
        for font_path in font_candidates:
            if os.path.exists(font_path):
                return font_path
        
        return None
    
    def put_text(self, frame: np.ndarray, text: str, 
                 position: Tuple[int, int], 
                 font_size: int = None,
                 color: Tuple[int, int, int] = (255, 255, 255),
                 bg_color: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
        """
        이미지에 한글 텍스트 추가
        
        Args:
            frame: OpenCV 이미지 (BGR)
            text: 표시할 텍스트
            position: (x, y) 좌표
            font_size: 폰트 크기 (None이면 기본값)
            color: 텍스트 색상 (BGR)
            bg_color: 배경 색상 (None이면 투명)
        
        Returns:
            텍스트가 추가된 이미지
        """
        if not self._pil_available:
            # PIL 없으면 OpenCV 기본 사용
            cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                       (font_size or self.default_size) / 30, color[::-1], 2)
            return frame
        
        # BGR -> RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = self._Image.fromarray(img_rgb)
        draw = self._ImageDraw.Draw(pil_img)
        
        # 폰트 크기 설정
        if font_size and font_size != self.default_size:
            try:
                font = self._ImageFont.truetype(self._font_path, font_size)
            except:
                font = self._font
        else:
            font = self._font
        
        # RGB로 색상 변환 (BGR -> RGB)
        rgb_color = (color[2], color[1], color[0])
        
        # 배경 그리기
        if bg_color:
            bbox = draw.textbbox(position, text, font=font)
            padding = 5
            bg_rgb = (bg_color[2], bg_color[1], bg_color[0])
            draw.rectangle(
                [bbox[0] - padding, bbox[1] - padding, 
                 bbox[2] + padding, bbox[3] + padding],
                fill=bg_rgb
            )
        
        # 텍스트 그리기
        draw.text(position, text, font=font, fill=rgb_color)
        
        # RGB -> BGR
        result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return result
    
    def put_text_centered(self, frame: np.ndarray, text: str,
                          center: Tuple[int, int],
                          font_size: int = None,
                          color: Tuple[int, int, int] = (255, 255, 255),
                          bg_color: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
        """중앙 정렬 텍스트"""
        if not self._pil_available:
            return self.put_text(frame, text, center, font_size, color, bg_color)
        
        # 텍스트 크기 계산
        if font_size and font_size != self.default_size:
            try:
                font = self._ImageFont.truetype(self._font_path, font_size)
            except:
                font = self._font
        else:
            font = self._font
        
        # PIL로 텍스트 크기 측정
        dummy_img = self._Image.new('RGB', (1, 1))
        dummy_draw = self._ImageDraw.Draw(dummy_img)
        bbox = dummy_draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # 중앙 정렬 좌표
        x = center[0] - text_width // 2
        y = center[1] - text_height // 2
        
        return self.put_text(frame, text, (x, y), font_size, color, bg_color)
    
    def get_text_size(self, text: str, font_size: int = None) -> Tuple[int, int]:
        """텍스트 크기 반환"""
        if not self._pil_available:
            return (len(text) * 15, 30)
        
        if font_size and font_size != self.default_size:
            try:
                font = self._ImageFont.truetype(self._font_path, font_size)
            except:
                font = self._font
        else:
            font = self._font
        
        dummy_img = self._Image.new('RGB', (1, 1))
        dummy_draw = self._ImageDraw.Draw(dummy_img)
        bbox = dummy_draw.textbbox((0, 0), text, font=font)
        
        return (bbox[2] - bbox[0], bbox[3] - bbox[1])


# 전역 인스턴스
_renderer = None

def get_renderer() -> KoreanTextRenderer:
    """싱글톤 렌더러 반환"""
    global _renderer
    if _renderer is None:
        _renderer = KoreanTextRenderer()
    return _renderer


def put_korean_text(frame: np.ndarray, text: str, 
                    position: Tuple[int, int],
                    font_size: int = 30,
                    color: Tuple[int, int, int] = (255, 255, 255),
                    bg_color: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
    """편의 함수: 한글 텍스트 추가"""
    return get_renderer().put_text(frame, text, position, font_size, color, bg_color)


# 테스트
if __name__ == "__main__":
    print("한글 렌더링 테스트")
    
    # 테스트 이미지 생성
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    img[:] = (50, 50, 50)
    
    renderer = KoreanTextRenderer()
    
    # 다양한 텍스트 테스트
    test_texts = [
        ("한글 테스트", (50, 50), 40, (255, 255, 255)),
        ("토끼 🐰", (50, 120), 35, (100, 255, 100)),
        ("자음: ㄱㄴㄷㄹㅁㅂㅅ", (50, 180), 30, (255, 200, 100)),
        ("모음: ㅏㅓㅗㅜㅡㅣ", (50, 230), 30, (100, 200, 255)),
        ("CBrain 스피드게임", (50, 300), 45, (255, 255, 0)),
    ]
    
    for text, pos, size, color in test_texts:
        img = renderer.put_text(img, text, pos, size, color)
    
    # 중앙 정렬 테스트
    img = renderer.put_text_centered(img, "중앙 정렬", (300, 370), 30, (255, 100, 100))
    
    cv2.imshow("Korean Text Test", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("테스트 완료!")
