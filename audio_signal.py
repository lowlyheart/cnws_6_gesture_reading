"""
오디오 신호 모듈
준비/실행 위상 전환 시 비프음 재생
"""

import numpy as np
import threading
import time
from typing import Optional


class AudioSignal:
    """오디오 신호 생성 및 재생"""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.sample_rate = 44100
        self._audio_available = False
        self._pyaudio = None
        self._stream = None
        
        if enabled:
            self._init_audio()
    
    def _init_audio(self):
        """오디오 시스템 초기화"""
        try:
            import pyaudio
            self._pyaudio = pyaudio.PyAudio()
            self._audio_available = True
            print("[Audio] PyAudio 초기화 성공")
        except ImportError:
            print("[Audio] PyAudio 없음, 오디오 비활성화")
            print("        설치: pip install pyaudio")
            self._audio_available = False
        except Exception as e:
            print(f"[Audio] 초기화 실패: {e}")
            self._audio_available = False
    
    def _generate_tone(self, frequency: float, duration: float, 
                       volume: float = 0.5) -> np.ndarray:
        """사인파 톤 생성"""
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        tone = np.sin(2 * np.pi * frequency * t) * volume
        
        # Fade in/out (클릭 방지)
        fade_samples = int(self.sample_rate * 0.01)
        tone[:fade_samples] *= np.linspace(0, 1, fade_samples)
        tone[-fade_samples:] *= np.linspace(1, 0, fade_samples)
        
        return (tone * 32767).astype(np.int16)
    
    def play_tone(self, frequency: float, duration: float, 
                  volume: float = 0.5, blocking: bool = False):
        """톤 재생"""
        if not self.enabled or not self._audio_available:
            return
        
        def _play():
            try:
                import pyaudio
                tone = self._generate_tone(frequency, duration, volume)
                
                stream = self._pyaudio.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=self.sample_rate,
                    output=True
                )
                stream.write(tone.tobytes())
                stream.stop_stream()
                stream.close()
            except Exception as e:
                print(f"[Audio] 재생 오류: {e}")
        
        if blocking:
            _play()
        else:
            thread = threading.Thread(target=_play, daemon=True)
            thread.start()
    
    def play_prep_signal(self):
        """준비 시작 신호 (낮은 톤)"""
        self.play_tone(440, 0.15, 0.3)  # A4, 짧게
    
    def play_exec_signal(self):
        """실행 시작 신호 (높은 톤)"""
        self.play_tone(880, 0.2, 0.4)  # A5, 조금 길게
    
    def play_confirm_signal(self):
        """입력 확정 신호 (상승 톤)"""
        self.play_tone(660, 0.1, 0.3)  # E5
        time.sleep(0.05)
        self.play_tone(880, 0.1, 0.3)  # A5
    
    def play_error_signal(self):
        """오류 신호 (하강 톤)"""
        self.play_tone(440, 0.15, 0.4)
        time.sleep(0.05)
        self.play_tone(330, 0.2, 0.4)
    
    def play_complete_signal(self):
        """완료 신호 (멜로디)"""
        notes = [523, 659, 784, 1047]  # C5, E5, G5, C6
        for note in notes:
            self.play_tone(note, 0.12, 0.3, blocking=True)
            time.sleep(0.02)
    
    def cleanup(self):
        """리소스 정리"""
        if self._pyaudio:
            self._pyaudio.terminate()


class SimpleBeep:
    """
    PyAudio 없이 시스템 비프 사용 (대체 방법)
    Windows: winsound, Mac/Linux: print('\a')
    """
    
    def __init__(self):
        self._platform = self._detect_platform()
    
    def _detect_platform(self) -> str:
        import platform
        system = platform.system().lower()
        if system == 'windows':
            return 'windows'
        elif system == 'darwin':
            return 'mac'
        else:
            return 'linux'
    
    def beep(self, frequency: int = 440, duration: int = 200):
        """비프음 재생 (플랫폼별)"""
        try:
            if self._platform == 'windows':
                import winsound
                winsound.Beep(frequency, duration)
            else:
                # Mac/Linux: 터미널 벨
                print('\a', end='', flush=True)
        except Exception:
            pass
    
    def play_prep_signal(self):
        self.beep(440, 100)
    
    def play_exec_signal(self):
        self.beep(880, 150)
    
    def play_confirm_signal(self):
        self.beep(660, 100)
    
    def play_error_signal(self):
        self.beep(330, 200)
    
    def play_complete_signal(self):
        for freq in [523, 659, 784]:
            self.beep(freq, 100)
            time.sleep(0.05)


def get_audio_system(use_pyaudio: bool = True) -> object:
    """사용 가능한 오디오 시스템 반환"""
    if use_pyaudio:
        audio = AudioSignal(enabled=True)
        if audio._audio_available:
            return audio
    
    print("[Audio] SimpleBeep 사용 (시스템 비프)")
    return SimpleBeep()


# 테스트
if __name__ == "__main__":
    print("오디오 시스템 테스트")
    
    audio = get_audio_system(use_pyaudio=True)
    
    print("1. 준비 신호")
    audio.play_prep_signal()
    time.sleep(0.5)
    
    print("2. 실행 신호")
    audio.play_exec_signal()
    time.sleep(0.5)
    
    print("3. 확정 신호")
    audio.play_confirm_signal()
    time.sleep(0.5)
    
    print("4. 오류 신호")
    audio.play_error_signal()
    time.sleep(0.5)
    
    print("5. 완료 신호")
    audio.play_complete_signal()
    
    print("\n테스트 완료!")
